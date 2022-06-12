import sys
import pickle
from collections import OrderedDict
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torch.distributions as td
import torchvision as tv

from compressai.entropy_models import GaussianConditional


def get_object_size(obj, unit='bits'):
    assert unit == 'bits'
    return sys.getsizeof(pickle.dumps(obj)) * 8


def deconv(in_ch, out_ch, kernel_size=5, stride=2, zero_weights=False):
    conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              output_padding=stride - 1, padding=kernel_size // 2)
    if zero_weights:
        conv.weight.data.mul_(0.0)
    return conv

def subpix_conv(in_ch, out_ch, up_rate=2):
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch * up_rate ** 2, kernel_size=1, padding=0),
        nn.PixelShuffle(up_rate)
    )
    return conv

def get_conv(in_ch, out_ch, kernel_size, stride, padding, zero_bias=True, zero_weights=False):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    if zero_bias:
        conv.bias.data.mul_(0.0)
    if zero_weights:
        conv.weight.data.mul_(0.0)
    return conv

def get_3x3(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 3, 1, 1, zero_bias, zero_weights)

def get_1x1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 1, 1, 0, zero_bias, zero_weights)


def gaussian_log_prob_mass(mean, log_scale, x, bin_size=0.01, prob_clamp=1e-5):
    gaussian = td.Normal(mean, torch.exp(log_scale))
    prob_mass = gaussian.cdf(x + 0.5*bin_size) - gaussian.cdf(x - 0.5*bin_size)
    # _counts = [(prob_mass <= bound).sum().item() for bound in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0)]

    log_prob = torch.where(
        prob_mass > prob_clamp,
        torch.log(prob_mass.clamp(min=1e-8)),
        gaussian.log_prob(x) + math.log(bin_size)
    )
    # return log_prob, _counts
    return log_prob


class GaussianNLLOutputNet(nn.Module):
    def __init__(self, conv_mean, conv_scale, bin_size=1/127.5):
        super().__init__()
        self.conv_mean  = conv_mean
        self.conv_scale = conv_scale
        self.bin_size = bin_size
        self.loss_name = 'nll'

    @torch.autocast('cuda', enabled=False)
    def forward_loss(self, feature, x_tgt):
        feature = feature.float()
        p_mean = self.conv_mean(feature)
        p_logscale = self.conv_scale(feature)
        p_logscale = tnf.softplus(p_logscale + 16) - 16 # logscale lowerbound
        log_prob = gaussian_log_prob_mass(p_mean, p_logscale, x_tgt, bin_size=self.bin_size)
        assert log_prob.shape == x_tgt.shape
        nll = -log_prob.mean(dim=(1,2,3)) # BCHW -> (B,)
        return nll, p_mean

    def mean(self, feature):
        p_mean = self.conv_mean(feature)
        return p_mean

    def sample(self, feature, mode='continuous', temprature=None):
        p_mean = self.conv_mean(feature)
        p_logscale = self.conv_scale(feature)
        p_scale = torch.exp(p_logscale)
        if temprature is not None:
            p_scale = p_scale * temprature

        if mode == 'continuous':
            samples = p_mean + p_scale * torch.randn_like(p_mean)
        elif mode == 'discrete':
            raise NotImplementedError()
        else:
            raise ValueError()
        return samples

    def update(self):
        self.discrete_gaussian = GaussianConditional(None, scale_bound=0.11)
        device = next(self.parameters()).device
        self.discrete_gaussian = self.discrete_gaussian.to(device=device)
        lower = self.discrete_gaussian.lower_bound_scale.bound.item()
        # max_scale = self.max_scale.item()
        max_scale = 20
        scale_table = torch.exp(torch.linspace(math.log(lower), math.log(max_scale), steps=128))
        updated = self.discrete_gaussian.update_scale_table(scale_table)
        self.discrete_gaussian.update()

    def _preapre_codec(self, feature, x=None):
        assert not feature.requires_grad
        pm = self.conv_mean(feature)
        pm = torch.round(pm * 127.5 + 127.5) / 127.5 - 1 # workaround to make sure lossless
        plogv = self.conv_scale(feature)
        # scale (-1,1) range to (-127.5, 127.5) range
        pm = pm / self.bin_size
        plogv = plogv - math.log(self.bin_size)
        if x is not None:
            x = x / self.bin_size
        return pm, plogv, x

    def compress(self, feature, x):
        pm, plogv, x = self._preapre_codec(feature, x)
        # compress
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        strings = self.discrete_gaussian.compress(x, indexes, means=pm)
        return strings

    def decompress(self, feature, strings):
        pm, plogv, _ = self._preapre_codec(feature)
        # decompress
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        x_hat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        x_hat = x_hat * self.bin_size
        return x_hat


class MSEOutputNet(nn.Module):
    def __init__(self, in_ch, x_ch, mse_lmb, kernel_size=5, up_stride=1, use_conv=True):
        super().__init__()
        if not use_conv:
            self.conv_mean = nn.Identity()
        elif up_stride == 1:
            self.conv_mean  = get_conv(in_ch, x_ch, kernel_size, stride=1, padding=(kernel_size-1)//2)
            # self.conv_scale = None
        elif up_stride > 1:
            self.conv_mean  = deconv(in_ch, x_ch, kernel_size, up_stride)
            # self.conv_scale = None
        else:
            raise ValueError(f'Invalid up_stride={up_stride}')
        self.mse_lmb = float(mse_lmb)
        self.loss_name = 'mse'

    def forward_loss(self, feature, x_tgt):
        x_hat = self.conv_mean(feature)
        assert x_hat.shape == x_tgt.shape
        mse = tnf.mse_loss(x_hat, x_tgt, reduction='none').mean(dim=(1,2,3))
        loss = mse * self.mse_lmb
        return loss, x_hat

    def mean(self, feature, temprature=None):
        x_hat = self.conv_mean(feature)
        return x_hat
    sample = mean


class VDBlock(nn.Module):
    """ Adapted from VDVAE (https://github.com/openai/vdvae)
    - Paper: Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images
    - arxiv: https://arxiv.org/abs/2011.10650
    """
    def __init__(self, in_ch, hidden_ch=None, out_ch=None, residual=True,
                 use_3x3=True, zero_last=False):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or round(in_ch * 0.25)
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.residual = residual
        self.c1 = get_1x1(in_ch, hidden_ch)
        self.c2 = get_3x3(hidden_ch, hidden_ch) if use_3x3 else get_1x1(hidden_ch, hidden_ch)
        self.c3 = get_3x3(hidden_ch, hidden_ch) if use_3x3 else get_1x1(hidden_ch, hidden_ch)
        self.c4 = get_1x1(hidden_ch, out_ch, zero_weights=zero_last)

    def residual_scaling(self, N):
        # This residual scaling improves stability and performance with many layers
        # https://arxiv.org/pdf/2011.10650.pdf, Appendix Table 3
        self.c4.weight.data.mul_(math.sqrt(1 / N))

    def forward(self, x):
        xhat = self.c1(tnf.gelu(x))
        xhat = self.c2(tnf.gelu(xhat))
        xhat = self.c3(tnf.gelu(xhat))
        xhat = self.c4(tnf.gelu(xhat))
        out = (x + xhat) if self.residual else xhat
        return out

class VDBlockPatchDown(VDBlock):
    def __init__(self, in_ch, out_ch, down_rate=2):
        super().__init__(in_ch, residual=True)
        self.downsapmle = get_conv(in_ch, out_ch, kernel_size=down_rate, stride=down_rate, padding=0)

    def forward(self, x):
        x = super().forward(x)
        out = self.downsapmle(x)
        return out


from timm.models.convnext import ConvNeXtBlock
class MyConvNeXtBlock(ConvNeXtBlock):
    def __init__(self, dim, mlp_ratio=2, kernel_size=7):
        super().__init__(dim, drop_path=0., ls_init_value=1e-6,
                         conv_mlp=False, mlp_ratio=mlp_ratio, norm_layer=None)
        p = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p, groups=dim)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class MyConvNeXtPatchDown(MyConvNeXtBlock):
    def __init__(self, in_ch, out_ch, down_rate=2, mlp_ratio=2, kernel_size=7):
        super().__init__(in_ch, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
        self.downsapmle = get_conv(in_ch, out_ch, kernel_size=down_rate, stride=down_rate, padding=0)

    def forward(self, x):
        x = super().forward(x)
        out = self.downsapmle(x)
        return out


class BottomUpEncoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)
        self._init_weights()

    def _init_weights(self):
        # This residual scaling improves stability and performance with many layers
        # as shown in the Appendix (Table 3), https://arxiv.org/pdf/2011.10650.pdf
        total_blocks = len([b for b in self.enc_blocks if hasattr(b, 'residual_scaling')])
        for block in self.enc_blocks:
            if hasattr(block, 'residual_scaling'):
                block.residual_scaling(total_blocks)

    def forward(self, x):
        feature = x
        enc_features = dict()
        for i, block in enumerate(self.enc_blocks):
            feature = block(feature)
            res = int(feature.shape[2])
            enc_features[res] = feature
        return enc_features


class QLatentBlockBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels:  int
        self.out_channels: int

        self.resnet_front: nn.Module
        self.resnet_end:   nn.Module
        self.posterior:    nn.Module
        self.prior:        nn.Module
        self.z_proj:       nn.Module
        self.discrete_gaussian = GaussianConditional(None, scale_bound=0.11)

    def residual_scaling(self, N):
        raise NotImplementedError()

    def transform_prior(self, feature):
        feature = self.resnet_front(feature)
        # prior p(z)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        return feature, pm, plogv

    def forward_train(self, feature, enc_feature, get_latents=False):
        feature, pm, plogv = self.transform_prior(feature)
        # posterior q(z|x)
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        qm = self.posterior(torch.cat([feature, enc_feature], dim=1))
        # compute KL divergence
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = gaussian_log_prob_mass(pm, plogv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            pv = torch.exp(plogv)
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        feature = feature + self.z_proj(z_sample)
        feature = self.resnet_end(feature)
        if get_latents:
            return feature, dict(z=z_sample.detach(), kl=kl)
        return feature, dict(kl=kl)

    def forward_uncond(self, feature, t=1.0, latent=None, paint_box=None):
        feature, pm, plogv = self.transform_prior(feature)
        pv = torch.exp(plogv)
        pv = pv * t # modulate the prior scale by the temperature t
        if latent is None: # normal case. Just sampling.
            z = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
        elif paint_box is not None: # partial sampling for inpainting
            nB, zC, zH, zW = latent.shape
            if min(zH, zW) == 1:
                z = latent
            else:
                x1, y1, x2, y2 = paint_box
                h_slice = slice(round(y1*zH), round(y2*zH))
                w_slice = slice(round(x1*zW), round(x2*zW))
                z_sample = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
                z_patch = z_sample[:, :, h_slice, w_slice]
                z = torch.clone(latent)
                z[:, :, h_slice, w_slice] = z_patch
            debug = 1
        else: # if `latent` is provided and `paint_box` is not provided, directly use it.
            assert pm.shape == latent.shape
            z = latent
        feature = feature + self.z_proj(z)
        feature = self.resnet_end(feature)
        return feature

    def update(self):
        min_scale = 0.1
        max_scale = 20
        log_scales = torch.linspace(math.log(min_scale), math.log(max_scale), steps=64)
        scale_table = torch.exp(log_scales)
        updated = self.discrete_gaussian.update_scale_table(scale_table)
        self.discrete_gaussian.update()

    def compress(self, feature, enc_feature):
        feature, pm, plogv = self.transform_prior(feature)
        # posterior q(z|x)
        qm = self.posterior(torch.cat([feature, enc_feature], dim=1))
        # compress
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
        zhat = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
        # add the new information to feature
        feature = feature + self.z_proj(zhat)
        feature = self.resnet_end(feature)
        return feature, strings

    def decompress(self, feature, strings):
        feature, pm, plogv = self.transform_prior(feature)
        # decompress
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        zhat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        # add the new information to feature
        feature = feature + self.z_proj(zhat)
        feature = self.resnet_end(feature)
        return feature


class QLatentBlockX(QLatentBlockBase):
    def __init__(self, width, zdim, enc_width=None, kernel_size=7):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_width = enc_width or width
        hidden = int(max(width, enc_width) * 0.25)
        concat_ch = (width * 2) if enc_width is None else (width + enc_width)
        use_3x3 = (kernel_size >= 3)
        self.resnet_front = MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.resnet_end   = MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.posterior = VDBlock(concat_ch, hidden, zdim, residual=False, use_3x3=use_3x3)
        self.prior     = VDBlock(width, hidden, zdim * 2, residual=False, use_3x3=use_3x3,
                                 zero_last=True)
        self.z_proj = nn.Sequential(
            get_3x3(zdim, hidden//2) if use_3x3 else get_1x1(zdim, hidden//2),
            nn.GELU(),
            get_1x1(hidden//2, width),
        )

    def residual_scaling(self, N):
        self.z_proj[2].weight.data.mul_(math.sqrt(1 / 3*N))
        # self.z_proj[2].weight.data.mul_(math.sqrt(1 / N))


class QLatentBlockVD(QLatentBlockBase):
    def __init__(self, width, zdim, enc_width=None, use_3x3=True):
        super().__init__()
        self.in_channels = width
        self.out_channels = width

        enc_width = enc_width or width
        hidden = int(max(width, enc_width) * 0.25)
        concat_ch = (width * 2) if enc_width is None else (width + enc_width)
        self.resnet_front = VDBlock(width, hidden, width, use_3x3=use_3x3, zero_last=True)
        self.resnet_end   = VDBlock(width, hidden, width, use_3x3=use_3x3)
        self.posterior = VDBlock(concat_ch, hidden, zdim, residual=False, use_3x3=use_3x3)
        self.prior     = VDBlock(width, hidden, zdim * 2, residual=False, use_3x3=use_3x3,
                                 zero_last=True)
        self.z_proj = nn.Sequential(
            get_3x3(zdim, hidden//2) if use_3x3 else get_1x1(zdim, hidden//2),
            nn.GELU(),
            get_1x1(hidden//2, width),
        )

    def residual_scaling(self, N):
        self.z_proj[2].weight.data.mul_(math.sqrt(1 / (2*N)))
        self.resnet_end.residual_scaling(2*N)


class TopDownDecoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.dec_blocks = nn.ModuleList(blocks)

        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self._init_weights()

    def _init_weights(self):
        total_blocks = len([1 for b in self.dec_blocks if hasattr(b, 'residual_scaling')])
        for block in self.dec_blocks:
            if hasattr(block, 'residual_scaling'):
                block.residual_scaling(total_blocks)

    def forward(self, enc_features, get_latents=False):
        stats = []
        min_res = min(enc_features.keys())
        feature = self.bias.expand(enc_features[min_res].shape)
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_train'):
                res = int(feature.shape[2])
                f_enc = enc_features[res]
                feature, block_stats = block.forward_train(feature, f_enc, get_latents=get_latents)
                stats.append(block_stats)
            else:
                feature = block(feature)
        return feature, stats

    def forward_uncond(self, nhw_repeat=(1, 1, 1), t=1.0):
        nB, nH, nW = nhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, t)
            else:
                feature = block(feature)
        return feature

    def forward_with_latents(self, latents, nhw_repeat=None, t=1.0, paint_box=None):
        if nhw_repeat is None:
            nB, _, nH, nW = latents[0].shape
            feature = self.bias.expand(nB, -1, nH, nW)
        else: # use defined
            nB, nH, nW = nhw_repeat
            feature = self.bias.expand(nB, -1, nH, nW)
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, t, latent=latents[idx], paint_box=paint_box)
                idx += 1
            else:
                feature = block(feature)
        return feature

    def update(self):
        for block in self.dec_blocks:
            if hasattr(block, 'update'):
                block.update()

    def compress(self, enc_features):
        # assert len(self.bias_xs) == 1
        min_res = min(enc_features.keys())
        feature = self.bias.expand(enc_features[min_res].shape)
        strings_all = []
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'compress'):
                # res = block.up_rate * feature.shape[2]
                res = feature.shape[2]
                f_enc = enc_features[res]
                feature, strs_batch = block.compress(feature, f_enc)
                strings_all.append(strs_batch)
            else:
                feature = block(feature)
        return strings_all, feature

    def decompress(self, compressed_object: list):
        # assert len(self.bias_xs) == 1
        smallest_shape = compressed_object[-1]
        feature = self.bias.expand(smallest_shape)
        # assert len(compressed_object) == len(self.dec_blocks)
        str_i = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'decompress'):
                strs_batch = compressed_object[str_i]
                str_i += 1
                feature = block.decompress(feature, strs_batch)
            else:
                feature = block(feature)
        assert str_i == len(compressed_object) - 1, f'decoded={str_i}, len={len(compressed_object)}'
        return feature


class HierarchicalVAE(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = BottomUpEncoder(blocks=config.pop('enc_blocks'))
        self.decoder = TopDownDecoder(blocks=config.pop('dec_blocks'))
        self.out_net = config.pop('out_net')

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self._stats_log = dict()
        self._log_images = config.get('log_images', None)
        self._log_smpl_k = [1, 2]
        self._flops_mode = False
        self.compressing = False

    def preprocess_input(self, im: torch.Tensor):
        """ Shift and scale the input image

        Args:
            im (torch.Tensor): a batch of images, values should be between (0, 1)
        """
        if not self._flops_mode:
            assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = (im + self.im_shift) * self.im_scale
        return x

    def process_output(self, x):
        im = x * 0.5 + 0.5
        return im

    def preprocess_target(self, im: torch.Tensor):
        if not self._flops_mode:
            assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = (im - 0.5) * 2.0
        return x

    def forward(self, im, return_rec=False):
        x = self.preprocess_input(im)
        x_target = self.preprocess_target(im)

        enc_features = self.encoder(x)
        feature, stats_all = self.decoder(enc_features)
        out_loss, x_hat = self.out_net.forward_loss(feature, x_target)

        if self._flops_mode: # testing flops
            return x_hat

        # ================ Training ================
        nB, imC, imH, imW = im.shape # batch, channel, height, width
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = imC * imH * imW
        kl = sum(kl_divergences) / ndims
        loss = (kl + out_loss).mean(0) # rate + distortion

        # ================ Logging ================
        with torch.no_grad():
            nats_per_dim = kl.detach().cpu().mean(0).item()
            im_hat = self.process_output(x_hat.detach())
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            # logging
            kls = torch.stack([kl.mean(0) / ndims for kl in kl_divergences], dim=0)
            bpdim = kls * self.log2_e
            mode = 'train' if self.training else 'eval'
            self._stats_log[f'{mode}_bpdim'] = bpdim.tolist()
            self._stats_log[f'{mode}_bppix'] = (bpdim * imC).tolist()
            channel_bpps = [stat['kl'].sum(dim=(2,3)).mean(0).cpu() / (imH * imW) for stat in stats_all]
            self._stats_log[f'{mode}_channels'] = [(bpps*self.log2_e).tolist() for bpps in channel_bpps]
            debug = 1

        stats = OrderedDict()
        stats['loss']  = loss
        stats['kl']    = nats_per_dim
        stats[self.out_net.loss_name] = out_loss.detach().cpu().mean(0).item()
        stats['bppix'] = nats_per_dim * self.log2_e * imC
        stats['psnr']  = psnr
        if return_rec:
            stats['im_hat'] = im_hat
        return stats

    def forward_eval(self, im):
        nB, imC, imH, imW = im.shape
        stats = self.forward(im)
        if self.compressing:
            compressed_object = self.compress(im)
            num_bits = get_object_size(compressed_object)
            im_hat = self.decompress(compressed_object)
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            nB, imC, imH, imW = im.shape
            stats['real-bppix'] = num_bits / (nB * imH * imW)
            if hasattr(self.out_net, 'compress'): # lossless compression
                assert torch.equal(im_hat.mul(255).round(), im.mul(255).round())
                stats['real-mse'] = im_mse.detach().item()
            else: # loss compression
                stats['real-psnr'] = -10 * math.log10(im_mse.detach().item())
        return stats

    def uncond_sample(self, nhw_repeat, temprature=1.0):
        """ unconditionally sample, ie, generate new images

        Args:
            nhw_repeat (tuple): repeat the initial constant feature n,h,w times
            temprature (float): temprature
        """
        feature = self.decoder.forward_uncond(nhw_repeat, t=temprature)
        x_samples = self.out_net.sample(feature, temprature=temprature)
        im_samples = self.process_output(x_samples)
        return im_samples

    def cond_sample(self, latents, nhw_repeat=None, temprature=1.0, paint_box=None):
        """ conditional sampling with latents

        Args:
            latents (torch.Tensor): latent variables
            nhw_repeat (tuple): repeat the constant n,h,w times
            temprature (float): temprature
            paint_box (tuple of floats): (x1,y1,x2,y2), in 0-1 range
        """
        feature = self.decoder.forward_with_latents(latents, nhw_repeat, t=temprature, paint_box=paint_box)
        x_samples = self.out_net.sample(feature, temprature=temprature)
        im_samples = self.process_output(x_samples)
        return im_samples

    def forward_get_latents(self, im):
        """ forward pass and return all the latent variables
        """
        x = self.preprocess_input(im)
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    def inpaint(self, im, paint_box, steps=1, temprature=1.0):
        nB, imC, imH, imW = im.shape
        x1, y1, x2, y2 = paint_box
        h_slice = slice(round(y1*imH), round(y2*imH))
        w_slice = slice(round(x1*imW), round(x2*imW))
        im_input = im.clone()
        for i in range(steps):
            stats_all = self.forward_get_latents(im_input)
            latents = [st['z'] for st in stats_all]
            im_sample = self.cond_sample(latents, temprature=temprature, paint_box=paint_box)
            torch.clamp_(im_sample, min=0, max=1)
            im_input = im.clone()
            im_input[:, :, h_slice, w_slice] = im_sample[:, :, h_slice, w_slice]
        return im_sample

    def study(self, save_dir):
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir(parents=False)

        device = next(self.parameters()).device
        # unconditional samples
        for k in self._log_smpl_k:
            num = 6
            im_samples = self.uncond_sample(nhw_repeat=(num,k,k))
            save_path = save_dir / f'samples_k{k}_hw{im_samples.shape[2]}.png'
            tv.utils.save_image(im_samples, fp=save_path, nrow=math.ceil(num**0.5))
        # reconstructions
        if self._log_images is None:
            img_names = {
                32:  ['gun128.png', 'butterfly64.png', 'cat64.png', 'collie32.png'],
                64:  ['gun128.png', 'butterfly64.png', 'cat64.png'],
                128: ['cactus.png', 'zebra256.png', 'gun128.png'],
                256: ['cactus.png', 'zebra256.png']
            }
            img_names = img_names[self.max_stride]
        else:
            img_names = self._log_images
        for imname in img_names:
            impath = f'images/{imname}'
            im = tv.io.read_image(impath).unsqueeze_(0).float().div_(255.0).to(device=device)
            stats = self.forward(im, return_rec=True)
            to_save = torch.cat([im, stats['im_hat']], dim=0)
            tv.utils.save_image(to_save, fp=save_dir / imname)
        # bits per layer logging
        for key, vlist in self._stats_log.items():
            if isinstance(vlist[0], (float, int)):
                with open(save_dir / f'{key}.txt', 'a') as f:
                    print(''.join([f'{a:<7.4f} ' for a in vlist]), file=f)
            elif isinstance(vlist[0], list):
                with open(save_dir / f'{key}.txt', 'w') as f:
                    for line in vlist:
                        print(''.join([f'{a:<7.4f} ' for a in line]), file=f)
            else:
                raise NotImplementedError(f'vlist: {type(vlist)}, {vlist}')

    def compress_mode(self, mode=True):
        if mode:
            self.decoder.update()
            if hasattr(self.out_net, 'compress'):
                self.out_net.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im):
        x = self.preprocess_input(im)
        enc_features = self.encoder(x)
        strings_all, feature = self.decoder.compress(enc_features)
        min_res = min(enc_features.keys())
        strings_all.append(tuple(enc_features[min_res].shape))
        if hasattr(self.out_net, 'compress'): # lossless compression
            x_tgt = self.preprocess_target(im)
            final_str = self.out_net.compress(feature, x_tgt)
            strings_all.append(final_str)
        return strings_all

    @torch.no_grad()
    def decompress(self, compressed_object):
        if hasattr(self.out_net, 'compress'): # lossless compression
            feature = self.decoder.decompress(compressed_object[:-1])
            x_hat = self.out_net.decompress(feature, compressed_object[-1])
        else: # lossy compression
            feature = self.decoder.decompress(compressed_object)
            x_hat = self.out_net.mean(feature)
        im_hat = self.process_output(x_hat)
        return im_hat
