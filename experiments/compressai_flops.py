import torch
import torch.nn as nn

from mycv.paths import MYCV_DIR, IMPROC_DIR
from mycv.models.nlaic.nlaic import NLAIC
from mycv.external.compressai.zoo import _load_model
import mycv.utils.torch_utils as mytu


def _get_nlaic():
    model = NLAIC(enable_bpp=False)
    weights_path = MYCV_DIR / 'weights/nlaic/nlaic_mse1600.pt'
    mytu.load_partial(model, weights_path)
    return model

def _get_tic():
    q = 8
    from mycv.external.tic.tic_ctx import TIC
    model = TIC()
    weights_path = MYCV_DIR / f'weights/tic/{q}.pt'
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.g_a
        self.decoder = model.g_s

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat


@torch.no_grad()
def main():
    model_list = [
        # _load_model('bmshj2018-factorized', 'mse', 8, pretrained=True),
        # _load_model('mbt2018-mean', 'mse', 8, pretrained=True),
        _load_model('mbt2018', 'mse', 8, pretrained=True),
        # _load_model('cheng2020-anchor', 'mse', 6, pretrained=True),
        # _get_nlaic(),
        # _get_tic()
    ]
    for model in model_list:
        # _flops_model = Wrapper(model)
        # _flops_model.eval()
        # mytu.model_benchmark(model, input_shape=(3,256,256), n_cpu=20, n_cuda=100)
        mytu.flops_benchmark(model, input_shape=(3,256,256))

    debug = 1


if __name__ == '__main__':
    main()
