import argparse
from time import time
from tqdm import tqdm
import torch
import torchvision as tv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.library import qres34m


@torch.no_grad()
def speedtest(model, image_path, repeat=10):
    device = next(model.parameters()).device

    encode_time = 0
    decode_time = 0
    for _ in tqdm(range(repeat)):
        im = tv.io.read_image(image_path).unsqueeze_(0).float().div_(255.0).to(device=device)

        t_start = time()
        compressed_obj = model.compress(im)
        t_enc_finish = time()
        output = model.decompress(compressed_obj)
        t_dec_finish = time()

        encode_time += (t_enc_finish - t_start)
        decode_time += (t_dec_finish - t_enc_finish)

    enc_time = encode_time / repeat
    dec_time = decode_time / repeat
    return enc_time, dec_time


@torch.no_grad()
def main(args):
    device = torch.device(args.device)

    model = qres34m(lmb=1024)
    msd = torch.load(args.weights_path)['model']
    model.load_state_dict(msd)

    model = model.to(device=device)
    model.eval()
    model.compress_mode()

    _ = speedtest(model, args.impath, repeat=1) # warm up
    enc_time, dec_time = speedtest(model, args.impath, repeat=10)
    print(f'{type(model)}, device={device}, {args.impath}')
    print(f'encode time={enc_time:.3f}s, decode time={dec_time:.3f}s')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-i', '--impath', type=str, default='d:/datasets/improcessing/kodak/kodim01.png')
    parser.add_argument('-w', '--weights_path', type=str,
                        default='checkpoints/qres34m/lmb1024/last_ema.pt')
    args = parser.parse_args()
    main(args)
