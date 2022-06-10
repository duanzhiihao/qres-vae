import argparse
from time import time
from tqdm import tqdm
import torch
import torchvision as tv

from compressai.zoo.image import bmshj2018_factorized, mbt2018_mean, mbt2018, cheng2020_anchor

from mycv.paths import IMPROC_DIR
image_path = str(IMPROC_DIR / 'kodak/kodim01.png')


@torch.no_grad()
def speedtest(model, repeat=10):
    device = next(model.parameters()).device

    encode_time = 0
    decode_time = 0
    for _ in tqdm(range(repeat)):
        im = tv.io.read_image(image_path).unsqueeze_(0).float().div_(255.0).to(device=device)

        t_start = time()
        compressed_obj = model.compress(im)
        t_enc_finish = time()
        output = model.decompress(compressed_obj['strings'], compressed_obj['shape'])
        t_dec_finish = time()

        encode_time += (t_enc_finish - t_start)
        decode_time += (t_dec_finish - t_enc_finish)

    enc_time = encode_time / repeat
    dec_time = decode_time / repeat
    return enc_time, dec_time


@torch.no_grad()
def main(args):
    device = torch.device(args.device)

    for model in [
        bmshj2018_factorized(8, metric='mse', pretrained=True),
        bmshj2018_factorized(8, metric='mse', pretrained=True),
        mbt2018_mean(8, metric='mse', pretrained=True),
        mbt2018_mean(8, metric='mse', pretrained=True),
        mbt2018(8, metric='mse', pretrained=True),
        mbt2018(8, metric='mse', pretrained=True),
        cheng2020_anchor(6, metric='mse', pretrained=True),
        cheng2020_anchor(6, metric='mse', pretrained=True),
        mbt2018(8, metric='mse', pretrained=True),
    ]:
        model = model.to(device=device)
        model.eval()
        model.update()

        _ = speedtest(model, repeat=1) # warm up
        enc_time, dec_time = speedtest(model, repeat=10)
        print(f'{type(model)}, device={device}, {image_path}')
        print(f'encode time={enc_time:.3f}s, decode time={dec_time:.3f}s')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
