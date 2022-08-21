from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from PIL import Image
import json
import argparse
import math
import torch
import torchvision.transforms.functional as tvf

from models.library import get_model_func


@torch.no_grad()
def evaluate_model(model, dataset_root):
    tmp_bit_path = Path('tmp.bits')

    img_paths = list(Path(dataset_root).rglob('*.*'))
    img_paths.sort()
    pbar = tqdm(img_paths)
    accumulated_stats = defaultdict(float)
    for impath in pbar:
        model.compress_file(impath, tmp_bit_path)
        num_bits = tmp_bit_path.stat().st_size * 8
        fake = model.decompress_file(tmp_bit_path).squeeze(0).cpu()
        tmp_bit_path.unlink()

        # compute psnr
        real = tvf.to_tensor(Image.open(impath))
        # fake = tvf.to_tensor(Image.open(tmp_rec_path))
        mse = (real - fake).square().mean().item()
        psnr = -10 * math.log10(mse)
        # compute bpp
        bpp = num_bits / float(real.shape[1] * real.shape[2])
        # accumulate stats
        stats = {
            'bpp':  float(bpp),
            'mse':  float(mse),
            'psnr': float(psnr)
        }
        accumulated_stats['count'] += 1.0
        for k,v in stats.items():
            accumulated_stats[k] += v

        # logging
        msg = ', '.join([f'{k}={v:.3f}' for k,v in stats.items()])
        pbar.set_description(f'image {impath.stem}: {msg}')

    # average over all images
    count = accumulated_stats.pop('count')
    results = {k: v/count for k,v in accumulated_stats.items()}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',    type=str, default='d:/datasets/improcessing/kodak')
    parser.add_argument('--model',   type=str, default='qres34m')
    parser.add_argument('--lambdas', type=int, default=[16, 32, 64, 128, 256, 512, 1024, 2048],
                        nargs='+')
    parser.add_argument('--save_json', type=str, default=None)
    args = parser.parse_args()

    weights_root = Path('checkpoints/qres34m')
    dataset_root = Path(args.root)
    save_json_path = args.save_json or f'results/{dataset_root.stem}-{args.model}.json'
    if Path(save_json_path).is_file():
        print(f'Warning: {save_json_path} already exists. Will overwrite it.')

    all_lmb_results = defaultdict(list)
    for lmb in args.lambdas:
        # initialize model
        model = get_model_func(args.model)()
        # load weights
        wpath = weights_root / f'lmb{lmb}/last_ema.pt'
        model.load_state_dict(torch.load(wpath)['model'])

        print(f'Evaluating lmb={lmb}. Model weights={wpath}.')
        model.compress_mode()
        model = model.cuda()
        model.eval()

        # evaluate
        results = evaluate_model(model, dataset_root)
        print('results:', results, '\n')

        # accumulate results
        all_lmb_results['lambda'].append(lmb)
        for k,v in results.items():
            all_lmb_results[k].append(v)

        # save to json
        with open(save_json_path, 'w') as f:
            json.dump(all_lmb_results, fp=f, indent=4)

    # final print
    for k, vlist in all_lmb_results.items():
        vlist_str = ', '.join([f'{v:.12f}'[:8] for v in vlist])
        print(f'{k:<6s} = [{vlist_str}]')


if __name__ == '__main__':
    main()
