import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

from timm.utils import ModelEmaV2


class ImageDataset(Dataset):
    def __init__(self, root, crop=None):
        transform = []
        if crop is not None: # training with simple augmentation
            transform.append(tv.transforms.RandomCrop(256, pad_if_needed=True, padding_mode='reflect'))
            transform.append(tv.transforms.RandomHorizontalFlip(p=0.5))
        transform = tv.transforms.Compose(transform + [tv.transforms.ToTensor()])
        self.transform = transform
        self.img_paths = list(tqdm(Path(root).rglob('*.*')))
        print(f'Number of files found in {root}: {len(self.img_paths)}')
        print(f'{root} transform={transform}', '\n')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        impath = self.img_paths[index]
        img = Image.open(impath).convert('RGB') 
        im = self.transform(img)
        return im


def get_dataloader(root, crop=None, batch_size=1, workers=0, shuffle=True):
    dataset = ImageDataset(root, crop)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=workers, pin_memory=True, drop_last=False)
    return dataloader


def parse_cli_args():
    # ====== training settings ======
    parser = argparse.ArgumentParser()
    # model setting
    parser.add_argument('--model',      type=str,   default='qres34m')
    parser.add_argument('--lmb',        type=float, default=None)
    # data setting
    parser.add_argument('--train_root', type=str,   default='d:/datasets/coco/train2017')
    parser.add_argument('--train_crop', type=int,   default=None)
    parser.add_argument('--val_root',   type=str,   default='d:/datasets/improcessing/kodak')
    # optimization setting
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--grad_clip',  type=float, default=2.0)
    # training policy and tricks setting
    parser.add_argument('--epochs',     type=int,   default=400)
    # device setting
    parser.add_argument('--workers',    type=int,   default=0)
    cfg = parser.parse_args()
    return cfg


def get_model(name, lmb=None):
    from models.library import qres34m, qres34m_lossless, qres17m
    registry = {'qres34m': qres34m, 'qres34m_ll': qres34m_lossless, 'qres17m': qres17m}
    model = registry[name](lmb) if (lmb is not None) else registry[name]()
    return model


def main():
        cfg = parse_cli_args()

        # set device
        device = torch.device('cuda', 0)
        torch.backends.cudnn.benchmark = True

        # set dataset
        trainloader = get_dataloader(cfg.train_root, cfg.train_crop, batch_size=cfg.batch_size,
                                     workers=cfg.workers, shuffle=True)
        valloader = get_dataloader(cfg.val_root, crop=None, batch_size=max(1, cfg.batch_size // 2),
                                   workers=cfg.workers//2, shuffle=False)
        # set model
        model: torch.nn.Module = get_model(cfg.model)
        model = model.to(device)
        # EMA
        ema = ModelEmaV2(model, decay=0.9998)
        print(f'Using model {type(model)}, lmb={cfg.lmb}. EMA decay={ema.decay}', '\n')

        # set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

        # logging
        log_dir = Path(f'runs/{Path(cfg.train_root).stem}/{cfg.model}-lmb{cfg.lmb}')
        log_dir.mkdir(parents=True, exist_ok=False)
        print(f'\u001b[94m -- Logging run at {log_dir} -- \u001b[0m', '\n')

        # ======================== start training ========================
        for epoch in range(cfg.epochs):
            pbar = tqdm(pbar, total=len(trainloader))
            model.train()
            for imgs in pbar:
                imgs = imgs.to(device=device)

                # forward
                stats = model(imgs)
                loss = stats['loss']
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

                minibatch_log(pbar, cfg, epoch, grad_norm.item(), stats)
            pbar.close()

            results1 = evaluate_model(model, valloader)
            results2 = evaluate_model(ema.module, valloader)
            save_checkpoint(model,      epoch, results1, path=log_dir/'last.pt', optimizer=optimizer)
            save_checkpoint(ema.module, epoch, results2, path=log_dir/'last_ema.pt')


def minibatch_log(pbar, cfg, epoch, grad_norm, stats):
    msg =  f'Epoch: {epoch}/{cfg.epochs-1} || grad={grad_norm:.4g}'
    for k,v in stats.items():
        msg += f' || {k}={v:.4g}'
    pbar.set_description(msg)


@torch.no_grad()
def evaluate_model(model, valloader):
    all_image_stats = defaultdict(float)
    for imgs in tqdm(valloader):
        stats = model.forward_eval(imgs)
        for k, v in stats.items():
            all_image_stats[k] += float(v)
        all_image_stats['count'] += 1
    # average over all images
    count = all_image_stats.pop('count')
    results = {k: v/count for k,v in all_image_stats.items()}
    return results


def save_checkpoint(model, epoch, results, path, optimizer=None):
    # save last checkpoint
    print(f'Saving model to {path} ...\n results:\n {results}\n')
    checkpoint = {
        'model'     : model.state_dict(),
        'epoch'     : epoch,
        'results'   : results,
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, path)


if __name__ == '__main__':
    main()
