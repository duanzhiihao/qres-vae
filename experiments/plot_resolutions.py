from pathlib import Path
import json
import matplotlib.pyplot as plt



def plot(stats, label, ls='-', ax=None, **kwargs):
    module = ax if ax is not None else plt
    p = module.plot(stats['bppix'], stats['psnr'], label=label,
        marker='.', markersize=6,
        linestyle=ls, linewidth=1, **kwargs
    )
    return p


def main():
    resolutions = [192, 256, 384, 512, 768, 1024, 1536, 2048]
    N = len(resolutions)

    with open('results/clic-test-qres34m.json', 'r') as f:
        qres_stats = json.load(f)
    with open('results/clic-test-cheng2020-anchor.json', 'r') as f:
        cheng2020_stats = json.load(f)
    with open('results/clic-test-mbt2018.json', 'r') as f:
        mbt2018_stats = json.load(f)

    fig, axs = plt.subplots(2, 4, figsize=(16,8))
    for i, res in enumerate(resolutions):
        ax = axs[i // int(N/2), i % int(N/2)]
        plot(qres_stats[str(res)], label='QRes-VAE (34M)', ax=ax)
        plot(cheng2020_stats[str(res)], label='Cheng 2020 Anchor', ax=ax)
        plot(mbt2018_stats[str(res)], label='MBT 2018 Joint AR & H', ax=ax)

        ax: plt.Axes
        ax.set_xlabel('Bpp')
        ax.set_ylabel('PSNR')
        ax.set_xlim(0.0, 3.2)
        ax.set_ylim(28, 44)
        ax.set_title(f'Resolution $r = {res}$')
        ax.grid(True, alpha=0.24)
        ax.legend(loc='lower right')

    plt.tight_layout()
    # plt.savefig('results/exp-clic-resolutions.pdf')
    plt.show()
    debug = 1


if __name__ == '__main__':
    main()
