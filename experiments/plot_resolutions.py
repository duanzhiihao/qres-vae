from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def bd_rate(r1, psnr1, r2, psnr2):
    """ Compute average bit rate saving of RD-2 over RD-1.

    Equivalent to the implementations in:
    https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
    https://github.com/google/compare-codecs/blob/master/lib/visual_metrics.py
    """
    lr1 = np.log(r1)
    lr2 = np.log(r2)

    # fit each curve by a polynomial
    degree = 3
    p1 = np.polyfit(psnr1, lr1, deg=degree)
    p2 = np.polyfit(psnr2, lr2, deg=degree)
    # compute integral of the polynomial
    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)
    # area under the curve = integral(max) - integral(min)
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    auc1 = np.polyval(p_int1, max_psnr) - np.polyval(p_int1, min_psnr)
    auc2 = np.polyval(p_int2, max_psnr) - np.polyval(p_int2, min_psnr)

    # find avgerage difference
    avg_exp_diff = (auc2 - auc1) / (max_psnr - min_psnr)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    return avg_diff


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

    dataset_name = 'clic2022-test'
    with open(f'results/appendix/{dataset_name}-resolutions-qres34m.json', 'r') as f:
        qres_stats = json.load(f)
    with open(f'results/appendix/{dataset_name}-resolutions-cheng2020-anchor.json', 'r') as f:
        cheng2020_stats = json.load(f)
    with open(f'results/appendix/{dataset_name}-resolutions-mbt2018.json', 'r') as f:
        mbt2018_stats = json.load(f)

    fig, axs = plt.subplots(2, 4, figsize=(16,8))
    for i, res in enumerate(resolutions):
        bd_cheng = bd_rate(
            mbt2018_stats[str(res)]['bppix'], mbt2018_stats[str(res)]['psnr'],
            cheng2020_stats[str(res)]['bppix'], cheng2020_stats[str(res)]['psnr'],
        )
        bd_ours = bd_rate(
            mbt2018_stats[str(res)]['bppix'], mbt2018_stats[str(res)]['psnr'],
            qres_stats[str(res)]['bppix'], qres_stats[str(res)]['psnr'],
        )

        ax = axs[i // int(N/2), i % int(N/2)]
        space = '                '
        plot(qres_stats[str(res)],      label=f'QRes-VAE (34M), {space} BD-rate = {bd_ours:.1f}%', ax=ax)
        plot(cheng2020_stats[str(res)], label=f'Cheng 2020 LIC, {space} BD-rate = {bd_cheng:.1f}%', ax=ax)
        plot(mbt2018_stats[str(res)],   label=f'Minnen 2018 Joint AR & H,  BD-rate = 0.0%', ax=ax)

        ax: plt.Axes
        ax.set_xlabel('Bpp')
        ax.set_ylabel('PSNR')
        ax.set_xlim(0.0, 3.5)
        ax.set_ylim(26, 44)
        ax.set_title(f'Resolution $r = {res}$')
        ax.grid(True, alpha=0.24)
        ax.legend(loc='lower right', prop={'size':8})

    plt.tight_layout()
    plt.savefig(f'results/appendix/exp-{dataset_name}-resolutions.pdf')
    plt.show()
    debug = 1


if __name__ == '__main__':
    main()
