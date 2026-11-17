"""按工况（文件编号标签）生成九通道振动信号，反归一化到原始电压量纲并保存 NPY / CSV。

用法：
  D:\\Anaconda\\envs\\DL3.91\\python.exe sample.py                               # 全部工况 × 全部 guidance_scale
  D:\\Anaconda\\envs\\DL3.91\\python.exe sample.py --labels 19 --scales 2.0 --n 32
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from vibddpm.data import Normalizer, norm_path
from vibddpm.model import build_model
from vibddpm.utils import labels_to_index, load_config, load_json, run_dir, save_json, seed_everything


def load_trained(cfg, ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    diffusion = build_model(cfg).to(device)
    # 使用 EMA 权重生成
    ema_sd = {k[len('ema_model.'):]: v for k, v in ck['ema'].items() if k.startswith('ema_model.')}
    diffusion.load_state_dict(ema_sd)
    diffusion.eval()
    return diffusion, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--ckpt', default=None, help='默认 run 目录下 checkpoints/best.pt')
    ap.add_argument('--labels', type=int, nargs='*', default=None, help='工况标签（文件编号），默认全部')
    ap.add_argument('--scales', type=float, nargs='*', default=None)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--sampling_timesteps', type=int, default=None)
    ap.add_argument('--out', default='samples')
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--seed', type=int, default=None, help='采样随机种子；同一种子结果可复现，换种子得到新样本')
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg['project']['run_name'] = args.run_name
    sc = cfg['sample']
    seed = cfg['project']['seed'] if args.seed is None else args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rd = run_dir(cfg)
    ckpt = Path(args.ckpt) if args.ckpt else rd / 'checkpoints' / 'best.pt'
    diffusion, ck = load_trained(cfg, ckpt, device)

    stats = load_json(norm_path(cfg))
    if ck['norm_version'] != stats['version']:
        raise RuntimeError(f'checkpoint 训练所用归一化 {ck["norm_version"]} 与当前 {stats["version"]} 不一致')
    nz = Normalizer(stats)
    labels = args.labels or cfg['data']['labels']
    scales = args.scales or sc['guidance_scales']
    n = args.n or sc['num_per_condition']
    S = args.sampling_timesteps or sc['sampling_timesteps']
    fs = cfg['data']['target_fs']
    out_dir = rd / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[sample] ckpt {ckpt.name} (step {ck["step"]}), 标签 {labels}, w={scales}, N={n}, 采样步数 {S}')

    for w in scales:
        for label in labels:
            seed_everything(seed + 1000 * int(label) + int(w * 100))       # 每个(工况, w)固定种子，可复现
            idx = torch.from_numpy(labels_to_index(cfg, [label]))
            zs = []
            for b0 in range(0, n, sc['batch_size']):
                bs = min(sc['batch_size'], n - b0)
                zs.append(diffusion.sample(idx.repeat(bs), guidance_scale=w, sampling_timesteps=S).cpu().numpy())
            z = np.concatenate(zs, 0)
            x = nz.inverse(z).astype(np.float32)                           # 逐通道反归一化 -> 电压 (V)
            assert x.shape == (n, cfg['model']['channels'], cfg['model']['seq_length'])
            stem = f'gen_label{label}_w{w:g}' + (f'_s{args.seed}' if args.seed is not None else '')
            meta = dict(label=int(label), condition_desc=cfg['data']['label_desc'][label],
                        internal_index=int(idx.item()), guidance_scale=w, shape=list(x.shape),
                        layout='[N, channel, time]', sample_rate_hz=fs, unit='V',
                        channels=stats['channels'], norm_version=stats['version'],
                        checkpoint=str(ckpt), checkpoint_step=int(ck['step']), sampling_timesteps=S,
                        seed=seed + 1000 * int(label) + int(w * 100))
            if 'npy' in sc['save_format']:
                np.save(out_dir / f'{stem}.npy', x)
            save_json(meta, out_dir / f'{stem}.json')
            if 'csv' in sc['save_format']:
                write_csv(out_dir / f'{stem}.csv', x[:sc['csv_max_windows']], meta, fs)
            rms = np.sqrt((x ** 2).mean((0, 2)))
            print(f'  标签 {label} w={w:g}: {x.shape}  RMS(V) ' + ' '.join(f'{r:.3f}' for r in rms))


def write_csv(path, x, meta, fs):
    """长表：window, sample, time_s, 电压_0..电压_8；首部 # 注释行写元信息。"""
    n, C, L = x.shape
    t = np.arange(L) / fs
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        for k in ('label', 'condition_desc', 'guidance_scale', 'sample_rate_hz', 'unit', 'norm_version', 'checkpoint_step'):
            f.write(f'# {k}: {meta[k]}\n')
        f.write('window,sample,time_s,' + ','.join(meta['channels']) + '\n')
        for i in range(n):
            block = np.column_stack([np.full(L, i), np.arange(L), t, x[i].T])
            np.savetxt(f, block, delimiter=',', fmt=['%d', '%d', '%.6f'] + ['%.8f'] * C)


if __name__ == '__main__':
    main()
