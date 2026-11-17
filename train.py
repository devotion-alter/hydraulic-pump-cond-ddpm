"""条件 1D DDPM 训练：验证、按工况统计损失、条件敏感性检查、Checkpoint 保存与恢复。

用法：
  D:\\Anaconda\\envs\\DL3.91\\python.exe train.py                      # 完整训练
  D:\\Anaconda\\envs\\DL3.91\\python.exe train.py --resume auto        # 从 last.pt 恢复
"""
import argparse
import csv
import random
import time
from collections import defaultdict

import numpy as np
import torch
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from vibddpm.data import RandomCropDataset, WindowDataset, prepare
from vibddpm.model import build_model
from vibddpm.utils import load_config, run_dir, save_json, seed_everything


def cycle(dl):
    while True:
        for b in dl:
            yield b


@torch.no_grad()
def evaluate_loss(diffusion, ds, cfg, device, num_t, seed=1234):
    """固定时间步网格 + 固定噪声的验证损失，按工况分别统计；
    同时用 交换标签 / null 标签 评估同一批样本，检验模型是否真正利用了条件。"""
    diffusion.eval()
    K = cfg['model']['num_classes']
    g = torch.Generator().manual_seed(seed)
    T = diffusion.num_timesteps
    t_grid = torch.linspace(0, T - 1, num_t).long()
    res = defaultdict(lambda: defaultdict(list))
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    for x, c in dl:
        x, c = x.to(device), c.to(device)
        for tv in t_grid:
            t = torch.full((len(x),), int(tv), device=device, dtype=torch.long)
            noise = torch.randn(x.shape, generator=g).to(device)
            variants = {'correct': c, 'swapped': (c + 1) % K,
                        'null': torch.full_like(c, diffusion.model.null_index)}
            for name, cc in variants.items():
                l = diffusion.p_losses(x, t, cc, noise=noise, cond_drop_prob=0.)
                for idx in c.unique().tolist():
                    res[idx][name].append(l[c == idx].cpu())
    out = {}
    labels = cfg['data']['labels']
    for idx, d in res.items():
        out[labels[idx]] = {k: torch.cat(v).mean().item() for k, v in d.items()}
    out['all'] = {k: float(np.mean([out[l][k] for l in labels])) for k in ('correct', 'swapped', 'null')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--resume', default=None, help="checkpoint 路径，或 'auto' 表示 run 目录下 last.pt")
    ap.add_argument('--steps', type=int, default=None, help='覆盖 train_steps')
    ap.add_argument('--run_name', default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg['project']['run_name'] = args.run_name
    tc = cfg['train']
    if args.steps:
        tc['train_steps'] = args.steps
    seed_everything(cfg['project']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rd = run_dir(cfg)
    (rd / 'checkpoints').mkdir(exist_ok=True)

    splits, normalizer, stats = prepare(cfg)
    if tc.get('random_crop', False):
        segs = [normalizer(s.T).T for s in splits['train']['segments']]
        train_ds = RandomCropDataset(segs, cfg['data']['labels'], cfg, epoch_len=len(splits['train']['x']))
        print(f'[data] 训练使用随机裁剪增强：每工况训练段 {[len(s) for s in segs]} 点，可取窗口起点数 '
              f'{[len(s) - cfg["data"]["window_size"] + 1 for s in segs]}')
    else:
        train_ds = WindowDataset(normalizer(splits['train']['x']), splits['train']['y'], cfg)
    val_ds = WindowDataset(normalizer(splits['val']['x']), splits['val']['y'], cfg)
    print(f'[data] train windows {splits["train"]["x"].shape}  val {tuple(val_ds.x.shape)}  norm={stats["version"]}')
    for l in cfg['data']['labels']:
        print(f'[data]   标签 {l} ({cfg["data"]["label_desc"][l]}): {stats["split"][l]["n_windows"]}')
    save_json({k: v for k, v in cfg.items() if not k.startswith('_')}, rd / 'config_used.json')

    diffusion = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in diffusion.parameters())
    print(f'[model] 参数量 {n_params / 1e6:.2f} M，设备 {device}')
    opt = torch.optim.Adam(diffusion.parameters(), lr=tc['lr'])
    ema = EMA(diffusion, beta=tc['ema_decay'], update_every=tc['ema_update_every']).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=tc['amp'])

    step, best_val = 0, float('inf')
    ckpt_p = args.resume
    if ckpt_p == 'auto':
        ckpt_p = rd / 'checkpoints' / 'last.pt'
        ckpt_p = ckpt_p if ckpt_p.exists() else None
    if ckpt_p:
        ck = torch.load(ckpt_p, map_location=device)
        diffusion.load_state_dict(ck['model']); ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        step, best_val = ck['step'], ck.get('best_val', best_val)
        if ck.get('norm_version') != stats['version']:
            print(f'[warn] checkpoint 归一化版本 {ck.get("norm_version")} 与当前 {stats["version"]} 不一致')
        torch.set_rng_state(ck['rng']['torch']); np.random.set_state(ck['rng']['numpy'])
        random.setstate(ck['rng']['python'])
        if torch.cuda.is_available() and ck['rng'].get('cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        print(f'[resume] 从 {ckpt_p} 恢复，step={step}')

    g = torch.Generator().manual_seed(cfg['project']['seed'] + step)
    dl = cycle(DataLoader(train_ds, batch_size=tc['batch_size'], shuffle=True, drop_last=True,
                          num_workers=tc['num_workers'], generator=g, pin_memory=True))

    def save(name):
        torch.save(dict(model=diffusion.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                        scaler=scaler.state_dict(), step=step, best_val=best_val,
                        cfg={k: v for k, v in cfg.items() if not k.startswith('_')},
                        norm_version=stats['version'], labels=cfg['data']['labels'],
                        rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                                 python=random.getstate(),
                                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)),
                   rd / 'checkpoints' / name)

    log_p = rd / 'train_log.csv'
    labels = cfg['data']['labels']
    new_log = not log_p.exists() or step == 0
    logf = open(log_p, 'w' if new_log else 'a', newline='', encoding='utf-8')
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(['step', 'kind'] + [f'{k}' for k in ['all'] + labels] + ['swapped_all', 'null_all', 'lr', 'sec'])

    run_loss = defaultdict(list)
    t0 = time.time()
    diffusion.train()
    while step < tc['train_steps']:
        x, c = next(dl)
        x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=tc['amp']):
            loss, per = diffusion(x, c)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), tc['grad_clip'])
        scaler.step(opt); scaler.update()
        ema.update()
        step += 1
        per = per.detach()
        run_loss['all'].append(loss.item())
        for idx in c.unique().tolist():
            run_loss[labels[idx]].append(per[c == idx].mean().item())

        if step % tc['log_every'] == 0:
            m = {k: float(np.mean(v)) for k, v in run_loss.items()}
            el = time.time() - t0
            print(f'step {step:6d} | loss {m["all"]:.4f} | ' +
                  ' '.join(f'{l}:{m.get(l, float("nan")):.4f}' for l in labels) +
                  f' | {tc["log_every"] / el:.1f} it/s')
            writer.writerow([step, 'train', m['all']] + [m.get(l) for l in labels] + ['', '', tc['lr'], round(el, 1)])
            logf.flush()
            run_loss.clear(); t0 = time.time()

        if step % tc['val_every'] == 0 or step == tc['train_steps']:
            v = evaluate_loss(ema.ema_model, val_ds, cfg, device, tc['val_num_t'])
            print(f'  [val EMA] correct {v["all"]["correct"]:.4f} | swapped {v["all"]["swapped"]:.4f} | '
                  f'null {v["all"]["null"]:.4f} | ' +
                  ' '.join(f'{l}: c={v[l]["correct"]:.4f}/s={v[l]["swapped"]:.4f}' for l in labels))
            writer.writerow([step, 'val', v['all']['correct']] + [v[l]['correct'] for l in labels] +
                            [v['all']['swapped'], v['all']['null'], tc['lr'], ''])
            logf.flush()
            if v['all']['correct'] < best_val:
                best_val = v['all']['correct']
                save('best.pt')
                print(f'  [ckpt] best.pt 更新 (val {best_val:.4f})')
            diffusion.train()

        if step % tc['save_every'] == 0 or step == tc['train_steps']:
            save('last.pt')

    logf.close()
    print(f'[done] 训练结束 step={step}，best val={best_val:.4f}，输出目录 {rd}')


if __name__ == '__main__':
    main()
