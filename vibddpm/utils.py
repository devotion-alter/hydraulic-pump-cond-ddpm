import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / 'configs' / 'default.yaml'
    with open(path, 'r', encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f)
    cfg['_config_path'] = str(path)
    resolve_labels(cfg)
    return cfg


def resolve_labels(cfg):
    """工况标签 = CSV 文件名编号。生成 label <-> 内部 Embedding 索引映射。"""
    dcfg = cfg['data']
    raw_dir = ROOT / dcfg['raw_dir']
    labels = dcfg.get('labels', 'auto')
    if labels == 'auto':
        labels = sorted(int(p.stem) for p in raw_dir.glob('*.csv') if re.fullmatch(r'\d+', p.stem))
    labels = sorted(int(l) for l in labels)
    assert len(labels) >= 2, f'至少需要两个工况 CSV，当前: {labels}'
    for l in labels:
        assert (raw_dir / f'{l}.csv').exists(), f'缺少 {raw_dir / f"{l}.csv"}'
    dcfg['labels'] = labels
    dcfg['label_to_index'] = {l: i for i, l in enumerate(labels)}
    desc = dcfg.get('label_desc') or {}
    dcfg['label_desc'] = {l: str(desc.get(l, desc.get(str(l), f'condition_{l}'))) for l in labels}
    if cfg['model'].get('num_classes', 'auto') == 'auto':
        cfg['model']['num_classes'] = len(labels)
    assert cfg['model']['num_classes'] == len(labels)
    return cfg


def labels_to_index(cfg, labels):
    m = cfg['data']['label_to_index']
    return np.array([m[int(l)] for l in np.atleast_1d(labels)], dtype=np.int64)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_dir(cfg):
    d = ROOT / cfg['project']['out_dir'] / cfg['project']['run_name']
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
