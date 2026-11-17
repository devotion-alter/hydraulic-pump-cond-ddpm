"""时域/频域/统计特征与对比指标。输入均为原始电压量纲 [N, C, L]。"""
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis


def time_features(x):
    """返回 dict: 每个 [N, C]。"""
    rms = np.sqrt((x ** 2).mean(-1))
    return dict(RMS=rms, Std=x.std(-1), Peak=np.abs(x).max(-1), Kurtosis=kurtosis(x, axis=-1, fisher=False))


def fft_amp(x, fs):
    """单边 FFT 幅值谱（Hann 窗，幅值校正），返回 f, [N, C, F]。"""
    L = x.shape[-1]
    w = np.hanning(L)
    X = np.fft.rfft((x - x.mean(-1, keepdims=True)) * w, axis=-1)
    amp = 2 * np.abs(X) / w.sum()
    return np.fft.rfftfreq(L, 1 / fs), amp


def psd(x, fs, nperseg=256):
    f, p = welch(x, fs=fs, nperseg=nperseg, axis=-1)
    return f, p                       # [N, C, F]


def corr_matrices(x):
    """逐窗口 9x9 通道相关矩阵，返回 [N, C, C]。"""
    xc = x - x.mean(-1, keepdims=True)
    cov = np.einsum('ncl,ndl->ncd', xc, xc)
    d = np.sqrt(np.einsum('ncc->nc', cov))
    return cov / (d[:, :, None] * d[:, None, :] + 1e-12)


def db(p):
    return 10 * np.log10(p + 1e-20)


def psd_db_distance(pa, pb):
    """两组 PSD 平均谱的平均绝对 dB 差（跨通道、频率平均）。"""
    return float(np.abs(db(pa.mean(0)) - db(pb.mean(0))).mean())


def corr_distance(ca, cb):
    C = ca.shape[-1]
    m = ~np.eye(C, dtype=bool)
    return float(np.abs(ca.mean(0) - cb.mean(0))[m].mean())


def feature_vector(x, fs, nperseg=256):
    """用于工况判别的特征：各通道 log-PSD（Welch）+ log RMS。"""
    _, p = psd(x, fs, nperseg)
    f1 = db(p).reshape(len(x), -1)
    f2 = np.log(time_features(x)['RMS'])
    return np.concatenate([f1, f2], 1)


class NearestCentroid:
    """在真实训练窗口上拟合的标准化最近质心分类器，用于检查生成样本是否体现对应工况。"""

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        self.classes = np.unique(y)
        self.centroids = np.stack([Z[y == c].mean(0) for c in self.classes])
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        d = ((Z[:, None, :] - self.centroids[None]) ** 2).sum(-1)
        return self.classes[d.argmin(1)]
