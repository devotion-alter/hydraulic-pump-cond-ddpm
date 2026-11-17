"""基于 lucidrains/denoising-diffusion-pytorch (v2.1.1) Unet1D / GaussianDiffusion1D 的类别条件 + CFG 1D 适配。

仓库现状（已核对源码）：
  * Unet1D.forward(x, time, x_self_cond) 不支持类别条件；
  * 仓库的 classifier_free_guidance.py 是 2D 图像 Unet（Conv2d），不能直接用于 1D；
  * GaussianDiffusion1D 的 p_mean_variance / ddim_sample 把 x0 硬裁剪到 [-1, 1]，
    且 auto_normalize 假设数据在 [0, 1]，均不适用于 z-score 振动数据。
因此：
  * ConditionalUnet1D 复用 Unet1D 的全部层（ResnetBlock / LinearAttention / Attention / 上下采样），
    新增可学习类别 Embedding（含单独的 null embedding，索引 = num_classes），
    经 MLP 后与时间步嵌入相加，作为 ResnetBlock 的 scale/shift 条件注入每个残差模块；
  * ConditionalGaussianDiffusion1D 复用父类的噪声日程、q_sample、后验系数，
    重写预测/采样/损失路径以传递类别条件、实现 CFG，并把 x0 裁剪改为可配置的 z-score 范围。
"""
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
    Unet1D, GaussianDiffusion1D, ModelPrediction, extract,
)


class ConditionalUnet1D(Unet1D):
    def __init__(self, *, num_classes, cond_drop_prob=0.1, **kwargs):
        kwargs.setdefault('self_condition', False)
        assert not kwargs['self_condition'], '本项目不使用 self-conditioning'
        super().__init__(**kwargs)
        time_dim = self.time_mlp[-1].out_features
        dim = kwargs['dim']
        self.num_classes = num_classes
        self.null_index = num_classes                 # 单独的 null condition embedding
        self.cond_drop_prob = cond_drop_prob
        self.class_emb = nn.Embedding(num_classes + 1, dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

    def cond_embedding(self, time, classes, cond_drop_prob):
        b = classes.shape[0]
        if cond_drop_prob > 0:
            drop = torch.rand(b, device=classes.device) < cond_drop_prob
            classes = torch.where(drop, torch.full_like(classes, self.null_index), classes)
        return self.time_mlp(time) + self.class_mlp(self.class_emb(classes))

    def forward(self, x, time, classes, cond_drop_prob=None):
        """classes: [B] long，取值 0..K-1；等于 null_index 表示无条件。"""
        cond_drop_prob = self.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
        t = self.cond_embedding(time, classes, cond_drop_prob if self.training else 0.)

        # 以下与 Unet1D.forward 相同，只是 t 换成 (时间 + 类别) 联合嵌入
        x = self.init_conv(x)
        r = x.clone()
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t); h.append(x)
            x = block2(x, t); x = attn(x); h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1); x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1); x = block2(x, t); x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

    def forward_with_cfg(self, x, time, classes, guidance_scale=1.0):
        """eps = eps_uncond + w * (eps_cond - eps_uncond)；w=1 退化为纯条件预测。"""
        if guidance_scale == 1.0:
            return self.forward(x, time, classes, cond_drop_prob=0.)
        null = torch.full_like(classes, self.null_index)
        out = self.forward(torch.cat([x, x]), torch.cat([time, time]),
                           torch.cat([classes, null]), cond_drop_prob=0.)
        eps_c, eps_u = out.chunk(2)
        return eps_u + guidance_scale * (eps_c - eps_u)


class ConditionalGaussianDiffusion1D(GaussianDiffusion1D):
    def __init__(self, model, *, seq_length, clip_x0=None, **kwargs):
        kwargs['auto_normalize'] = False              # 数据已在外部做 z-score
        super().__init__(model, seq_length=seq_length, **kwargs)
        assert self.objective == 'pred_noise', '第一版仅支持 pred_noise，使 CFG 组合一致的噪声预测'
        self.clip_x0 = clip_x0

    def _clip(self, x0):
        return x0.clamp(-self.clip_x0, self.clip_x0) if self.clip_x0 else x0

    # ---------------- 预测
    def model_predictions(self, x, t, classes, guidance_scale=1.0, clip_x_start=True):
        pred_noise = self.model.forward_with_cfg(x, t, classes, guidance_scale)
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        if clip_x_start and self.clip_x0:
            x_start = self._clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)   # 与裁剪后的 x0 保持一致
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, guidance_scale=1.0, clip_denoised=True):
        x_start = self.model_predictions(x, t, classes, guidance_scale, clip_denoised).pred_x_start
        mean, var, log_var = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return mean, var, log_var, x_start

    # ---------------- 采样
    @torch.no_grad()
    def p_sample(self, x, t, classes, guidance_scale=1.0, clip_denoised=True):
        bt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        mean, _, log_var, x_start = self.p_mean_variance(x, bt, classes, guidance_scale, clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.
        return mean + (0.5 * log_var).exp() * noise, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, classes, guidance_scale=1.0, progress=True):
        img = torch.randn(shape, device=self.betas.device)
        for t in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps,
                      desc=f'DDPM w={guidance_scale}', disable=not progress, leave=False):
            img, _ = self.p_sample(img, t, classes, guidance_scale)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, classes, guidance_scale=1.0, sampling_timesteps=None, progress=True):
        device, T = self.betas.device, self.num_timesteps
        S = sampling_timesteps or self.sampling_timesteps
        eta = self.ddim_sampling_eta
        times = list(reversed(torch.linspace(-1, T - 1, steps=S + 1).int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)
        for time, time_next in tqdm(pairs, desc=f'DDIM{S} w={guidance_scale}', disable=not progress, leave=False):
            tc = torch.full((shape[0],), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(img, tc, classes, guidance_scale)
            if time_next < 0:
                img = x_start
                continue
            a, an = self.alphas_cumprod[time], self.alphas_cumprod[time_next]
            sigma = eta * ((1 - a / an) * (1 - an) / (1 - a)).sqrt()
            c = (1 - an - sigma ** 2).sqrt()
            img = x_start * an.sqrt() + c * pred_noise + sigma * torch.randn_like(img)
        return img

    @torch.no_grad()
    def sample(self, classes, guidance_scale=1.0, sampling_timesteps=None, progress=True):
        """classes: [B] long 内部索引。返回 z-score 空间样本 [B, C, L]。"""
        classes = classes.to(self.betas.device)
        shape = (classes.shape[0], self.channels, self.seq_length)
        S = sampling_timesteps or self.sampling_timesteps
        if S < self.num_timesteps:
            return self.ddim_sample(shape, classes, guidance_scale, S, progress)
        return self.p_sample_loop(shape, classes, guidance_scale, progress)

    # ---------------- 训练损失
    def p_losses(self, x_start, t, classes, noise=None, cond_drop_prob=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        out = self.model(x, t, classes, cond_drop_prob=cond_drop_prob)
        loss = F.mse_loss(out, noise, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        return loss * extract(self.loss_weight, t, loss.shape)        # [B]，便于按工况统计

    def forward(self, x, classes):
        b, c, n = x.shape
        assert n == self.seq_length and c == self.channels, f'输入形状应为 [B,{self.channels},{self.seq_length}]，实际 {tuple(x.shape)}'
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        per_sample = self.p_losses(x, t, classes)
        return per_sample.mean(), per_sample


def build_model(cfg):
    m, d = cfg['model'], cfg['diffusion']
    unet = ConditionalUnet1D(
        dim=m['dim'], dim_mults=tuple(m['dim_mults']), channels=m['channels'],
        num_classes=m['num_classes'], cond_drop_prob=m['cond_drop_prob'], dropout=m.get('dropout', 0.),
    )
    diffusion = ConditionalGaussianDiffusion1D(
        unet, seq_length=m['seq_length'], timesteps=d['timesteps'],
        sampling_timesteps=d['sampling_timesteps'], objective=d['objective'],
        beta_schedule=d['beta_schedule'], clip_x0=d.get('clip_x0'),
    )
    return diffusion
