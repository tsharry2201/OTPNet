#!/usr/bin/env python3
"""
评估全景锐化结果
支持评估PSNR、SSIM、SAM、ERGAS、Q8指标
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
from scipy.ndimage import uniform_filter
import cv2


def norm_blocco(x):
    """
    Q2n 辅助函数：块归一化
    """
    a = np.mean(x)
    c = np.std(x, ddof=1)
    
    if c == 0:
        c = np.finfo(np.float64).eps
    
    y = ((x - a) / c) + 1
    
    return y, a, c


def onion_mult(onion1, onion2):
    """
    Q2n 辅助函数：超复数乘法（1D）
    """
    N = len(onion1)
    
    if N > 1:
        L = N // 2
        
        a = onion1[:L]
        b = onion1[L:]
        b = np.concatenate([[b[0]], -b[1:]])
        c = onion2[:L]
        d = onion2[L:]
        d = np.concatenate([[d[0]], -d[1:]])
        
        if N == 2:
            ris = np.array([a[0]*c[0] - d[0]*b[0], a[0]*d[0] + c[0]*b[0]])
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, np.concatenate([[b[0]], -b[1:]]))
            ris3 = onion_mult(np.concatenate([[a[0]], -a[1:]]), d)
            ris4 = onion_mult(c, b)
            
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            
            ris = np.concatenate([aux1, aux2])
    else:
        ris = onion1 * onion2
    
    return ris


def onion_mult2D(onion1, onion2):
    """
    Q2n 辅助函数：超复数乘法（2D）
    """
    N3 = onion1.shape[2]
    
    if N3 > 1:
        L = N3 // 2
        
        a = onion1[:, :, :L]
        b = onion1[:, :, L:]
        b = np.concatenate([b[:, :, 0:1], -b[:, :, 1:]], axis=2)
        c = onion2[:, :, :L]
        d = onion2[:, :, L:]
        d = np.concatenate([d[:, :, 0:1], -d[:, :, 1:]], axis=2)
        
        if N3 == 2:
            ris = np.concatenate([a * c - d * b, a * d + c * b], axis=2)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, np.concatenate([b[:, :, 0:1], -b[:, :, 1:]], axis=2))
            ris3 = onion_mult2D(np.concatenate([a[:, :, 0:1], -a[:, :, 1:]], axis=2), d)
            ris4 = onion_mult2D(c, b)
            
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            
            ris = np.concatenate([aux1, aux2], axis=2)
    else:
        ris = onion1 * onion2
    
    return ris


def onions_quality(dat1, dat2, size1):
    """
    Q2n 辅助函数：基于超复数的质量评估
    """
    dat1 = dat1.astype(np.float64)
    dat2 = dat2.astype(np.float64)
    dat2 = np.concatenate([dat2[:, :, 0:1], -dat2[:, :, 1:]], axis=2)
    N3 = dat1.shape[2]
    size2 = size1
    
    # 块归一化
    for i in range(N3):
        a1, s, t = norm_blocco(dat1[:, :, i])
        dat1[:, :, i] = a1
        
        if s == 0:
            if i == 0:
                dat2[:, :, i] = dat2[:, :, i] - s + 1
            else:
                dat2[:, :, i] = -((-dat2[:, :, i]) - s + 1)
        else:
            if i == 0:
                dat2[:, :, i] = ((dat2[:, :, i] - s) / t) + 1
            else:
                dat2[:, :, i] = -((((-dat2[:, :, i]) - s) / t) + 1)
    
    m1 = np.zeros(N3)
    m2 = np.zeros(N3)
    
    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = np.zeros((size1, size2))
    mod_q2 = np.zeros((size1, size2))
    
    for i in range(N3):
        m1[i] = np.mean(dat1[:, :, i])
        m2[i] = np.mean(dat2[:, :, i])
        mod_q1m += m1[i] ** 2
        mod_q2m += m2[i] ** 2
        mod_q1 += dat1[:, :, i] ** 2
        mod_q2 += dat2[:, :, i] ** 2
    
    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)
    
    termine2 = mod_q1m * mod_q2m
    termine4 = (mod_q1m ** 2) + (mod_q2m ** 2)
    int1 = (size1 * size2) / ((size1 * size2) - 1) * np.mean(mod_q1 ** 2)
    int2 = (size1 * size2) / ((size1 * size2) - 1) * np.mean(mod_q2 ** 2)
    termine3 = int1 + int2 - (size1 * size2) / ((size1 * size2) - 1) * ((mod_q1m ** 2) + (mod_q2m ** 2))
    
    mean_bias = 2 * termine2 / (termine4 + np.finfo(np.float64).eps)
    
    if termine3 == 0:
        q = np.zeros(N3)
        q[N3 - 1] = mean_bias
    else:
        cbm = 2 / termine3
        qu = onion_mult2D(dat1, dat2)
        
        qm = onion_mult(m1, m2)
        qv = np.zeros(N3)
        for i in range(N3):
            qv[i] = (size1 * size2) / ((size1 * size2) - 1) * np.mean(qu[:, :, i])
        q = qv - (size1 * size2) / ((size1 * size2) - 1) * qm
        
        q = q * mean_bias * cbm
    
    return q


def q2n_index(img1, img2, block_size=32, q_shift=32):
    """
    Q2n index - 基于超复数理论的多光谱图像质量评估
    
    参考文献:
    [Garzelli09] A. Garzelli and F. Nencini, "Hypercomplex quality assessment of multi/hyper-spectral images,"
                 IEEE Geoscience and Remote Sensing Letters, 2009.
    
    Args:
        img1: 参考图像 (H, W, C)
        img2: 待评估图像 (H, W, C)
        block_size: 块大小
        q_shift: 块移动步长
    
    Returns:
        Q2n 指标值
    """
    N1, N2, N3 = img1.shape
    size2 = block_size
    
    stepx = int(np.ceil(N1 / q_shift))
    stepy = int(np.ceil(N2 / q_shift))
    
    if stepy <= 0:
        stepy = 1
        stepx = 1
    
    est1 = (stepx - 1) * q_shift + block_size - N1
    est2 = (stepy - 1) * q_shift + block_size - N2
    
    # 边界填充（镜像填充）
    if est1 != 0 or est2 != 0:
        refref = []
        fusfus = []
        
        for i in range(N3):
            a1 = img1[:, :, i]
            
            ia1 = np.zeros((N1 + est1, N2 + est2))
            ia1[:N1, :N2] = a1
            if est2 > 0:
                ia1[:, N2:N2 + est2] = ia1[:, N2 - 1:N2 - est2 - 1:-1]
            if est1 > 0:
                ia1[N1:N1 + est1, :] = ia1[N1 - 1:N1 - est1 - 1:-1, :]
            
            if i == 0:
                refref = ia1[:, :, np.newaxis]
            else:
                refref = np.concatenate([refref, ia1[:, :, np.newaxis]], axis=2)
        
        for i in range(N3):
            a2 = img2[:, :, i]
            
            ia2 = np.zeros((N1 + est1, N2 + est2))
            ia2[:N1, :N2] = a2
            if est2 > 0:
                ia2[:, N2:N2 + est2] = ia2[:, N2 - 1:N2 - est2 - 1:-1]
            if est1 > 0:
                ia2[N1:N1 + est1, :] = ia2[N1 - 1:N1 - est1 - 1:-1, :]
            
            if i == 0:
                fusfus = ia2[:, :, np.newaxis]
            else:
                fusfus = np.concatenate([fusfus, ia2[:, :, np.newaxis]], axis=2)
        
        img1 = refref
        img2 = fusfus
    
    # 转换为 uint16
    img1 = img1.astype(np.uint16)
    img2 = img2.astype(np.uint16)
    
    N1, N2, N3 = img1.shape
    
    # 将波段数填充到 2 的幂次
    if np.ceil(np.log2(N3)) != np.log2(N3):
        Ndif = int(2 ** np.ceil(np.log2(N3))) - N3
        dif = np.zeros((N1, N2, Ndif), dtype=np.uint16)
        img1 = np.concatenate([img1, dif], axis=2)
        img2 = np.concatenate([img2, dif], axis=2)
    
    N1, N2, N3 = img1.shape
    
    valori = np.zeros((stepx, stepy, N3))
    
    # 分块计算质量指标
    for j in range(stepx):
        for i in range(stepy):
            row_start = j * q_shift
            row_end = j * q_shift + block_size
            col_start = i * q_shift
            col_end = i * q_shift + size2
            
            o = onions_quality(
                img1[row_start:row_end, col_start:col_end, :],
                img2[row_start:row_end, col_start:col_end, :],
                block_size
            )
            valori[j, i, :] = o
    
    # 计算 Q2n
    q2n_index_map = np.sqrt(np.sum(valori ** 2, axis=2))
    q2n_value = np.mean(q2n_index_map)
    
    return q2n_value


def qnr_index(img1, img2, sensor='WV3', block_size=32):
    """
    根据传感器类型自动选择合适的 Q 指标
    - WV3: Q8 (8个波段)
    - GF2/QB: Q4 (4个波段)
    
    Args:
        img1: 参考图像 (H, W, C)
        img2: 待评估图像 (H, W, C)
        sensor: 传感器类型
        block_size: 块大小
    
    Returns:
        Q 指标值
    """
    # 根据传感器或波段数自动判断
    num_bands = img1.shape[2] if img1.ndim == 3 else 1
    
    if sensor.upper() in ['WV3', 'WV2'] or num_bands == 8:
        # WV3/WV2 使用 Q8
        return q2n_index(img1, img2, block_size, block_size)
    elif sensor.upper() in ['QB', 'GF2', 'IKONOS'] or num_bands == 4:
        # QB/GF2 使用 Q4
        return q2n_index(img1, img2, block_size, block_size)
    else:
        # 其他情况使用通用 Q2n
        return q2n_index(img1, img2, block_size, block_size)


def sam(img1, img2):
    """
    SAM (Spectral Angle Mapper) for 3D image, shape (H, W, C)
    Lower is better (0 is best)
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_ ** 2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_ ** 2).sum(axis=2))
    
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))


def psnr(img1, img2, dynamic_range=2047):
    """
    PSNR metric
    Higher is better (Inf is best)
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    mse = np.mean((img1_ - img2_) ** 2)
    
    if mse <= 1e-10:
        return np.inf
    return 20 * np.log10(dynamic_range / (np.sqrt(mse) + np.finfo(np.float64).eps))


def ssim(img1, img2, dynamic_range=2047):
    """
    SSIM (Structural Similarity Index)
    Higher is better (1 is best)
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    def _ssim_single_band(img1, img2, dynamic_range=2047):
        C1 = (0.01 * dynamic_range) ** 2
        C2 = (0.03 * dynamic_range) ** 2
        
        img1_ = img1.astype(np.float64)
        img2_ = img2.astype(np.float64)
        
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = cv2.filter2D(img1_, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2_, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[5:-5, 5:-5] - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    if img1.ndim == 2:
        return _ssim_single_band(img1, img2, dynamic_range)
    elif img1.ndim == 3:
        ssims = [_ssim_single_band(img1[..., i], img2[..., i], dynamic_range) for i in range(img1.shape[2])]
        return np.array(ssims).mean()
    else:
        raise ValueError("Image dimension error")


def ergas(img_fake, img_real, scale=4):
    """
    ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)
    Lower is better (0 is best)
    """
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_) ** 2)
        return 100 / scale * np.sqrt(mse / (mean_real ** 2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_) ** 2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real ** 2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError("Image dimension error")


def evaluate_single_image(sr, gt, dynamic_range=2047, scale=4, block_size=32, sensor='WV3'):
    """
    评估单张图像的所有指标
    
    Args:
        sr: 超分辨率结果，shape (C, H, W) 或 (H, W, C)
        gt: 真实标签，shape (C, H, W) 或 (H, W, C)
        dynamic_range: 动态范围 (WV3: 2047, GF2: 1023)
        scale: 下采样倍率
        block_size: Q2n的块大小（默认32）
        sensor: 传感器类型 (WV3/WV2: Q8, QB/GF2: Q4)
    
    Returns:
        dict: 包含各项指标的字典
    """
    # 确保形状是 (H, W, C)
    if sr.ndim == 3 and sr.shape[0] < sr.shape[2]:
        sr = np.transpose(sr, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    if gt.ndim == 3 and gt.shape[0] < gt.shape[2]:
        gt = np.transpose(gt, (1, 2, 0))
    
    # 确保值在有效范围内
    sr = np.clip(sr, 0, dynamic_range)
    gt = np.clip(gt, 0, dynamic_range)
    
    # 根据传感器确定 Q 指标名称
    num_bands = gt.shape[2] if gt.ndim == 3 else 1
    if sensor.upper() in ['WV3', 'WV2'] or num_bands == 8:
        q_name = 'Q8'
    elif sensor.upper() in ['QB', 'GF2', 'IKONOS'] or num_bands == 4:
        q_name = 'Q4'
    else:
        q_name = f'Q{num_bands}'
    
    metrics = {}
    metrics['PSNR'] = psnr(gt, sr, dynamic_range)
    metrics['SSIM'] = ssim(gt, sr, dynamic_range)
    metrics['SAM'] = sam(gt, sr)
    metrics['ERGAS'] = ergas(sr, gt, scale)
    metrics[q_name] = qnr_index(gt, sr, sensor, block_size)
    
    return metrics


def evaluate_mat_file(mat_path, dynamic_range=2047, scale=4, block_size=32, sensor='WV3'):
    """
    评估.mat文件中的结果
    
    Args:
        mat_path: .mat文件路径，应包含'sr'和'gt'字段
        dynamic_range: 动态范围
        scale: 下采样倍率  
        block_size: Q2n的块大小（默认32）
        sensor: 传感器类型 (WV3/WV2: Q8, QB/GF2: Q4)
    """
    print(f"Loading results from: {mat_path}")
    data = sio.loadmat(mat_path)
    
    if 'sr' not in data or 'gt' not in data:
        raise ValueError("Mat file must contain 'sr' and 'gt' fields")
    
    sr_images = data['sr']  # shape: (N, C, H, W) or similar
    gt_images = data['gt']
    
    print(f"SR shape: {sr_images.shape}")
    print(f"GT shape: {gt_images.shape}")
    
    # 处理数据格式
    if isinstance(sr_images, np.ndarray):
        if sr_images.ndim == 4:  # (N, C, H, W)
            num_images = sr_images.shape[0]
        else:
            num_images = 1
            sr_images = sr_images[np.newaxis, ...]
            gt_images = gt_images[np.newaxis, ...]
    else:
        # 如果是list或cell array
        sr_images = np.array(sr_images)
        gt_images = np.array(gt_images)
        num_images = len(sr_images)
    
    # 根据传感器确定 Q 指标名称
    sample_gt = gt_images[0] if num_images > 1 else gt_images[0]
    sample_gt = np.squeeze(sample_gt)
    if sample_gt.ndim == 3:
        num_bands = sample_gt.shape[2] if sample_gt.shape[2] < sample_gt.shape[0] else sample_gt.shape[0]
    else:
        num_bands = 1
    
    if sensor.upper() in ['WV3', 'WV2'] or num_bands == 8:
        q_name = 'Q8'
    elif sensor.upper() in ['QB', 'GF2', 'IKONOS'] or num_bands == 4:
        q_name = 'Q4'
    else:
        q_name = f'Q{num_bands}'
    
    print(f"\nEvaluating {num_images} images...")
    print(f"Using {q_name} (Q2n) index for {sensor} sensor with {num_bands} bands")
    print("=" * 80)
    
    all_metrics = {
        'PSNR': [],
        'SSIM': [],
        'SAM': [],
        'ERGAS': [],
        q_name: []
    }
    
    for i in range(num_images):
        sr = sr_images[i] if num_images > 1 else sr_images[0]
        gt = gt_images[i] if num_images > 1 else gt_images[0]
        
        # 去除多余的维度
        sr = np.squeeze(sr)
        gt = np.squeeze(gt)
        
        metrics = evaluate_single_image(sr, gt, dynamic_range, scale, block_size, sensor)
        
        print(f"\nImage {i+1}/{num_images}:")
        print(f"  PSNR:  {metrics['PSNR']:.4f} dB")
        print(f"  SSIM:  {metrics['SSIM']:.4f}")
        print(f"  SAM:   {metrics['SAM']:.4f}")
        print(f"  ERGAS: {metrics['ERGAS']:.4f}")
        print(f"  {q_name}:    {metrics[q_name]:.4f}")
        
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
    
    # 计算平均值和标准差
    print("\n" + "=" * 80)
    print("Average Metrics:")
    print("=" * 80)
    for key in all_metrics:
        values = np.array(all_metrics[key])
        mean_val = values.mean()
        std_val = values.std()
        print(f"{key:6s}: {mean_val:.4f} ± {std_val:.4f}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='评估全景锐化结果')
    parser.add_argument('--mat_file', type=str, required=True,
                        help='.mat文件路径，包含sr和gt字段')
    parser.add_argument('--dynamic_range', type=int, default=2047,
                        help='动态范围 (WV3: 2047, GF2: 1023)')
    parser.add_argument('--scale', type=int, default=4,
                        help='下采样倍率')
    parser.add_argument('--block_size', type=int, default=32,
                        help='Q2n的块大小')
    parser.add_argument('--sensor', type=str, default='WV3',
                        choices=['WV2', 'WV3', 'QB', 'GF2', 'IKONOS'],
                        help='传感器类型 (WV3/WV2: Q8, QB/GF2: Q4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mat_file):
        print(f"Error: File not found: {args.mat_file}")
        return
    
    evaluate_mat_file(args.mat_file, args.dynamic_range, args.scale, args.block_size, args.sensor)


if __name__ == '__main__':
    main()

