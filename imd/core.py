"""
IMD作製のための関数モジュール
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage.draw import disk
from skimage.restoration import rolling_ball
from skimage import measure
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.optimize import curve_fit
import cv2
from concurrent.futures import ThreadPoolExecutor
from skimage.restoration import rolling_ball
import napari

def create_imd(donor, acceptor, rmax=2.4, rmin=1.2, dmax=2000, dmin=10):
    """
    IMD (Intensity Modulated Display) 画像を生成
    
    Parameters:
    -----------
    donor : numpy.ndarray
        ドナー画像スタック (CFP)
    acceptor : numpy.ndarray
        アクセプター画像スタック (FRET)
    rmax, rmin : float
        Ratioの最大値と最小値
    dmax, dmin : float
        Donor強度の最大値と最小値
    
    Returns:
    --------
    numpy.ndarray
        RGB形式のIMD画像スタック
    """
    # 範囲の計算
    rrange = rmax - rmin
    drange = dmax - dmin
    
    # マスク作成（ドナー強度ベース）
    mask = donor.copy()
    mask = (mask - dmin) / drange
    mask = np.clip(mask, 0, 1)
    
    # Ratio画像の計算とスケーリング
    ratio = acceptor / donor
    ratio = (ratio - rmin) / rrange
    ratio = np.clip(ratio, 0, 1)
    ratio = (ratio * 255).astype(np.uint8)
    
    # physicスケールのカラーマップの作成（0-255の範囲で）
    colors = np.zeros((256, 3), dtype=np.uint8)
    # 赤成分
    colors[:, 0] = np.minimum(255, np.maximum(0, 255 * (2 - 4 * np.abs(np.linspace(0, 1, 256) - 0.75))))
    # 緑成分
    colors[:, 1] = np.minimum(255, np.maximum(0, 255 * (2 - 4 * np.abs(np.linspace(0, 1, 256) - 0.5))))
    # 青成分
    colors[:, 2] = np.minimum(255, np.maximum(0, 255 * (2 - 4 * np.abs(np.linspace(0, 1, 256) - 0.25))))
    
    # RGBチャンネルの作成
    rgb_stack = np.zeros((*ratio.shape, 3), dtype=np.uint8)
    for i in range(len(ratio)):
        # LUTの適用
        rgb = colors[ratio[i]]
        # マスクの適用
        for c in range(3):
            rgb_stack[i, :, :, c] = (rgb[:, :, c] * mask[i]).astype(np.uint8)
    
    return rgb_stack

def view_imd_with_napari(imd_stack, cfp_stack=None, fret_stack=None):
    """
    napariを使用してIMD画像とオプションでCFP、FRET画像を表示
    
    Parameters:
    -----------
    imd_stack : numpy.ndarray
        RGB形式のIMD画像スタック (frames, height, width, 3)
    cfp_stack : numpy.ndarray, optional
        CFP画像スタック
    fret_stack : numpy.ndarray, optional
        FRET画像スタック
    """
    # napariビューアーの作成
    viewer = napari.Viewer()
    
    # IMD画像の追加
    viewer.add_image(
        imd_stack,
        name='IMD',
        rgb=True
    )
    
    # オプションでCFPとFRET画像を追加
    if cfp_stack is not None:
        viewer.add_image(
            cfp_stack,
            name='CFP',
            colormap='blue',
            visible=False  # デフォルトは非表示
        )
    
    if fret_stack is not None:
        viewer.add_image(
            fret_stack,
            name='FRET',
            colormap='green',
            visible=False  # デフォルトは非表示
        )
    
    return viewer

        
def rolling_ball_background_correction(img_stack, radius=50):
    """Rolling Ball法を使用して不均一な背景を補正する関数"""
    
    if img_stack is None:
        print("画像データが読み込まれていません")
        return None
    
    corrected_stack = np.zeros_like(img_stack, dtype=np.float32)
    
    for i in range(img_stack.shape[0]):
        background = rolling_ball(img_stack[i], radius=radius)
        corrected_stack[i] = img_stack[i] - background
        
        if i % 10 == 0:
            print(f"Processing frame {i}/{img_stack.shape[0]}")
    
    return np.maximum(corrected_stack, 0)

def rolling_ball_background_correction_fast_with_options(img_stack, radius=50, scale_factor=1.0, use_gpu=False):
    """
    より多くのオプションを持つ高速なローリングボール背景補正
    
    Parameters:
    -----------
    img_stack : np.ndarray
        画像スタック
    radius : int
        ローリングボールの半径
    scale_factor : float
        処理前の画像縮小率（高速化のため）
    use_gpu : bool
        GPUを使用するかどうか（OpenCVがGPUサポート付きでビルドされている場合）
        
    Returns:
    --------
    np.ndarray
        背景補正済みの画像データ配列
    """
    
    def process_single_frame(frame):
        # 元のサイズを保存
        original_size = frame.shape
        
        # スケーリング（必要な場合）
        if scale_factor != 1.0:
            new_size = tuple(int(dim * scale_factor) for dim in frame.shape)
            frame = cv2.resize(frame, (new_size[1], new_size[0]))
        
        # カーネルサイズの計算（スケーリングを考慮）
        kernel_size = int(2 * radius * scale_factor) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if use_gpu:
            # GPUメモリに転送
            frame_gpu = cv2.UMat(frame)
            background_gpu = cv2.morphologyEx(frame_gpu, cv2.MORPH_OPEN, kernel)
            corrected_gpu = cv2.subtract(frame_gpu, background_gpu)
            corrected = cv2.UMat.get(corrected_gpu)
        else:
            background = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            corrected = frame - background
        
        # 元のサイズに戻す（必要な場合）
        if scale_factor != 1.0:
            corrected = cv2.resize(corrected, (original_size[1], original_size[0]))
        
        return np.maximum(corrected, 0)
    
    # データ型の変換
    img_stack = img_stack.astype(np.float32)
    
    # 並列処理
    with ThreadPoolExecutor() as executor:
        corrected_stack = list(executor.map(process_single_frame, img_stack))
    
    return np.array(corrected_stack)
    
def compare_background_correction(original, corrected, frame_idx=0):
    """補正前後の画像を比較表示する関数"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    vmin1 = np.percentile(original[frame_idx], 1)
    vmax1 = np.percentile(original[frame_idx], 99)
    im1 = ax1.imshow(original[frame_idx], cmap='gray', vmin=vmin1, vmax=vmax1)
    ax1.set_title('Original')
    plt.colorbar(im1, ax=ax1)
    ax1.axis('off')
    
    vmin2 = np.percentile(corrected[frame_idx], 1)
    vmax2 = np.percentile(corrected[frame_idx], 99)
    im2 = ax2.imshow(corrected[frame_idx], cmap='gray', vmin=vmin2, vmax=vmax2)
    ax2.set_title('Background Corrected')
    plt.colorbar(im2, ax=ax2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    