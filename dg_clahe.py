import numpy as np
import cv2
from typing import List, Union, Tuple

def compute_clip_limit(
    block: np.ndarray, alpha: float = 40, pi: float = 1.5, R: int = 255
) -> int:
    avg = block.mean()
    l_max = block.max()
    n = l_max - block.min()
    sigma = block.std()
    m = block.size
    return int(
        (m / n) * (1 + (pi * l_max / R) + (alpha * sigma / 100.0) / (avg + 0.001))
    )

def clip_and_redistribute_hist(hist: np.ndarray, beta: int) -> np.ndarray:
    mask = hist > beta
    exceed_values = hist[mask]
    bin_value = (exceed_values.sum() - exceed_values.size * beta) // hist.size
    hist[mask] = beta
    hist += bin_value
    return hist

def compute_gamma(l_max: int, d_range: int, weighted_cdf: np.ndarray) -> np.ndarray:
    return l_max * np.power(np.arange(d_range + 1) / l_max, (1 + weighted_cdf) / 2.0)

def compute_w_en(l_alpha: int, l_max: int, cdf: np.ndarray) -> np.ndarray:
    return np.power(l_max / l_alpha, 1 - (np.log(np.e + cdf) / 8))

def compute_block_coords(center_block: int, block_index: int, index: int, n_blocks: int) -> tuple:
    if index < center_block:
        block_min = block_index - 1
        if block_min < 0:
            block_min = 0
        block_max = block_index
    else:
        block_min = block_index
        block_max = block_index + 1
        if block_max >= n_blocks:
            block_max = block_index
    return block_min, block_max

def compute_mn_factors(coords: tuple, index: int) -> float:
    if coords[1] - coords[0] == 0:
        return 0
    else:
        return (coords[1] - index) / (coords[1] - coords[0])

def dual_gamma_clahe(
    image: np.ndarray,
    block_size: Union[int, List] = [32, 32],
    alpha: float = 20,
    pi: float = 1.5,
    delta: float = 50,
    bins: int = 256,
):
    ndim = image.ndim
    R = bins - 1
    if ndim == 3 and image.shape[2] == 3:
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        gray_image = yuv_image[:, :, 0].copy()
    else:
        gray_image = image.copy()
    gray_image = gray_image.astype(np.float32)
    height, width = gray_image.shape
    if isinstance(block_size, int):
        block_size = [block_size, block_size]
    h_block, w_block = block_size
    glob_l_max = gray_image.max()
    glob_hist = np.histogram(gray_image, bins=bins)[0]
    glob_cdf = np.cumsum(glob_hist)
    glob_cdf = glob_cdf / glob_cdf[-1]
    glob_l_a = np.argwhere(glob_cdf > 0.75)[0]
    pad_h = h_block // 2
    pad_w = w_block // 2
    padded = np.pad(gray_image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    pad_height, pad_width = padded.shape[:2]
    n_height_blocks = int(pad_height / h_block)
    n_width_blocks = int(pad_width / w_block)
    hists = np.zeros((n_height_blocks, n_width_blocks, bins))
    beta_thresholds = np.zeros((n_height_blocks, n_width_blocks))
    result = np.zeros_like(padded)
    for ii in range(n_height_blocks):
        for jj in range(n_width_blocks):
            max_val_block = padded[
                ii : ii + h_block, jj : jj + w_block
            ].max()
            r_block = (
                max_val_block
                - padded[ii : ii + h_block, jj : jj + w_block].min()
            )
            hists[ii, jj] = np.histogram(
                padded[ii : ii + h_block, jj : jj + w_block], bins=bins
            )[0]
            beta_thresholds[ii, jj] = compute_clip_limit(
                padded[ii : ii + h_block, jj : jj + w_block],
                alpha=alpha,
                pi=pi,
                R=R,
            )
            hists[ii, jj] = clip_and_redistribute_hist(
                hists[ii, jj], beta_thresholds[ii, jj]
            )
            pdf_min = hists[ii, jj].min()
            pdf_max = hists[ii, jj].max()
            weighted_hist = pdf_max * (hists[ii, jj] - pdf_min) / (pdf_max - pdf_min)
            weighted_cum_hist = np.cumsum(weighted_hist)
            pdf_sum = weighted_cum_hist[-1]
            weighted_cum_hist /= pdf_sum
            hists[ii, jj] = np.cumsum(hists[ii, jj])
            norm_cdf = hists[ii, jj] / hists[ii, jj, -1]
            w_en = compute_w_en(l_max=glob_l_max, l_alpha=glob_l_a, cdf=norm_cdf)
            tau_1 = max_val_block * w_en * norm_cdf
            gamma = compute_gamma(
                l_max=glob_l_max, d_range=R, weighted_cdf=weighted_cum_hist
            )
            if r_block > delta:
                hists[ii, jj] = np.maximum(tau_1, gamma)
            else:
                hists[ii, jj] = gamma
    for i in range(pad_height):
        for j in range(pad_width):
            p_i = int(padded[i][j])
            block_x = i // h_block
            block_y = j // w_block
            center_block_x = block_x * h_block + n_height_blocks // 2
            center_block_y = block_y * w_block + n_width_blocks // 2
            block_y_a, block_y_c = compute_block_coords(
                center_block_x, block_x, i, n_height_blocks
            )
            block_x_a, block_x_b = compute_block_coords(
                center_block_y, block_y, j, n_width_blocks
            )
            y_a = block_y_c * h_block + h_block // 2
            y_c = block_y_a * h_block + h_block // 2
            x_a = block_x_a * w_block + w_block // 2
            x_b = block_x_b * w_block + w_block // 2
            m = compute_mn_factors((y_a, y_c), i)
            n = compute_mn_factors((x_a, x_b), j)
            def safe_idx(idx, maxval):
                return min(max(idx, 0), maxval-1)
            block_y_c = safe_idx(block_y_c, n_height_blocks)
            block_y_a = safe_idx(block_y_a, n_height_blocks)
            block_x_a = safe_idx(block_x_a, n_width_blocks)
            block_x_b = safe_idx(block_x_b, n_width_blocks)
            p_i_safe = min(max(int(p_i), 0), bins-1)
            Ta = hists[block_y_c, block_x_a, p_i_safe]
            Tb = hists[block_y_c, block_x_b, p_i_safe]
            Tc = hists[block_y_a, block_x_a, p_i_safe]
            Td = hists[block_y_a, block_x_b, p_i_safe]
            result[i, j] = int(
                m * (n * Ta + (1 - n) * Tb) + (1 - m) * (n * Tc + (1 - n) * Td)
            )
    result = result[pad_h:pad_h+height, pad_w:pad_w+width]
    result = np.clip(result, 0, 255).astype(np.uint8)
    if ndim == 3:
        yuv_result = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_result[:, :, 0] = result
        return cv2.cvtColor(yuv_result, cv2.COLOR_YUV2BGR)
    else:
        return result

def clahe_full(image: np.ndarray, block_size: Tuple[int, int] = (32, 32), alpha: float = 40, pi: float = 1.5, bins: int = 256) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = gray.astype(np.uint8)
    height, width = gray.shape
    h_block, w_block = block_size
    pad_h = h_block // 2
    pad_w = w_block // 2
    padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    h, w = padded.shape
    n_h_blocks = h // h_block
    n_w_blocks = w // w_block
    hist_map = np.zeros((n_h_blocks, n_w_blocks, bins))
    for i in range(n_h_blocks):
        for j in range(n_w_blocks):
            block = padded[i*h_block:(i+1)*h_block, j*w_block:(j+1)*w_block]
            beta = compute_clip_limit(block, alpha, pi)
            hist = np.histogram(block.flatten(), bins=bins, range=(0, 255))[0]
            hist = clip_and_redistribute_hist(hist, beta)
            cdf = hist.cumsum()
            cdf = 255 * cdf / cdf[-1]
            hist_map[i, j] = cdf
    result = np.zeros_like(padded)
    for y in range(h):
        for x in range(w):
            val = padded[y, x]
            i = y // h_block
            j = x // w_block
            i1 = min(i, n_h_blocks - 2)
            j1 = min(j, n_w_blocks - 2)
            dy = (y % h_block) / h_block
            dx = (x % w_block) / w_block
            tl = hist_map[i1, j1, val]
            tr = hist_map[i1, j1 + 1, val]
            bl = hist_map[i1 + 1, j1, val]
            br = hist_map[i1 + 1, j1 + 1, val]
            top = tl * (1 - dx) + tr * dx
            bottom = bl * (1 - dx) + br * dx
            result[y, x] = top * (1 - dy) + bottom * dy
    result = result[pad_h:pad_h+height, pad_w:pad_w+width]
    return result.astype(np.uint8)
