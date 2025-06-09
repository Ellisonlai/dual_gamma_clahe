import cv2
import numpy as np

def calculate_tv(image: np.ndarray) -> float:
    dx = np.abs(np.diff(image.astype(float), axis=1))
    dy = np.abs(np.diff(image.astype(float), axis=0))
    return float(np.sum(dx) + np.sum(dy)) / image.size

def calculate_ambe(original: np.ndarray, enhanced: np.ndarray) -> float:
    return float(abs(original.mean() - enhanced.mean()))

def calculate_eme(image: np.ndarray, k1=8, k2=8, c=1e-5) -> float:
    h, w = image.shape
    block_h, block_w = h // k1, w // k2
    eme = 0
    for i in range(k1):
        for j in range(k2):
            block = image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            Imax, Imin = block.max(), block.min()
            if Imin == 0:
                eme += 20 * np.log10((Imax + c) / c)
            else:
                eme += 20 * np.log10((Imax + c) / (Imin + c))
    return float(eme / (k1 * k2))

def calculate_cqe(tv, ambe, eme, w1=0.3, w2=0.3, w3=0.4) -> float:
    tv_score = 1 / (1 + tv)
    ambe_score = 1 / (1 + ambe)
    eme_score = eme / 100
    return round(w1 * tv_score + w2 * ambe_score + w3 * eme_score, 4)

def evaluate_all(original: np.ndarray, enhanced: np.ndarray):
    if original.ndim == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if enhanced.ndim == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    tv = calculate_tv(enhanced)
    ambe = calculate_ambe(original, enhanced)
    eme = calculate_eme(enhanced)
    cqe = calculate_cqe(tv, ambe, eme)
    return {
        "TV": round(tv, 4),
        "AMBE": round(ambe, 4),
        "EME": round(eme, 4),
        "CQE": round(cqe, 4),
    }

def clahe(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_obj.apply(gray)

def he(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return cv2.equalizeHist(gray)

def proposed(image: np.ndarray, gamma_dark: float = 0.7, gamma_bright: float = 1.5) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    norm = gray.astype(np.float32) / 255.0
    dark = np.power(norm, gamma_dark)
    bright = np.power(norm, gamma_bright)
    weight = 1.0 / (1.0 + np.exp(-10 * (norm - 0.5)))
    fused = (1 - weight) * dark + weight * bright
    fused_uint8 = (fused * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(fused_uint8)
    return result
