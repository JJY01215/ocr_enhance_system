import os
import glob
import cv2
import numpy as np

# ===== 你可以改這些參數 =====
INPUT_DIR = os.path.join("dataset", "A_clear")
OUT_B = os.path.join("dataset", "B_lowlight")
OUT_C = os.path.join("dataset", "C_blur")
OUT_D = os.path.join("dataset", "D_complex")

BRIGHTNESS_DELTA = -60   # B 低光：-60
BLUR_RADIUS = 20        # C 模糊：radius=4
NOISE_STD = 90      # D 背景複雜：雜訊強度（建議 10~25）
# ===========================

def ensure_dirs():
    for d in [OUT_B, OUT_C, OUT_D]:
        os.makedirs(d, exist_ok=True)

def lowlight(bgr: np.ndarray, delta: int) -> np.ndarray:
    """用亮度偏移做低光：每個像素加上 delta (負數變暗)"""
    out = cv2.convertScaleAbs(bgr, alpha=1.0, beta=delta)
    return out

def gaussian_blur(bgr: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian blur：OpenCV 用 kernel size = 2*radius+1"""
    k = 2 * radius + 1
    return cv2.GaussianBlur(bgr, (k, k), 0)

def add_noise(bgr: np.ndarray, std: float) -> np.ndarray:
    """加高斯雜訊，模擬背景變髒、感光雜訊"""
    noise = np.random.normal(0, std, bgr.shape).astype(np.float32)
    out = bgr.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def main():
    ensure_dirs()
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")

    files = []
    for e in exts:
        files += glob.glob(os.path.join(INPUT_DIR, e))

    if not files:
        print(f"[!] No images found in: {INPUT_DIR}")
        return

    print(f"[+] Found {len(files)} images in {INPUT_DIR}")

    for p in files:
        img = cv2.imread(p)
        if img is None:
            print(f"[!] Skip (cannot read): {p}")
            continue

        base = os.path.splitext(os.path.basename(p))[0]

        # B: low light
        b = lowlight(img, BRIGHTNESS_DELTA)
        cv2.imwrite(os.path.join(OUT_B, f"{base}_B_lowlight.png"), b)

        # C: blur
        c = gaussian_blur(img, BLUR_RADIUS)
        cv2.imwrite(os.path.join(OUT_C, f"{base}_C_blur.png"), c)

        # D: complex (noise)
        d = add_noise(img, NOISE_STD)
        cv2.imwrite(os.path.join(OUT_D, f"{base}_D_complex.png"), d)

    print("[✓] Done. Output saved to:")
    print("   ", OUT_B)
    print("   ", OUT_C)
    print("   ", OUT_D)

if __name__ == "__main__":
    main()
