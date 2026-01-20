import cv2
import numpy as np

def apply_enhancement(bgr: np.ndarray, method: str) -> np.ndarray:
    """
    bgr: OpenCV BGR image
    method:
      original | brightness | contrast | clahe |
      denoise_gaussian | denoise_median |
      sharpen | thresh_otsu
    """
    if method == "original":
        return bgr

    img = bgr.copy()

    if method == "brightness":
        # 用 HSV 的 V 通道調亮度比較穩
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 40)  # 可調整：20~60
        hsv2 = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    if method == "contrast":
        # alpha > 1 增強對比，beta 微調亮度
        alpha = 1.4
        beta = 10
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if method == "clahe":
        # 局部對比增強：低光影像常有效
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    if method == "denoise_gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)

    if method == "denoise_median":
        return cv2.medianBlur(img, 5)

    if method == "sharpen":
        # 銳化不要太強，避免筆畫斷掉
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    if method == "thresh_otsu":
        # 二值化：先轉灰階 + Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unknown method: {method}")
