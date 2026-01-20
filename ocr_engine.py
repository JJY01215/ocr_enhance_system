import os
import pytesseract
import numpy as np
import cv2

# ✅ 你的 tesseract 安裝路徑（你已成功安裝在這裡）
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def setup_tesseract():
    if os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    else:
        # 如果找不到，會讓你知道要改路徑
        raise FileNotFoundError(f"Cannot find tesseract.exe at: {TESSERACT_CMD}")

setup_tesseract()

def ocr_image(bgr: np.ndarray, lang: str = "eng", psm: int = 6) -> str:
    """
    lang: eng (英文)
    psm: Page Segmentation Mode
         6 = Assume a uniform block of text (適合一般句子/段落)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(gray, lang=lang, config=config)

    return text.strip()
