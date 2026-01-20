import os
import time
from datetime import datetime

import cv2
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

from preprocess import apply_enhancement
from ocr_engine import ocr_image
from evaluate import char_accuracy, levenshtein_distance

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "static", "uploads")
RESULT_DIR = os.path.join(APP_DIR, "static", "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)

METHODS = [
    ("original", "Original"),
    ("brightness", "Brightness"),
    ("contrast", "Contrast"),
    ("clahe", "CLAHE (Local Contrast)"),
    ("denoise_gaussian", "Denoise (Gaussian)"),
    ("denoise_median", "Denoise (Median)"),
    ("sharpen", "Sharpen"),
    ("thresh_otsu", "Threshold (Otsu)"),
]

@app.get("/")
def index():
    return render_template("index.html", methods=METHODS)

@app.post("/run")
def run_ocr():
    file = request.files.get("image")
    method = request.form.get("method", "original")
    lang = "eng"  # ✅ 只做英文
    gt_text = request.form.get("ground_truth", "").strip()

    if not file or file.filename.strip() == "":
        return redirect(url_for("index"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts}_{file.filename}"
    in_path = os.path.join(UPLOAD_DIR, filename)
    file.save(in_path)

    bgr = cv2.imread(in_path)
    if bgr is None:
        return redirect(url_for("index"))

    # 影像增強
    enhanced = apply_enhancement(bgr, method)

    # 存增強後圖片
    out_img_name = f"enh_{filename}.png"
    out_path = os.path.join(RESULT_DIR, out_img_name)
    cv2.imwrite(out_path, enhanced)

    # OCR
    t0 = time.time()
    pred = ocr_image(enhanced, lang=lang, psm=6)
    t_ms = (time.time() - t0) * 1000.0

    # 評估（如果有輸入 Ground Truth）
    acc = None
    dist = None
    if gt_text:
        acc = char_accuracy(gt_text, pred)
        dist = levenshtein_distance(gt_text, pred)

    # 存到 results.csv（每次跑都會追加一筆）
    row = {
        "timestamp": ts,
        "filename": filename,
        "method": method,
        "lang": lang,
        "processing_ms": round(t_ms, 2),
        "ground_truth": gt_text,
        "ocr_text": pred,
        "char_accuracy": (round(acc, 4) if acc is not None else ""),
        "edit_distance": (dist if dist is not None else ""),
    }

    csv_path = os.path.join(APP_DIR, "results.csv")
    df = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return render_template(
        "index.html",
        methods=METHODS,
        in_image=url_for("static", filename=f"uploads/{filename}"),
        out_image=url_for("static", filename=f"results/{out_img_name}"),
        method=method,
        ground_truth=gt_text,
        pred_text=pred,
        processing_ms=round(t_ms, 2),
        char_accuracy=(round(acc * 100, 2) if acc is not None else None),
        edit_distance=dist,
        results_csv="results.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)
