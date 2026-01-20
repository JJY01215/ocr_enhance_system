# OCR 影像增強輔助文字辨識系統（OCR Image Enhancement System）

本專案是一套以 **Tesseract OCR** 為核心，結合 **OpenCV 影像前處理**的光學文字辨識系統，目標是分析不同影像增強方法在 **低光、模糊、雜訊**等情境下，對 OCR 辨識準確率的影響。  
系統提供網頁操作介面，可上傳圖片、選擇處理方式並輸出辨識結果，同時可輸入 Ground Truth 計算 **字元正確率（Character Accuracy）**與 **編輯距離（Levenshtein Distance）**，並將結果記錄到 `results.csv` 以利後續統計分析與撰寫小論文。

---

## 功能特色
- 上傳圖片並進行 OCR 文字辨識
- 支援影像增強處理方法：
  - `original`（原始影像）
  - `contrast`（對比增強）
  - `clahe`（局部對比增強）
  - `thresh_otsu`（Otsu 二值化）
- 輸入 Ground Truth 後自動計算：
  - **Character Accuracy（字元正確率）**
  - **Edit Distance（編輯距離 / Levenshtein Distance）**
- 自動將每次測試結果記錄到 `results.csv`
- 支援批次生成影像版本（低光/模糊/雜訊）以建立測試資料集

---

## 專案結構

```bash
ocr_enhance_system/
├─ app.py
├─ preprocess.py
├─ ocr_engine.py
├─ evaluate.py
├─ make_variants.py
├─ requirements.txt
└─ templates/
   └─ index.html
