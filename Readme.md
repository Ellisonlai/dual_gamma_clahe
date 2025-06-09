# Dual Gamma CLAHE & Image Enhancement Toolkit

This project implements and extends the algorithm from the paper:

> **Automatic Contrast-Limited Adaptive Histogram Equalization With Dual Gamma Correction**  
> [IEEE Xplore Link](https://ieeexplore.ieee.org/document/8269243)

---
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
## Features

- **Dual Gamma CLAHE** (paper method, robust block-based, YUV/gray)
- **Proposed (dual gamma+CLAHE)**, **OpenCV CLAHE**, **Histogram Equalization (HE)**
- **Batch processing**: Enhance all images in a folder
- **Robust evaluation metrics**: TV, AMBE, EME, CQE
- **Visual comparison**: Auto-generate comparison figures for all methods
- **Flexible CLI**: Choose method, batch/single, output, and comparison

---

## Installation

1. 建立虛擬環境並安裝依賴：

```bash
python -m venv env
# Windows:
./env/Scripts/activate
# macOS/Linux:
source env/bin/activate
pip install -r requirements.txt
```

---

## Usage

批次處理 `images` 資料夾下所有圖片，並產生增強結果、評估指標與比較圖：

```bash
python main.py --input_dir images --output_dir output --method all --compare
```

- `--method` 可選：`dual_gamma_clahe`、`proposed`、`clahe`、`he`、`all`
- `--compare` 會自動產生原圖+各方法結果的比較圖

### 只處理單一方法

```bash
python main.py --input_dir images --output_dir output --method dual_gamma_clahe
```

### 參數說明

- `--input_dir`：輸入圖片資料夾
- `--output_dir`：輸出結果資料夾
- `--method`：增強方法（`dual_gamma_clahe`/`proposed`/`clahe`/`he`/`all`）
- `--compare`：是否產生比較圖

---

## Results

- 增強後圖片與比較圖會存於 `output` 資料夾
- 評估指標（TV, AMBE, EME, CQE）會自動輸出成 `results.csv`

---

## Reference

本專案核心演算法基於：

> **Automatic Contrast-Limited Adaptive Histogram Equalization With Dual Gamma Correction**  
> [IEEE Xplore Link](https://ieeexplore.ieee.org/document/8269243)

如需引用，請引用原論文。

---

## Example Output

> Comparison Example
>
> ![](./output/Comparison_example.png)

---

## Contact

有任何問題或建議，歡迎開 issue 或聯絡作者。
