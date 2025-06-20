import os
import argparse
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from dg_clahe import dual_gamma_clahe
from enhance_methods import clahe, he, proposed, evaluate_all

def save_comparison_figure(original, enhanced_dict, filename, output_dir="output"):
    fig, axes = plt.subplots(1, len(enhanced_dict)+1, figsize=(5*(len(enhanced_dict)+1), 4))
    methods = ["Original"] + list(enhanced_dict.keys())
    images = [original] + [enhanced_dict[m] for m in enhanced_dict]
    for ax, img, title in zip(axes, images, methods):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    out_path = os.path.join(output_dir, f"Comparison_{filename}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def process_images(input_dir, output_dir, method, save_compare=False):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    method_map = {
        "dual_gamma_clahe": dual_gamma_clahe,
        "proposed": proposed,
        "clahe": clahe,
        "he": he
    }
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(img_extensions):
            continue
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"無法讀取圖片：{filename}")
            continue
        enhanced_dict = {}
        if method == "all":
            for m in ["dual_gamma_clahe", "proposed", "clahe", "he"]:
                enhanced = method_map[m](image.copy())
                out_name = f"{m}_{filename}"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, enhanced)
                scores = evaluate_all(image, enhanced)
                scores["Image"] = filename
                scores["Method"] = m
                results.append(scores)
                enhanced_dict[m] = enhanced
        else:
            enhanced = method_map[method](image.copy())
            out_name = f"{method}_{filename}"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, enhanced)
            scores = evaluate_all(image, enhanced)
            scores["Image"] = filename
            scores["Method"] = method
            results.append(scores)
            enhanced_dict[method] = enhanced
        # 儲存比較圖
        if save_compare:
            original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            save_comparison_figure(original_gray, enhanced_dict, os.path.splitext(filename)[0], output_dir)
    if not results:
        print("沒有成功處理任何圖片，請檢查 images 資料夾是否有有效圖片。")
        return
    df = pd.DataFrame(results)
    print(df[["Image", "Method", "TV", "AMBE", "EME", "CQE"]])
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="images", help="Input images folder")
    parser.add_argument("--output_dir", default="output", help="Output folder")
    parser.add_argument("--method", default="all", choices=["dual_gamma_clahe", "proposed", "clahe", "he", "all"], help="Enhancement method")
    parser.add_argument("--compare", action="store_true", help="Save comparison figure")
    args = parser.parse_args()
    process_images(args.input_dir, args.output_dir, args.method, args.compare)

if __name__ == "__main__":
    main()
