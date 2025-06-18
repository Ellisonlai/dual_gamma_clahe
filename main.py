import os
import argparse
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from dg_clahe import dual_gamma_clahe
from dg_clahe_rgb import dual_gamma_clahe_rgb
from enhance_methods import clahe, he, proposed, evaluate_all, he_rgb, clahe_rgb, clahe_rgb_all
from skimage import io



def save_comparison_figure(original, enhanced_dict, filename, output_dir="output"):
    fig, axes = plt.subplots(1, len(enhanced_dict)+1, figsize=(5*(len(enhanced_dict)+1), 4))
    methods = ["Original"] + list(enhanced_dict.keys())
    images = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if original.ndim == 3 else original] + [
        cv2.cvtColor(enhanced_dict[m], cv2.COLOR_BGR2RGB) if enhanced_dict[m].ndim == 3 else enhanced_dict[m]
        for m in enhanced_dict
    ]
    for ax, img, title in zip(axes, images, methods):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    out_path = os.path.join(output_dir, f"Comparison_{filename}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def process_images(input_dir, output_dir, method, save_compare=True, color_space="all"):
    """
    color space: 'gray' or 'rgb' or 'all'
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    gray_map = {
        "proposed": proposed,
        "clahe": clahe,
        "he": he,
    }
    rgb_map = {
        "HE": he_rgb,
        "CLAHE": clahe_rgb,
        # "clahe_rgb_all": clahe_rgb_all,
        "The Reproduced ": dual_gamma_clahe,
    }
    method_map = {}

    if color_space == "gray":
        method_map.update(gray_map)
    elif color_space == "rgb":
        method_map.update(rgb_map)
    elif color_space == "all":
        method_map.update(gray_map)
        method_map.update(rgb_map)

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
            for m in method_map.keys():
                image = cv2.imread(img_path)
                out_name = f"{m}_{filename}"
                out_path = os.path.join(output_dir, out_name)
                scores = {}
                enhanced = None
                enhanced = method_map[m](image.copy())
                cv2.imwrite(out_path, enhanced)
                
                enhanced = cv2.imread(out_path)
                #cv2.imwrite(out_path, enhanced)
                scores = evaluate_all(image, enhanced)
                scores["Image"] = filename
                scores["Method"] = m
                results.append(scores)
                enhanced_dict[m] = enhanced
                save_comparison_figure(image, enhanced_dict, os.path.splitext(filename)[0], output_dir)

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
            print(f"image ndim: {image.ndim}")
            original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            save_comparison_figure(original_gray, enhanced_dict, os.path.splitext(filename)[0], output_dir)
    if not results:
        print("沒有成功處理任何圖片，請檢查 images 資料夾是否有有效圖片。")
        return
    df = pd.DataFrame(results)
    print(df[["Image", "Method", "TV", "AMBE", "EME"]])
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="images", help="Input images folder")
    parser.add_argument("--output_dir", default="output", help="Output folder")
    parser.add_argument("--method", default="all", choices=["dual_gamma_clahe", "proposed", "clahe", "he", "all"], help="Enhancement method")
    parser.add_argument("--compare", action="store_true", help="Save comparison figure")
    parser.add_argument("--color_space", default="rgb", choices=["gray", "rgb", "all"], help="Color space")
    args = parser.parse_args()
    process_images(args.input_dir, args.output_dir, args.method, args.compare, args.color_space)

if __name__ == "__main__":
    main()
