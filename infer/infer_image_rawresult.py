import argparse
import sys
from pathlib import Path
import cv2

# 直接导入 YOLO
from ultralytics import YOLO

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser(description="只可视化 YOLO 2D 边界框和关键点")
    parser.add_argument("--input", required=True, help="输入图片文件或文件夹的路径")
    parser.add_argument("--output", default="infer_out_2d", help="输出文件夹路径")
    parser.add_argument("--weights", default="best.pt", help="YOLO 模型权重文件路径")
    args = parser.parse_args()

    # 1. 加载 YOLO 模型
    print(f"正在加载模型: {args.weights}...")
    model = YOLO(args.weights)

    # 2. 解析输入路径
    in_path = Path(args.input)
    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    # 3. 创建输出目录
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. 开始逐张推理与可视化
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"[跳过] 无法读取图片: {f}")
            continue

        # 将图片送入模型进行纯 2D 推理
        # 返回的是一个包含单个元素(因为只输入了一张图)的列表
        results = model(frame)

        # 核心步骤：使用 YOLO 原生的 plot() 方法
        # 该方法会自动在画面上画出预测的矩形框、类别置信度以及关键点连线
        annotated_frame = results[0].plot()

        # 生成输出路径并保存
        out_file_path = out_dir / f"{f.stem}_annot2d.jpg"
        cv2.imwrite(str(out_file_path), annotated_frame)
        print(f"[成功] 结果已保存至: {out_file_path}")

if __name__ == "__main__":
    main()