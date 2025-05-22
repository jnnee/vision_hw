from PIL import Image
import os
import argparse


def extract_right_image(input_path, output_folder):
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 自动识别单文件或文件夹
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for img_path in files:
        with Image.open(img_path) as img:
            # 智能尺寸检测
            width, height = img.size
            unit_width = width // 3  # 三等分

            # 右三分之一裁剪(兼容留白误差)
            right_box = (
                int(unit_width * 2 ),  # 左边界：从2/3处前移10像素
                0,  # 上边界
                width - 1,  # 右边界(防止越界)
                height  # 下边界
            )
            right_img = img.crop(right_box)

            # 自动生成文件名
            basename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, f"right_{basename}")
            right_img.save(output_path)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        help='输入文件/文件夹路径',default="/home/liangruyu/catvton/output_noip_384-512_dim-2_dream_######_110000_test/vitonhd-512/unpaired")
    parser.add_argument('--output', type=str, default='/home/liangruyu/catvton/output_noip_384-512_dim-2_dream_######_110000_test_cut/vitonhd-512/unpaired',
                        help='输出文件夹路径')
    args = parser.parse_args()

    extract_right_image(args.input, args.output)