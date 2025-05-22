from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size=(384, 512)):
    """
    批量调整图片尺寸
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param target_size: 目标尺寸 (width, height)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        try:
            # 读取图片
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path)

            # 转换为RGB模式（处理可能存在的alpha通道）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 调整尺寸
            resized_img = img.resize(target_size, Image.LANCZOS)

            # 保存图片
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)
            print(f"已处理: {filename}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

# 使用示例
input_dir = "/mnt/sda1/lzx/stabledata/zalando-hd-resized/test/image"  # 替换为你的输入文件夹路径
output_dir = "/home/liangruyu/catvton/test_image_384_512"  # 替换为输出文件夹路径

resize_images(input_dir, output_dir, target_size=(384, 512))