import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from ip_adapter.ip_adapter import ImageProjModel
from train import BaseAttnProcessor, SelfAttnWithIPProcessor, SkipCrossAttnProcessor
import glob


class IPAdapterInference:
    def __init__(self,unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
    # ... [保持原有初始化代码不变] ...

    def process_test_folder(
            self,
            test_dir: str,
            output_dir: str,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5
    ):
        """
        处理整个测试文件夹
        test_dir/
          ├── image/         # 人物图像
          ├── cloth/        # 衣物图像
          └── agnostic-mask # 分割掩码
        """
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有测试样本
        person_images = sorted(glob.glob(os.path.join(test_dir, "image", "*.jpg")))

        for person_path in person_images:
            # 构造对应路径
            base_name = os.path.basename(person_path)
            cloth_path = os.path.join(test_dir, "cloth", base_name)
            mask_path = os.path.join(test_dir, "agnostic-mask", base_name.replace(".jpg", "_mask.png"))

            if not os.path.exists(cloth_path):
                print(f"Warning: Missing cloth image for {base_name}")
                continue

            if not os.path.exists(mask_path):
                print(f"Warning: Missing mask for {base_name}")
                continue

            # 生成输出路径
            output_path = os.path.join(output_dir, f"output_{base_name}")

            # 执行生成
            self.generate_single(
                person_path=person_path,
                cloth_path=cloth_path,
                mask_path=mask_path,
                output_path=output_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

    def generate_single(self, person_path, cloth_path, mask_path, output_path, ​ ** ​kwargs):
        """处理单个样本的生成逻辑"""
        try:
            # 预处理输入
            inputs = self.preprocess_images(person_path, cloth_path, mask_path)

            with torch.no_grad():
                # ... [保持原有生成逻辑不变] ...

                # 保存结果
                Image.fromarray(generated_image).save(output_path)
                print(f"Successfully generated {output_path}")

        except Exception as e:
            print(f"Error processing {person_path}: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test folder containing image/cloth/agnostic-mask")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    # ... [其他参数保持原有定义不变] ...


if __name__ == "__main__":
    args = parse_args()

    # 验证测试文件夹结构
    required_subdirs = ["image", "cloth", "agnostic-mask"]
    for subdir in required_subdirs:
        if not os.path.exists(os.path.join(args.test_dir, subdir)):
            raise ValueError(f"Missing subdirectory: {subdir} in test folder")

    generator = IPAdapterInference(args)

    print(f"Processing test folder: {args.test_dir}")
    generator.process_test_folder(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    )
    print(f"All results saved to {args.output_dir}")