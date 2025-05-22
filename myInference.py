import os
import random
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from PIL import Image, ImageFilter
from torchvision import transforms
from myPipeline import MyCatVTONPipeline
from transformers import CLIPImageProcessor
'暂时先拿训练集来做测试'
# class VITONHDTrainDataset(Dataset):
#     def __init__(self, args):
#         super().__init__()  # 正确：无参数传递
#         self.args = args
#         self.data = self.load_data()  # 初始化时加载数据
#         self.i_drop_rate = 0.05
#         self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
#         self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
#                                                 do_convert_grayscale=True)
#         # 图像转换器（带归一化和增强）
#         self.img_transform = transforms.Compose([
#             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
#             transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ])
#         # 掩码转换器（无归一化）
#         self.mask_transform = transforms.Compose([
#             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
#             transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
#             transforms.ToTensor(),
#         ])
#         self.clip_image_processor = CLIPImageProcessor()
#     def __len__(self):
#         return len(self.data)
#     def load_data(self):
#         # 使用原始 data_root_path，不修改 self.args
#         data_root = os.path.join(self.args.data_dir, "train")  # 局部变量
#         #pair_txt = os.path.join(data_root, "train_pairs.txt")
#         assert os.path.exists(
#             pair_txt := os.path.join(self.args.data_dir, 'train_pairs.txt')), f"File {pair_txt} does not exist."
#         with open(pair_txt, 'r') as f:
#             lines = f.readlines()
#         #self.args.data_root_path = os.path.join(self.args.data_root_path, "train")
#         #output_dir = os.path.join(self.args.output_dir, "vitonhd", 'paired' if not self.args.eval_pair else 'paired')
#         data = []
#         for line in lines:
#             person_img, _ = line.strip().split(" ")
#             #im_name, _ = line.strip().split()
#             cloth_img = person_img
#             if self.args.eval_pair:
#                 cloth_img = person_img
#             data.append({
#                 'person_name': person_img,
#                 'person': os.path.join(data_root, 'image', person_img),
#                 'cloth': os.path.join(data_root, 'cloth', cloth_img),
#                 'mask': os.path.join(data_root, 'agnostic-mask',
#                                      person_img.replace('.jpg', '_mask.png')),
#             })
#         return data
#     def __getitem__(self, idx):
#         item = self.data[idx]
#
#         # 读取图像和掩码
#         person_pil = Image.open(item['person']).convert('RGB')
#         cloth_pil = Image.open(item['cloth']).convert('RGB')
#         mask_pil = Image.open(item['mask']).convert('L')
#         #cloth_mask_pil = Image.open(item['cloth_mask']).convert('L')
#
#         # 对掩码应用高斯模糊
#         # h = person_pil.height
#         # kernel_size = h // 50
#         # kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
#         # mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(kernel_size))
#         width=768
#         height = 1024
#         # drop
#         drop_image_embed = 0
#         rand_num = random.random()
#         if rand_num < self.i_drop_rate:
#             drop_image_embed = 1
#         # elif rand_num < (self.i_drop_rate + self.t_drop_rate):
#         #     text = ""
#         # elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
#         #     text = ""
#         #     drop_image_embed = 1
#
#         # 应用转换器
#         clip_image = self.clip_image_processor(images=cloth_pil, return_tensors="pt").pixel_values.squeeze(0)
#         person = self.vae_processor.preprocess(person_pil, height, width)[0]
#         cloth = self.vae_processor.preprocess(cloth_pil, height, width)[0]
#         mask = self.mask_processor.preprocess(mask_pil, height, width)[0]
#
#         # 结合掩码与人物图像
#         mask_bool = (mask < 0.5).float()
#         person_masked = person * mask_bool
#
#         # 构建数据字典
#         processed_data = {
#             'index': idx,
#             "person": person_masked,  # [3, H, W] 范围 [-1,1]
#             "cloth": cloth,  # [3, H, W]
#             "mask": mask,  # [1, H, W] 范围 [0,1]
#             "person_name": item['person_name']  # 保留文件名用于输出
#             , "clip_image": clip_image,
#             "drop_image_embeds": drop_image_embed
#         }
#         return processed_data

# class DressCodeTestDataset(InferenceDataset):
#     def load_data(self):
#         data = []
#         for sub_folder in ['upper_body', 'lower_body', 'dresses']:
#             assert os.path.exists(os.path.join(self.args.data_root_path, sub_folder)), f"Folder {sub_folder} does not exist."
#             pair_txt = os.path.join(self.args.data_root_path, sub_folder, 'test_pairs_paired.txt' if self.args.eval_pair else 'test_pairs_unpaired.txt')
#             assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
#             with open(pair_txt, 'r') as f:
#                 lines = f.readlines()
#
#             output_dir = os.path.join(self.args.output_dir, f"dresscode-{self.args.height}",
#                                       'unpaired' if not self.args.eval_pair else 'paired', sub_folder)
#             for line in lines:
#                 person_img, cloth_img = line.strip().split(" ")
#                 if os.path.exists(os.path.join(output_dir, person_img)):
#                     continue
#                 data.append({
#                     'person_name': os.path.join(sub_folder, person_img),
#                     'person': os.path.join(self.args.data_root_path, sub_folder, 'images', person_img),
#                     'cloth': os.path.join(self.args.data_root_path, sub_folder, 'images', cloth_img),
#                     'mask': os.path.join(self.args.data_root_path, sub_folder, 'agnostic_masks', person_img.replace('.jpg', '.png'))
#                 })
#         return data

class VITONHDTrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()  # 正确：无参数传递
        self.args = args
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                                do_convert_grayscale=True)
        self.data = self.load_data()  # 初始化时加载数据
        #self.i_drop_rate = args.cfg_dropout_rate
        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        # 使用原始 data_root_path，不修改 self.args
        data_root = os.path.join(self.args.data_dir, "test")  # 局部变量
        # pair_txt = os.path.join(data_root, "train_pairs.txt")
        assert os.path.exists(
            pair_txt := os.path.join(self.args.data_dir, 'test_pairs.txt')), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        # self.args.data_root_path = os.path.join(self.args.data_root_path, "train")
        # output_dir = os.path.join(self.args.output_dir, "vitonhd", 'paired' if not self.args.eval_pair else 'paired')
        data = []
        for line in lines:
            person_img, _ = line.strip().split(" ")
            # im_name, _ = line.strip().split()
            cloth_img = person_img
            if self.args.eval_pair:
                cloth_img = person_img
            data.append({
                'person_name': person_img,
                'person': os.path.join(data_root, 'image', person_img),
                'cloth': os.path.join(data_root, 'cloth', cloth_img),
                'mask': os.path.join(data_root, 'agnostic-mask',
                                     person_img.replace('.jpg', '_mask.png')),
            })
        return data

    def __getitem__(self, idx):
        item = self.data[idx]

        # 读取图像和掩码
        # person_pil = Image.open(item['person']).convert('RGB')
        # cloth_pil = Image.open(item['cloth']).convert('RGB')
        # mask_pil = Image.open(item['mask']).convert('L')
        person_pil = Image.open(item['person'])
        cloth_pil = Image.open(item['cloth'])
        mask_pil = Image.open(item['mask'])
        # cloth_mask_pil = Image.open(item['cloth_mask']).convert('L')
        width, height = self.args.width, self.args.height
        #print(width, height)
        # 对掩码应用高斯模糊
        # h = person_pil.height
        # kernel_size = h // 50
        # kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        # mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(kernel_size))

        # drop
        # drop_image_embed = 0
        # rand_num = random.random()
        # if rand_num < self.i_drop_rate:
        #     drop_image_embed = 1
        # elif rand_num < (self.i_drop_rate + self.t_drop_rate):
        #     text = ""
        # elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
        #     text = ""
        #     drop_image_embed = 1

        # 应用转换器
        clip_image = self.clip_image_processor(images=cloth_pil, return_tensors="pt").pixel_values.squeeze(0)
        person = self.vae_processor.preprocess(person_pil, height, width)[0]
        cloth = self.vae_processor.preprocess(cloth_pil, height, width)[0]
        mask = self.mask_processor.preprocess(mask_pil, height, width)[0]

        # 结合掩码与人物图像
        # mask_bool = (mask < 0.5).float()
        # person_masked = person * mask_bool

        # 构建数据字典
        processed_data = {
            "person": person,  # [3, H, W] 范围 [-1,1]
            "cloth": cloth,  # [3, H, W]
            "mask": mask,  # [1, H, W] 范围 [0,1]
            "person_name": item['person_name']  # 保留文件名用于输出
            ,"clip_image":clip_image,
           # "drop_image_embeds": drop_image_embed
        }
        return processed_data
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # parser.add_argument("--eval_pair", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default="/mnt/sda1/lzx/stabledata/zalando-hd-resized")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", type=bool, default=True)
    parser.add_argument("--random_flip", type=bool, default=True)
    parser.add_argument("--unet_ckpt", type=str,
                        default='/home/liangruyu/catvton/checkpoint-1000/model.safetensors')

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/mnt/sda2/lry/catvton_ckpt/stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    # parser.add_argument(
    #     "--resume_path",
    #     type=str,
    #     default="/mnt/sda2/lry/catvton_ckpt/CatVTON",
    #     help=(
    #         "The Path to the checkpoint of trained tryon model."
    #     ),
    # )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vitonhd",
        help="The datasets to use for evaluation.",
    )
    # parser.add_argument(
    #     "--data_root_path",
    #     type=str,
    #     default="/mnt/sda1/lzx/stabledata/zalando-hd-resized/test",
    #     help="Path to the dataset to evaluate."
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_batch128_1000_noconcat_origin",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--seed", type=int, default=555, help="A seed for reproducible evaluation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="The batch size for evaluation."
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to perform.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="The scale of classifier-free guidance for inference.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint",
        action="store_true",
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--eval_pair",
        default=True,
        action="store_true",
        help="Whether or not to evaluate the pair.",
    )
    parser.add_argument(
        "--concat_eval_results",
        action="store_true",
        help="Whether or not to  concatenate the all conditions into one image.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--concat_axis",
        type=str,
        choices=["x", "y", 'random'],
        default="y",
        help="The axis to concat the cloth feature, select from ['x', 'y', 'random'].",
    )
    parser.add_argument(
        "--enable_condition_noise",
        action="store_true",
        default=True,
        help="Whether or not to enable condition noise.",
    )

    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    return args


def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255#归一化mask
    # Step 3: 创建布尔掩码（假设mask=1表示生成区域）
   # mask_bool = (mask_np > 0.5).astype(np.float32)  # 二值化
    #mask_bool = (mask_np < 0.5).float()
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np#服装部分为1
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

@torch.no_grad()
def main():
    args = parse_args()
    # Pipeline
    pipeline = MyCatVTONPipeline(
        # attn_ckpt_version=args.dataset_name,
        # attn_ckpt=args.resume_path,
        base_ckpt=args.base_model_path,  # 只读unet的权重文件
        unet_ckpt=args.unet_ckpt,  # 新加了一个权重文件地址，指向我们训练好的权重
        weight_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device="cuda",
        skip_safety_check=True
    )
    # Dataset
    if args.dataset_name == "vitonhd":
        dataset = VITONHDTrainDataset(args)
    elif args.dataset_name == "dresscode":
        dataset = DressCodeTestDataset(args)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}.")
    print(f"Dataset {args.dataset_name} loaded, total {len(dataset)} pairs.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )
    # Inference
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    args.output_dir = os.path.join(args.output_dir, f"{args.dataset_name}-{args.height}", "paired" if args.eval_pair else "unpaired")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for batch in tqdm(dataloader):
        person_images = batch['person']
        cloth_images = batch['cloth']
        masks = batch['mask']
        '############新增clip处理后的条件特征代码'
        clip_image = batch['clip_image']
       # drop_image_embeds = batch['drop_image_embeds']
        '############新增clip处理后的条件特征代码'

        results = pipeline(
            person_images,
            cloth_images,
            masks,
            clip_image=clip_image,
            #drop_image_embeds=drop_image_embeds,

            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        )

        if args.concat_eval_results or args.repaint:
            person_images = to_pil_image(person_images)
            cloth_images = to_pil_image(cloth_images)
            masks = to_pil_image(masks)
        for i, result in enumerate(results):
            person_name = batch['person_name'][i]
            output_path = os.path.join(args.output_dir, person_name)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            if args.repaint:
                person_path, mask_path = dataset.data[batch['index'][i]]['person'], dataset.data[batch['index'][i]]['mask']
                person_image= Image.open(person_path).resize(result.size, Image.LANCZOS)
                mask = Image.open(mask_path).resize(result.size, Image.NEAREST)
                result = repaint(person_image, mask, result)
            if args.concat_eval_results:
                w, h = result.size
                concated_result = Image.new('RGB', (w*3, h))
                concated_result.paste(person_images[i], (0, 0))
                concated_result.paste(cloth_images[i], (w, 0))
                concated_result.paste(result, (w*2, 0))
                result = concated_result
            result.save(output_path)

if __name__ == "__main__":
    main()