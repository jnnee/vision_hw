import argparse
import logging
import os
import torch
import torch.nn.functional as F
import  time
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from accelerate import  Accelerator
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from safetensors import safe_open
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from diffusers.optimization import get_scheduler
import itertools
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor
import logging
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
from model.attn_processor import SkipAttnProcessor, AttnProcessor2_0
import os
from diffusers.training_utils import compute_dream_and_update_latents  # 确保该函数存在
import random
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用GPU 0
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from model.pipeline import CatVTONPipeline
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from diffusers.models.attention import Attention  # 正确导入路径
from diffusers.image_processor import VaeImageProcessor
import torchvision.transforms.functional as TF
from typing import Literal, Tuple,List
# 读取数据集
# class TrainDataset(Dataset):
#     def __init__(self, args):
#         self.args = args
#
#         self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
#         self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
#                                                 do_convert_grayscale=True)
#         self.data = self.load_data()
#
#     def load_data(self):
#         return []
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         data = self.data[idx]
#         person, cloth, mask,cloth_mask = [Image.open(data[key]) for key in ['person', 'cloth', 'mask','cloth_mask']]
#         _, h = person.size
#         kernal_size = h // 50
#         if kernal_size % 2 == 0:
#             kernal_size += 1
#         mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
#         person_np = np.array(person)
#         #result_np = np.array(result)
#         mask_np = np.array(mask) / 255  # 归一化mask
#         person=person_np * (mask_np < 0.5)
#         train_transforms = transforms.Compose(
#             [
#                 transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#                 transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
#                 transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#         )
#
#         # val_transforms = transforms.Compose(
#         #     [
#         #         transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#         #         transforms.ToTensor(),
#         #     ]
#         # )
#         person=train_transforms(person)
#         cloth, mask,cloth_mask = [train_transforms(data[key]) for key in [ 'cloth', 'mask','cloth_mask']]
#         return {
#             'index': idx,
#             'person_name': data['person_name'],
#             'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
#             'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
#             'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0],
#             'cloth_mask': self.mask_processor.preprocess(cloth_mask, self.args.height, self.args.width)[0],
#         }

class VITONHDTrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()  # 正确：无参数传递
        self.args = args
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                                do_convert_grayscale=True)
        self.data = self.load_data()  # 初始化时加载数据
        self.i_drop_rate =args.cfg_dropout_rate
    def __len__(self):
        return len(self.data)
    def load_data(self):
        # 使用原始 data_root_path，不修改 self.args
        data_root = os.path.join(self.args.data_dir, "train")  # 局部变量
        #pair_txt = os.path.join(data_root, "train_pairs.txt")
        assert os.path.exists(
            pair_txt := os.path.join(self.args.data_dir, 'train_pairs.txt')), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        #self.args.data_root_path = os.path.join(self.args.data_root_path, "train")
        #output_dir = os.path.join(self.args.output_dir, "vitonhd", 'paired' if not self.args.eval_pair else 'paired')
        data = []
        for line in lines:
            person_img, _ = line.strip().split(" ")
            #im_name, _ = line.strip().split()
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
        person_pil = Image.open(item['person']).convert('RGB')
        cloth_pil = Image.open(item['cloth']).convert('RGB')
        mask_pil = Image.open(item['mask']).convert('L')
        #cloth_mask_pil = Image.open(item['cloth_mask']).convert('L')
        width, height = args.img_size
        # 对掩码应用高斯模糊
        #h = person_pil.height
        # kernel_size = h // 50
        # kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        # mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(kernel_size))

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        # elif rand_num < (self.i_drop_rate + self.t_drop_rate):
        #     text = ""
        # elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
        #     text = ""
        #     drop_image_embed = 1

        # 应用转换器
        #clip_image = self.clip_image_processor(images=cloth_pil, return_tensors="pt").pixel_values.squeeze(0)
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
            , "clip_image": clip_image,
            "drop_image_embeds": drop_image_embed
        }
        return processed_data

def init_adapter(unet,
                 cross_attn_cls=SkipAttnProcessor,
                 self_attn_cls=None,
                 cross_attn_dim=None,
                 **kwargs):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
        else:
            attn_procs[name] = cross_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    return adapter_modules

def train(args):
    accelerator = Accelerator(
        # mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        #project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    # 初始化模型
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    #image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    # )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",use_safetensors=False,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",use_safetensors=False,
    )

    # For inpainting an additional image is used for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom inpainting dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.

    if unet.conv_in.in_channels == 4:  # 将预训练的4通道UNet模型转换为适用于图像修复（inpainting）任务的9通道UNet模型
        logger.info("Initializing the Inpainting UNet from the pretrained 4 channel UNet .")
        in_channels = 9
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # 初始化适配器（跳过交叉注意力）
    init_adapter(unet,  skip_cross_attn=True)


    for name, param in unet.named_parameters():
        if 'attn1' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
            param.requires_grad = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # 通过 accelerator.prepare 封装训练组
    # 手动设置非训练组件的设备和数据类型,即固定权重的组件
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    #text_encoder.to(accelerator.device, dtype=weight_dtype)
    #image_encoder.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = itertools.chain(unet.parameters())

    # trainable_params = [p for p in params_to_opt if p.requires_grad]
    # print(f"可训练参数数量: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer.load_state_dict(torch.load("/mnt/sda2/lry/catViton_modelckpt/checkpoint-26000/optimizer.bin",weights_only=True,
    #     map_location="cuda"  # 按需指定设备
    # ))
    # print("优化器学习率:", optimizer.param_groups[0]["lr"])  # 应等于保存时的学习率

    # dataloader
    # 数据集
    train_dataset = VITONHDTrainDataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # 学习率调度
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, train_dataloader,lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader,lr_scheduler)
    # 检查点恢复逻辑
    # 检查是否存在检查点并加载
    checkpoint_dirs = []
    if os.path.exists(args.output_dir):
        checkpoint_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
    global_step = 0
    if checkpoint_dirs:
        # 获取最新的检查点
        checkpoint_steps = [int(d.split('-')[1]) for d in checkpoint_dirs]
        latest_step = max(checkpoint_steps)
        latest_checkpoint = os.path.join(args.output_dir, f"checkpoint-{latest_step}")
        accelerator.print(f"从检查点恢复训练: {latest_checkpoint}")
        accelerator.load_state(latest_checkpoint)

        # 恢复全局步数
        global_step = latest_step

    # 训练循环
    for epoch in range(args.num_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            # 数据准备
            #person = batch['person'].to(args.device)
            person=prepare_image(batch['person']).to(args.device, dtype=weight_dtype)
            cloth =prepare_image(batch['cloth']).to(args.device, dtype=weight_dtype)
            mask = prepare_mask_image(batch['mask']).to(args.device, dtype=weight_dtype)
            # Mask image
            masked_image = person * (mask < 0.5)
            #cloth_mask = batch['cloth_mask'].to(args.device)
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    # VAE encoding
                    masked_latent = compute_vae_encodings(masked_image, vae)
                    cloth_latents = compute_vae_encodings(cloth, vae)
                    mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
                    del person, mask
                    # latents = vae.encode(
                    #     batch["person"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    # latents = latents * vae.config.scaling_factor
                    # # 布料图像编码
                    # cloth_latents = vae.encode(
                    #     batch["cloth"].to(accelerator.device, dtype=weight_dtype)
                    # ).latent_dist.sample()
                    # cloth_latents = cloth_latents * vae.config.scaling_factor  # [B, C, H_cloth, W_cloth]
                    #print(vae.config.scaling_factor)
                    cloth_latents_embeds_ = []
                    for cloth_latents_embed, drop_image_embed in zip(cloth_latents, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            cloth_latents_embeds_.append(torch.zeros_like(cloth_latents_embed))
                        else:
                            cloth_latents_embeds_.append(cloth_latents_embed)
                    cloth_latents = torch.stack(cloth_latents_embeds_)
                    image_latents = torch.cat([masked_latent, cloth_latents], dim=3)  # 沿宽度拼接
                    # Concatenate latents
                    #masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
                    # mask = torch.nn.functional.interpolate(
                    #     mask, size=(128, 128), mode='bilinear', align_corners=False
                    # )
                    mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=3)


                # Sample noise that we'll add to the latents
                generator = torch.Generator(device='cuda').manual_seed(555)
                # Prepare noise
                noise = randn_tensor(
                    image_latents.shape,
                    generator=generator,
                    device=image_latents.device,
                    dtype=weight_dtype,
                )



                # Prepare timesteps
                #noise_scheduler.set_timesteps(args.num_inference_steps, device=args.device)
                #timesteps = noise_scheduler.timesteps
                #latents = latents * noise_scheduler.init_noise_sigma
                #noise = torch.randn_like(image_latents)
                bsz = noise.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=noise.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(image_latents, noise, timesteps)
                # print("noisy_latents.shape:", noisy_latents.shape)
                # print("mask_latent_concat.shape:", mask_latent_concat.shape)
                # print("image_latents.shape:", image_latents.shape)


                combined_latents = torch.cat([noisy_latents, mask_latent_concat, image_latents], dim=1)

                noise_pred = unet(combined_latents, timesteps, encoder_hidden_states=None,return_dict=False,)[0]
                #print(combined_latents.shape,timesteps.shape, image_embeds.shape)
                # 假设 image_latents 的前半部分（沿宽度）为图像，后半部分为布料
                image_width = image_latents.shape[3]  # 原始图像潜在表示的宽度
                # 这是切割后的噪声
                noise_image = noise[:, :, :, :image_width]  # [B, C, H, W_image]
                # 切片噪声预测和真实噪声，仅保留图像部分
                noise_pred_image = noise_pred[:, :, :, :image_width]  # [B, 4, H, W_image]

                loss = F.mse_loss(noise_pred_image.float(), noise_image.float(), reduction="mean")

                # print(loss)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                #loss = torch.nan_to_num(loss)

                # Backpropagate
                accelerator.backward(loss)

                # for name, param in unet.named_parameters():
                #     if param.requires_grad and param.grad is None:
                #         print(f"参数 {name} 无梯度！")

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, global_step, load_data_time, time.perf_counter() - begin, avg_loss))

            global_step += 1

            if global_step % args.save_steps == 0:
                print("Saving model...")
                logging.info(
                    f"Epoch {epoch}, Step {global_step}: Loss {loss.item()}, LR {lr_scheduler.get_last_lr()[0]}"
                )
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                # 如果 accelerator.save_state 不支持直接保存调度器，手动补充保存（备用方案）
                # scheduler_path = os.path.join(save_path, "scheduler.bin")
                # torch.save(lr_scheduler.state_dict(), scheduler_path)
            begin = time.perf_counter()

            lr_scheduler.step()
            #global_step += 1

# 配置参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,default="/mnt/sda1/lzx/stabledata/zalando-hd-resized")

    '#######dream训练方式'
    parser.add_argument("--dream_training", action="store_true",
                        help="Use DREAM training method")
    parser.add_argument("--dream_detail_preservation", type=float, default=10.0,
                        help="Dream detail preservation factor")
    '#######dream训练方式'


    parser.add_argument("--eval_pair", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", type=bool, default=True)
    parser.add_argument("--random_flip", type=bool, default=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/mnt/sda2/lry/catvton_ckpt/stable-diffusion-inpainting")
    parser.add_argument("--image_encoder_path", type=str,
                        default="/mnt/sda2/lry/catvton_ckpt/IP-Adapter/models/image_encoder")
    parser.add_argument("--ip_adapter_weight", type=str, default="/mnt/sda2/lry/catvton_ckpt/IP-Adapter/models/ip-adapter_sd15.bin")
    #parser.add_argument("--attn_model", type=str, required=True)
    parser.add_argument("--report_to", type=str, default=None,
                        help="Logging integration (e.g., 'tensorboard', 'wandb')")

    parser.add_argument("--ip_adapter_path", type=str,default="/mnt/sda2/lry/catvton_ckpt/IP-Adapter/models/ip-adapter_sd15.bin")
    parser.add_argument("--output_dir", type=str, default="/mnt/sda2/lry/catViton_modelckpt_noip")
    parser.add_argument("--img_size", type=tuple, default=(512, 384))
    parser.add_argument("--train_batch_size", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--mixed_precision", action="store_true",default='fp16')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained_ip_adapter_path", type=str,
                        default="/mnt/sda2/lry/catvton_ckpt/IP-Adapter/models/ip-adapter_sd15.bin")
    parser.add_argument("--cfg_dropout_rate", type=float, default=0.1,
                        help="Probability of dropping condition during CFG training")
    return parser.parse_args()


# 运行入口
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
            logging.StreamHandler()
        ]
    )

    train(args)