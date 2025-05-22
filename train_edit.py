import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from diffusers.utils.torch_utils import randn_tensor
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from model.attn_processor import SkipAttnProcessor
from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用GPU 0
# Dataset
# class MyDataset(torch.utils.data.Dataset):
#
#     def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
#         super().__init__()
#
#         self.tokenizer = tokenizer
#         self.size = size
#         self.i_drop_rate = i_drop_rate
#         self.t_drop_rate = t_drop_rate
#         self.ti_drop_rate = ti_drop_rate
#         self.image_root_path = image_root_path
#
#         self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]
#
#         self.transform = transforms.Compose([
#             transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(self.size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ])
#         self.clip_image_processor = CLIPImageProcessor()
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         text = item["text"]
#         image_file = item["image_file"]
#
#         # read image
#         raw_image = Image.open(os.path.join(self.image_root_path, image_file))
#         image = self.transform(raw_image.convert("RGB"))
#         clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
#
#         # drop
#         drop_image_embed = 0
#         rand_num = random.random()
#         if rand_num < self.i_drop_rate:
#             drop_image_embed = 1
#         elif rand_num < (self.i_drop_rate + self.t_drop_rate):
#             text = ""
#         elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
#             text = ""
#             drop_image_embed = 1
#         # get text and tokenize
#         text_input_ids = self.tokenizer(
#             text,
#             max_length=self.tokenizer.model_max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         ).input_ids
#
#         return {
#             "image": image,
#             "text_input_ids": text_input_ids,
#             "clip_image": clip_image,
#             "drop_image_embed": drop_image_embed
#         }
#
#     def __len__(self):
#         return len(self.data)
class VITONHDTrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()  # 正确：无参数传递
        self.args = args
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                                do_convert_grayscale=True)
        self.data = self.load_data()  # 初始化时加载数据
        self.i_drop_rate = args.cfg_dropout_rate
        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        # 使用原始 data_root_path，不修改 self.args
        data_root = os.path.join(self.args.data_dir, "train")  # 局部变量
        # pair_txt = os.path.join(data_root, "train_pairs.txt")
        assert os.path.exists(
            pair_txt := os.path.join(self.args.data_dir, 'train_pairs.txt')), f"File {pair_txt} does not exist."
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
        width, height = args.img_size
        # 对掩码应用高斯模糊
        # h = person_pil.height
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
            ,"clip_images":clip_image,
            "drop_image_embeds": drop_image_embed
        }
        return processed_data

def get_module_by_path(model, path):
    """
    根据点分路径访问嵌套子模块，例如：
    path = "down_blocks.0.attentions.0.transformer_blocks.0.attn2"
    """
    modules = path.split(".")
    current_module = model
    for part in modules:
        # 处理数字索引（如列表中的第0项）
        if part.isdigit():
            current_module = current_module[int(part)]
        # 处理属性名（如 "down_blocks"）
        else:
            current_module = getattr(current_module, part)
    return current_module
def compute_hf_loss(self, x0_pred, x0_real):
    """计算高频特征MSE损失"""
    # 傅里叶变换
    fft_pred = torch.fft.fft2(x0_pred)
    fft_real = torch.fft.fft2(x0_real)

    # 应用高通滤波并计算损失
    hf_pred = fft_pred * self.hf_mask
    hf_real = fft_real * self.hf_mask
    return torch.mean(torch.abs(hf_pred - hf_real) ** 2)


def q_sample(self, x0, t, noise):
    """前向扩散过程（需根据具体扩散参数实现）"""
    sqrt_alpha = self.get_sqrt_alpha(t)  # 获取扩散系数
    return sqrt_alpha * x0 + (1 - sqrt_alpha) * noise


def predict_x0_from_noise(self, x_t, t, pred_noise):
    """从噪声预测重建x0（需根据具体参数化方式实现）"""
    sqrt_alpha = self.get_sqrt_alpha(t)
    return (x_t - (1 - sqrt_alpha) * pred_noise) / sqrt_alpha


def get_sqrt_alpha(self, t):
    """示例：获取当前时间步的alpha系数（需替换实际调度逻辑）"""
    return 1.0 - t.float() / self.num_timesteps


# def collate_fn(data):
#     images = torch.stack([example["image"] for example in data])
#     text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
#     clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
#     drop_image_embeds = [example["drop_image_embed"] for example in data]
#
#     return {
#         "images": images,
#         "text_input_ids": text_input_ids,
#         "clip_images": clip_images,
#         "drop_image_embeds": drop_image_embeds
#     }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        # encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # 把ip_token数据直接作为encoder——hidden-states
        encoder_hidden_states = ip_tokens
        # 在IPAdapter的forward方法中添加：
        #print("ip_tokens.shape:", ip_tokens.shape)  # 预期 (batch_size, sequence_length, hidden_size)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        #print("noise_pred.shape_ip:", noise_pred.shape)(1,4,128,48)
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="/mnt/sda2/lry/catvton_ckpt/stable-diffusion-inpainting")
    parser.add_argument("--device", type=str, default="cuda")
    '#######dream训练方式'
    parser.add_argument("--dream_training", action="store_true",
                        help="Use DREAM training method")
    parser.add_argument("--dream_detail_preservation", type=float, default=10.0,
                        help="Dream detail preservation factor")
    '#######dream训练方式'
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default="/mnt/sda2/lry/catvton_ckpt/IP-Adapter/models/ip-adapter_sd15.bin",
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        #required=True,
        help="Training data",
    )
    parser.add_argument("--data_dir", type=str, default="/mnt/sda1/lzx/stabledata/zalando-hd-resized")
    # parser.add_argument(
    #     "--data_root_path",
    #     type=str,
    #     default="/mnt/sda1/lzx/stabledata/zalando-hd-resized",
    #     #required=True,
    #     help="Training data root path",
    # )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="/mnt/sda2/lry/catvton_ckpt/IP-Adapter/models/image_encoder",
       # required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument("--cfg_dropout_rate", type=float, default=0.1,
                        help="Probability of dropping condition during CFG training")
    parser.add_argument("--eval_pair", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/sda2/lry/catViton_modelckpt_noip_384-512_dim-2_dream_######_edit",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate to use.",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=7275)
    parser.add_argument("--max_train_steps", type=int, default=145500)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--img_size", type=tuple, default=(384, 512))
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def train(args):
    # args = parse_args()
    # logging_dir = Path(args.output_dir, args.logging_dir)
    #
    # accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    weight_dtype = torch.float32
    # init adapter modules设置注意力
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        # 遍历每一层的name
        #print(name)
        # 用来判断是自注意力还是交叉注意力，
        # cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.endswith("attn1.processor"):
            cross_attention_dim = None
        else:
            cross_attention_dim = unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            # 自注意力跳过
            attn_procs[name] = SkipAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                )  # 跳过自注意力
            #print("跳过自注意力", name)
            # attn_procs[name] = AttnProcessor()
        else:
            # 交叉注意力
            layer_name = name.split(".processor")[0]
            layer = get_module_by_path(unet, layer_name)
            # if hasattr(layer, "to_k"):
            #     del layer.to_k
            # else:
            #     print("没删")
            del layer.to_k  # 删除 to_k 线性层
            del layer.to_v  # 删除 to_v 线性层
            #print(layer_name)layer_name指向自注意力层
            layer_name =layer_name.replace("attn2", "attn1")

            #print(layer_name)
            # 获取自注意力的权重
            weights_1 = {
                # "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                # "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                "to_q.weight": unet_sd[layer_name + ".to_q.weight"],
                #"to_k.weight": unet_sd[layer_name + ".to_k.weight"],
                #"to_v.weight": unet_sd[layer_name + ".to_v.weight"],
                "to_out.0.weight": unet_sd[layer_name + ".to_out.0.weight"],
                "to_out.0.bias": unet_sd[layer_name + ".to_out.0.bias"]
            }
            weights_2 = {
                "to_k.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v.weight": unet_sd[layer_name + ".to_v.weight"],
                "to_kv_ip.weight":torch.zeros((hidden_size, 768), dtype=weight_dtype, device=args.device),
            }
            # layer_name = name.split(".processor")[0]
            # weights = {
            #     "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
            #     "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            # }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            #print(attn_procs[name])
            layer_name.replace("attn1", "attn2")
            layer = get_module_by_path(unet, layer_name)
            #print(layer)
            layer.load_state_dict(weights_1, strict=False)
            # unet[layer_name].load_state_dict(weights)
            attn_procs[name].load_state_dict(weights_2)
            # print('交叉注意力的名字' + attn_procs[name])

        # if name.endswith("attn2.processor") :

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    # 示例：在初始化 IPAdapter 前验证参数
    print("adapter_modules 参数数量:", len(list(adapter_modules.parameters())))
    for name, param in unet.named_parameters():
       #if 'attn2' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
       if 'attn2' in name and 'to_q' in name :
            param.requires_grad = True
    # ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, None)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters(),
                                    ip_adapter.unet.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    # train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )
    train_dataset = VITONHDTrainDataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    # 学习率调度
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(ip_adapter, optimizer, train_dataloader,
                                                                                lr_scheduler)

    # def print_model_structure(model, prefix=""):
    #     for name, module in model.named_children():
    #         print(f"{prefix}{name} ({module.__class__.__name__})")
    #         if len(list(module.children())) > 0:
    #             print_model_structure(module, prefix + "  ")
    #         else:
    #             # 打印叶子层的参数形状（可选）
    #             for param_name, param in module.named_parameters():
    #                 print(f"{prefix}  {param_name}: {param.shape}")
    #
    # # 示例：打印IPAdapter结构
    # print_model_structure(ip_adapter)
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

        # 验证关键状态
        current_lr = optimizer.param_groups[0]['lr']
        accelerator.print(f"✅ 恢复状态验证:")
        accelerator.print(f"  当前学习率 = {current_lr:.2e}")

        # 验证数据加载器位置（需在保存检查点时保存采样器状态）
        try:
            accelerator.print(f"  数据加载器进度 = {train_dataloader.sampler.index}/{len(train_dataloader)}")
        except AttributeError:
            pass

    concat_dim = 2
    # 训练循环
    # global_step = 0
    for epoch in range(args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            # 数据准备
            # person = batch['person'].to(args.device)
            person = prepare_image(batch['person']).to(args.device, dtype=weight_dtype)
            cloth = prepare_image(batch['cloth']).to(args.device, dtype=weight_dtype)
            mask = prepare_mask_image(batch['mask']).to(args.device, dtype=weight_dtype)
            # Mask image
            masked_image = person * (mask < 0.5)
            # with accelerator.accumulate(ip_adapter):
            #     # Convert images to latent space
            #     with torch.no_grad():
            #         latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
            #         latents = latents * vae.config.scaling_factor
            #
            #     # Sample noise that we'll add to the latents
            #     noise = torch.randn_like(latents)
            #     bsz = latents.shape[0]
            #     # Sample a random timestep for each image
            #     timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            #     timesteps = timesteps.long()
            #
            #     # Add noise to the latents according to the noise magnitude at each timestep
            #     # (this is the forward diffusion process)
            #     noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            #
            #     with torch.no_grad():

            #
            #     with torch.no_grad():
            #         encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
            #
            #     noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)
            #
            #     loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            #
            #     # Gather the losses across all processes for logging (if we use distributed training).
            #     avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
            #
            #     # Backpropagate
            #     accelerator.backward(loss)
            #     optimizer.step()
            #     optimizer.zero_grad()
            #
            #     if accelerator.is_main_process:
            #         print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
            #             epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            #
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    # VAE encoding
                    person_latent = compute_vae_encodings(person, vae)  # 模型需要学习的信息
                    masked_latent = compute_vae_encodings(masked_image, vae)
                    cloth_latents = compute_vae_encodings(cloth, vae)
                    mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")

                    # del person, mask
                    # 清晰图像拼接条件信息（衣服）
                    person_latent = torch.cat([person_latent, cloth_latents], dim=concat_dim)  # 沿宽度拼接

                    '''无分类引导'''
                    cloth_latents_embeds_ = []
                    for cloth_latents_embed, drop_image_embed in zip(cloth_latents, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            cloth_latents_embeds_.append(torch.zeros_like(cloth_latents_embed))
                        else:
                            cloth_latents_embeds_.append(cloth_latents_embed)
                    cloth_latents = torch.stack(cloth_latents_embeds_)
                    #生成image——embeds，使用无分类引导
                    image_embeds = image_encoder(
                        batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                    image_embeds_ = []
                    for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            image_embeds_.append(torch.zeros_like(image_embed))
                        else:
                            image_embeds_.append(image_embed)
                    image_embeds = torch.stack(image_embeds_)
                    # image_latents = torch.cat([masked_latent, cloth_latents], dim=3)  # 沿宽度拼接
                    image_latents = torch.cat([masked_latent, cloth_latents], dim=concat_dim)  # 沿宽度拼接
                    '''#########'''

                    mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

                # Sample noise that we'll add to the latents
                generator = torch.Generator(device='cuda').manual_seed(555)
                # Prepare noise
                noise = randn_tensor(
                    image_latents.shape,
                    generator=generator,
                    device=image_latents.device,
                    dtype=weight_dtype,
                )
                noise = noise * noise_scheduler.init_noise_sigma

                # Prepare timesteps
                # noise_scheduler.set_timesteps(args.num_inference_steps, device=args.device)
                # timesteps = noise_scheduler.timesteps
                # latents = latents * noise_scheduler.init_noise_sigma
                # noise = torch.randn_like(image_latents)
                bsz = noise.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=noise.device)
                timesteps = timesteps.long()
                # 强制10%样本使用t=0（根据实验调整比例）
                force_t1_mask = torch.rand(timesteps.shape, device=timesteps.device) < 0.1
                timesteps[force_t1_mask] = 0  # 确保高频损失有足够样本
                # noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device='cuda')
                # timesteps = noise_scheduler.timesteps
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # 这里清晰图像上加噪
                noisy_latents = noise_scheduler.add_noise(person_latent, noise, timesteps)
                # print("noisy_latents.shape:", noisy_latents.shape)
                # print("mask_latent_concat.shape:", mask_latent_concat.shape)
                # print("image_latents.shape:", image_latents.shape)

                # 假设 image_latents 的前半部分（沿宽度）为图像，后半部分为布料
                # image_width = masked_latent.shape[concat_dim]  # 原始图像潜在表示的宽度
                # 噪声图像（加在人物上的）
                # noise_image = noise[:, :, :image_width, :]  # [B, C, H, W_image]

                '############添加dream训练代码'
                # 原代码中的目标定义需要调整
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise  # 这个是对的
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(image_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # 使用dream得到处理后的噪声图像，在拼接上条件
                combined_latents = torch.cat([noisy_latents, mask_latent_concat, image_latents], dim=1)

                '''重新定义dream的计算方式'''

                def compute_dream_and_update_latents(
                        unet,
                        noise_scheduler,
                        timesteps,
                        noise,  # 传进来的是做loss的噪声
                        noisy_latents,  # 输入的噪声（9通道）
                        target,
                        encoder_hidden_states,
                        # image_width,
                        noisy_latents_orin,  # 补充一个参数，加了噪声的latant图像
                        mask_latent_concat,  # 补充一个参数
                        image_latents,  # 补充一个参数
                        detail_preservation=10.0
                ):
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
                    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
                    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
                    dream_lambda = sqrt_one_minus_alphas_cumprod ** detail_preservation

                    pred = None
                    with torch.no_grad():
                        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        # pred = pred[:, :, :image_width, :]  # 对应上我们的预测的图像
                        # print(pred.shape)

                    _noisy_latents, _target = (None, None)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        predicted_noise = pred
                        delta_noise = (noise - predicted_noise).detach()
                        delta_noise.mul_(dream_lambda)  # 这个形状是对的
                        '''核心修改 noisy_latents -> noisy_latents_orin'''
                        _noisy_latents = noisy_latents_orin.add(sqrt_one_minus_alphas_cumprod * delta_noise)  # 这里报错
                        _target = target.add(delta_noise)
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        raise NotImplementedError("DREAM has not been implemented for v-prediction")
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    '''再把条件给拼接回来'''
                    combined_latents = torch.cat([_noisy_latents, mask_latent_concat, image_latents], dim=1)
                    return combined_latents, _target

                # 添加DREAM处理逻辑
                if args.dream_training:
                    combined_latents, target = compute_dream_and_update_latents(
                        unet,
                        noise_scheduler,
                        timesteps,
                        noise,
                        combined_latents,  # 放入完整的潜在变量数据
                        target,
                        None,  # 原版是encoder_hidden_states（文本条件），这里可能需要使用图像条件
                        # image_width=image_width,  # 添加的变量
                        noisy_latents_orin=noisy_latents,
                        mask_latent_concat=mask_latent_concat,
                        image_latents=image_latents,
                        # args.dream_detail_preservation
                    )
                '############'
                #noise和noise_pred维度对不上
                #noise_pred = ip_adapter(combined_latents, timesteps, image_embeds=image_embeds )[0]
                noise_pred = ip_adapter(combined_latents, timesteps, image_embeds=image_embeds)
                #print("noise.shape:", noise.shape)
                #print("noise_pred.shape:", noise_pred.shape)
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                #print(loss.shape)
                # 当t=0时添加高频损失
                # 仅当时间步包含1时执行
                t_mask = (timesteps == 0)

                if t_mask.any():
                    # 提取t=1样本
                    noisy_latents_t1 = noisy_latents[t_mask]
                    noise_pred_t1 = noise_pred[t_mask]

                    # 计算预测x0
                    alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps[t_mask]]
                    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                    #pred_x0 = (noisy_latents_t1 - (1 - alpha_prod_t) ** 0.5 * noise_pred_t1) / alpha_prod_t ** 0.5
                    # 在计算 pred_x0 后，提取实部
                    pred_x0 = (noisy_latents_t1 - (1 - alpha_prod_t) ** 0.5 * noise_pred_t1) / alpha_prod_t ** 0.5
                    #pred_x0 = pred_x0.real  # 取实部，确保为实数张量
                    # 解码
                    pred_x0 = pred_x0.split(pred_x0.shape[concat_dim] // 2, dim=concat_dim)[0]
                    pred_x0 = 1 / vae.config.scaling_factor * pred_x0
                    decoded_images = vae.decode(pred_x0.to(weight_dtype)).sample
                    decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
                    decoded_images_masked = decoded_images * (mask[t_mask] > 0.5)

                    # 获取真实图像
                    real_images_t1 = batch["person"][t_mask].to(args.device)  # [-1,1]
                    real_images_0to1 = ((real_images_t1 + 1) / 2).clamp(0.0, 1.0)
                    real_images_t1_masked = real_images_0to1 * (mask[t_mask] > 0.5)

                    # 计算高频损失
                    # pixel_x0_pred = torch.fft.fft2(torch.mean(decoded_images_masked, dim=1))
                    # pixel_values = torch.fft.fft2(torch.mean(real_images_t1_masked, dim=1))
                    # 计算FFT（保持通道维度）
                    pixel_x0_pred = torch.fft.fft2(decoded_images_masked)  # [B,C,H,W]
                    pixel_values = torch.fft.fft2(real_images_t1_masked)  # [B,C,H,W]

                    # 计算幅度谱
                    pixel_x0_mag = torch.abs(pixel_x0_pred)
                    pixel_values_mag = torch.abs(pixel_values)

                    # 计算高频损失（标量）
                    hf_loss = ((pixel_x0_mag - pixel_values_mag) ** 2).mean()
                    # print("pixel_x0_pred shape:", pixel_x0_pred.shape)
                    # print("pixel_values shape:", pixel_values.shape)
                    #hf_loss = (pixel_x0_pred.real.float() - pixel_values.real.float()) ** 2
                    #print(pixel_x0_pred.dtype, pixel_values.dtype)

                    # 加权总损失
                    loss += 0.1 * hf_loss
                # for i in timesteps:
                #     if i==1:
                #         # mask torch.Size([1, 1, 128, 96])
                #         # latents = 1 / self.vae.config.scaling_factor * latents
                #         # image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
                #         # image = (image / 2 + 0.5).clamp(0, 1)
                #         # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                #         # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                #
                #         # 预测的x0的潜空间图像
                #         latents = (noisy_latents - sigmas * noise_pred)  # / (1.0 - sigmas)
                #         # 分割出预测的图像信息，取前一半
                #         latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
                #         # vae固定操作计算
                #         latents = 1 / vae.config.scaling_factor * latents
                #         image = vae.decode(latents).sample
                #
                #         image = (image / 2 + 0.5).clamp(0, 1)
                #
                #
                #
                #         x0_pred = (x0_pred / vae.config.scaling_factor) + vae.config.shift_factor
                #
                #         pixel_x0_pred = vae.decode(x0_pred, return_dict=False)[0]
                #         mask = F.interpolate(batch["mask_input"], (pixel_values.shape[2], pixel_values.shape[3]))[0]
                #         pixel_x0_pred = pixel_x0_pred.float() * mask
                #         pixel_values = pixel_values.float() * mask
                #         pixel_x0_pred = torch.fft.fft2(torch.mean(pixel_x0_pred, dim=1))
                #         pixel_values = torch.fft.fft2(torch.mean(pixel_values, dim=1))
                #         # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                #         # Preconditioning of the model outputs.
                #         # if args.precondition_outputs:
                #         #     model_pred = model_pred * (-sigmas) + noisy_model_input
                #         # these weighting schemes use a uniform timestep sampling
                #         # and instead post-weight the loss
                #         weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                #
                #         loss = torch.mean(
                #             (weighting.float() * (pixel_x0_pred.float() - pixel_values.float()) ** 2).reshape(bsz, -1),
                #             1,
                #         )
                # if (t == 1).any():
                #     # 获取当前预测的清晰图像
                #     x0_pred = self.predict_x0_from_noise(x_t, t, pred_noise)
                #
                #     # 计算高频损失（仅对t=1的样本）
                #     mask = (t == 1).float().view(-1, 1, 1, 1)
                #     hf_loss = self.compute_hf_loss(x0_pred, x0)
                #     loss += mask.mean() * hf_loss  # 加权求和
                # # 切片噪声预测和真实噪声，仅保留图像部分
                # noise_pred_image = noise_pred[:, :, :image_width, :]  # [B, 4, W_image, H]

                # # 高频损失计算（仅对t=1的样本）
                # t1_mask = (timesteps == 1)
                # if t1_mask.any():
                #     # 获取预测的清晰图像（需实现x0预测逻辑）
                #     with torch.no_grad():
                #         # 根据扩散调度器参数计算x0预测
                #         alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                #         x0_pred = (noisy_latents - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                #
                #         # 从潜在空间解码到像素空间
                #         x0_pred_pixel = vae.decode(x0_pred / vae.config.scaling_factor, return_dict=False)[0]
                #         x0_real_pixel = vae.decode(person_latent / vae.config.scaling_factor, return_dict=False)[0]
                #
                #     # 仅处理t=1的样本
                #     x0_pred_t1 = x0_pred_pixel[t1_mask]
                #     x0_real_t1 = x0_real_pixel[t1_mask]
                #
                #     # 计算高频损失
                #     hf_loss = compute_hf_loss(x0_pred_t1, x0_real_t1)  # 需要预先实现该函数
                #
                #     # 动态加权（根据t=1样本比例）
                #     loss += hf_loss * (x0_pred_t1.shape[0] / x0_pred_pixel.shape[0])
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # loss = torch.nan_to_num(loss)

                # Backpropagate
                accelerator.backward(loss)

                # for name, param in unet.named_parameters():
                #     if param.requires_grad and param.grad is None:
                #         print(f"参数 {name} 无梯度！")

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, global_step, load_data_time, time.perf_counter() - begin, avg_loss))

            global_step += 1
            if accelerator.is_main_process:
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
                if  global_step == 100 or global_step == 1000 or global_step == 3000 or global_step == 5000 or global_step == 8000:
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
            # if global_step % args.save_steps == 0:
            #     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            #     accelerator.save_state(save_path)

            # begin = time.perf_counter()


if __name__ == "__main__":
    # main()
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
