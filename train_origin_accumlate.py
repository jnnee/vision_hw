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
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用GPU 0
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
#加入ema模型
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化影子权重
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        # 更新影子权重
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # 应用影子权重到模型
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data.copy_(self.shadow[name])

    def restore(self):
        # 恢复原始权重
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
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
        # person_pil = Image.open(item['person']).convert('RGB')
        # cloth_pil = Image.open(item['cloth']).convert('RGB')
        # mask_pil = Image.open(item['mask']).convert('L')
        person_pil = Image.open(item['person'])
        cloth_pil = Image.open(item['cloth'])
        mask_pil = Image.open(item['mask'])
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
            ,
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
        gradient_accumulation_steps=4
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

    # if unet.conv_in.in_channels == 4:  # 将预训练的4通道UNet模型转换为适用于图像修复（inpainting）任务的9通道UNet模型
    #     print(1111111111)
    #     logger.info("Initializing the Inpainting UNet from the pretrained 4 channel UNet .")
    #     in_channels = 9
    #     out_channels = unet.conv_in.out_channels
    #     unet.register_to_config(in_channels=in_channels)
    #
    #     with torch.no_grad():
    #         new_conv_in = torch.nn.Conv2d(
    #             in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
    #         )
    #         new_conv_in.weight.zero_()
    #         new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    #         unet.conv_in = new_conv_in

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # 初始化适配器（跳过交叉注意力）
    init_adapter(unet,  cross_attn_cls=SkipAttnProcessor)


    for name, param in unet.named_parameters():
        if 'attn1' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name or 'to_out' in name):
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
    # 打印所有训练参数的名称和形状
    # print("===== 训练参数列表 =====")
    # total_params = 0
    # for name, param in itertools.chain(
    #         #ip_adapter.image_proj_model.named_parameters(),  # 包含参数名称
    #         unet.named_parameters()
    # ):
    #     if param.requires_grad:  # 仅打印需要梯度的参数
    #         print(f"参数名称: {name}")
    #         print(f"形状: {param.shape}")
    #         print(f"参数数量: {param.numel()}")
    #         print("-" * 50)
    #         total_params += param.numel()
    #
    # print(f"\n总训练参数数量: {total_params}")
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
    # 初始化EMA（在 accelerator.prepare 之后）
    ema = EMA(unet, decay=0.9999)  # 可调整decay参数，decay越大模型越趋于稳定（稳定的意思就是和上一次更新的权重值相差不大）
    # def print_model_structure(model, prefix=""):
    #     for name, module in model.named_children():
    #         print(f"{prefix}{name} ({module.__class__.__name__})")
    #         if len(list(module.children())) > 0:
    #             print_model_structure(module, prefix + "  ")
    #         else:
    #             # 打印叶子层的参数形状（可选）
    #             for param_name, param in module.named_parameters():
    #                 print(f"{prefix}  {param_name}: {param.shape}")

    # 示例：打印IPAdapter结构
    #print_model_structure(unet)
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
                    person_latent = compute_vae_encodings(person, vae)  # 模型需要学习的信息
                    masked_latent = compute_vae_encodings(masked_image, vae)
                    cloth_latents = compute_vae_encodings(cloth, vae)
                    mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")

                    del person, mask
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
                    #image_latents = torch.cat([masked_latent, cloth_latents], dim=3)  # 沿宽度拼接
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
                #noise_scheduler.set_timesteps(args.num_inference_steps, device=args.device)
                #timesteps = noise_scheduler.timesteps
                #latents = latents * noise_scheduler.init_noise_sigma
                #noise = torch.randn_like(image_latents)
                bsz = noise.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=noise.device)
                timesteps = timesteps.long()
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
                        #image_width,
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
                        #pred = pred[:, :, :image_width, :]  # 对应上我们的预测的图像
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
                        #image_width=image_width,  # 添加的变量
                        noisy_latents_orin=noisy_latents,
                        mask_latent_concat=mask_latent_concat,
                        image_latents=image_latents,
                        # args.dream_detail_preservation
                    )
                '############'

                noise_pred = unet(combined_latents, timesteps, encoder_hidden_states=None,return_dict=False,)[0]
                # noise_pred = unet(combined_latents, timesteps, encoder_hidden_states=torch.zeros((6, 4, 768)), return_dict=False, )[0]
                # 切片噪声预测和真实噪声，仅保留图像部分
                # noise_pred_image = noise_pred[:, :, :image_width, :]  # [B, 4, W_image, H]
                #print("noise.shape:", noise.shape)
                #print("noise_pred.shape:", noise_pred.shape)
                # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                #使用DREAM
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                #loss = torch.nan_to_num(loss)

                # Backpropagate
                accelerator.backward(loss)
                # if (global_step + 1) % 16 == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # 更新EMA权重（仅在梯度同步时更新）
                #ema.update()
                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, global_step, load_data_time, time.perf_counter() - begin, avg_loss))
                # if (global_step + 1) % 16 == 0:
                #     optimizer.step()
                #     lr_scheduler.step()
                #     optimizer.zero_grad()
                #     if accelerator.is_main_process:
                #         print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                #             epoch, global_step, load_data_time, time.perf_counter() - begin, avg_loss))
                # for name, param in unet.named_parameters():
                #     if param.requires_grad and param.grad is None:
                #         print(f"参数 {name} 无梯度！")

                # optimizer.step()
                # lr_scheduler.step()
                # optimizer.zero_grad()



            global_step += 1
            if accelerator.is_main_process:
                if global_step % args.save_steps == 0:
                    print("Saving model...")
                    logging.info(
                        f"Epoch {epoch}, Step {global_step}: Loss {loss.item()}, LR {lr_scheduler.get_last_lr()[0]}"
                    )
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    # 保存EMA权重（需临时应用影子权重）
                    #ema.apply_shadow()
                    accelerator.save_state(save_path)
                    #ema.restore()  # 恢复原始权重以继续训练
                    # 如果 accelerator.save_state 不支持直接保存调度器，手动补充保存（备用方案）
                    # scheduler_path = os.path.join(save_path, "scheduler.bin")
                    # torch.save(lr_scheduler.state_dict(), scheduler_path)
                if  global_step==100 or global_step==500 or global_step==1000 or global_step==2000 or global_step==3000 or global_step==8000:
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
    parser.add_argument("--output_dir", type=str, default="/mnt/sda2/lry/catViton_dreamedit_accumalateNew_noema_batch48_0509")
    parser.add_argument("--img_size", type=tuple, default=(384, 512))
    parser.add_argument("--train_batch_size", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    #parser.add_argument("--warmup_steps", type=int, default=7275)
    #parser.add_argument("--warmup_steps", type=int, default=12800)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    #parser.add_argument("--max_train_steps", type=int, default=145500)
    parser.add_argument("--max_train_steps", type=int, default=20050)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--mixed_precision", action="store_true",default="no")
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