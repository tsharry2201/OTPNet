"""
SSDiff统一训练模型
结合VAE/ControlNet/CLIP/RAM模块和蒸馏学习（VSD loss）
支持：
1. 直接监督学习（L1 loss）
2. 知识蒸馏（VSD loss from teacher）
3. 混合训练（L1 + VSD）
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/home/zelilin/data/pansharpening/SSDiff_main')
from utils.script_util import create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults

# 直接从 ssdiff_vae_retrain 导入所有模块
from ssdiff_vae_retrain import (
    RAMCaptionGenerator,
    VAEEncoder,
    PerceptualLoss,
    ControlNet,
    ZeroConv,
    initialize_ssdiff_unet_with_lora,
    switch_conv2d_to_arconv_with_interpolation,
    _extract_effective_weight,
    _init_arconv_from_conv2d_interpolation,
)


class SSDiff_Unified_gen(nn.Module):
    """
    统一的生成器模型
    结合：
    1. VAE/ControlNet/CLIP/RAM 条件模块（from retrain）
    2. Teacher-Student 蒸馏框架（from distill）
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.current_step = 0
        
        # ========================================
        # Part 1: Teacher 模型（冻结，用于蒸馏）
        # ========================================
        print("="*70)
        print("初始化 Teacher 模型（冻结，无条件模块）")
        print("="*70)
        teacher_args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
        # 只设置合法的参数，避免传递不存在的参数
        if 'use_arconv' in teacher_args_dict:
            teacher_args_dict['use_arconv'] = False
        if 'use_scene_token' in teacher_args_dict:
            teacher_args_dict['use_scene_token'] = False
        self.unet_teacher, _ = create_model_and_diffusion(**teacher_args_dict)
        
        teacher_state = torch.load(args.pretrained_ssdiff_path, map_location='cpu')
        missing_keys, _ = self.unet_teacher.load_state_dict(teacher_state, strict=False)
        
        self.unet_teacher.eval()
        for p in self.unet_teacher.parameters():
            p.requires_grad = False
        print("Teacher 模型已冻结")
        
        # ========================================
        # Part 2: Student 模型（带 LoRA）
        # ========================================
        print("\n" + "="*70)
        print("初始化 Student 模型（带 LoRA + 条件模块）")
        print("="*70)
        self.unet, self.lora_target_modules = initialize_ssdiff_unet_with_lora(
            args, 
            pretrained_path=args.pretrained_ssdiff_path
        )
        
        # 加载预训练的LoRA权重（如果提供）
        if hasattr(args, 'lora_checkpoint_path') and args.lora_checkpoint_path:
            lora_state = torch.load(args.lora_checkpoint_path, map_location='cpu')
            if 'unet_state_dict' in lora_state:
                for name, param in self.unet.named_parameters():
                    if 'lora' in name and name in lora_state['unet_state_dict']:
                        param.data.copy_(lora_state['unet_state_dict'][name])
                print("已加载预训练 LoRA 权重")
        
        # ========================================
        # Part 3: 条件模块（VAE/ControlNet/CLIP/RAM）
        # ========================================
        print("\n" + "="*70)
        print("初始化条件模块")
        print("="*70)
        
        self.vae_encoder = None
        self.controlnet_full = None
        self.tokenizer = None
        self.text_encoder = None
        self.use_clip = False
        self.use_ram = False
        self.ram_caption_generator = None
        self.use_perceptual_loss = False
        self.use_kl_loss = False
        
        # VAE编码器
        if hasattr(args, 'use_vae') and args.use_vae:
            latent_dim = getattr(args, 'vae_latent_dim', 256)
            self.vae_encoder = VAEEncoder(
                input_channels=1,
                latent_dim=latent_dim
            )
            print(f"已启用VAE编码器（潜在维度: {latent_dim}）")
            
            self.use_perceptual_loss = getattr(args, 'use_perceptual_loss', False)
            if self.use_perceptual_loss:
                self.perceptual_loss = PerceptualLoss()
                self.lambda_perceptual = getattr(args, 'lambda_perceptual', 0.1)
                print(f"  - 感知损失: 启用 (λ={self.lambda_perceptual})")
            
            self.use_kl_loss = getattr(args, 'use_kl_loss', False)
            self.lambda_kl = getattr(args, 'lambda_kl', 0.001)
            if self.use_kl_loss:
                print(f"  - KL散度损失: 启用 (λ={self.lambda_kl})")
        
        # ControlNet
        if hasattr(args, 'use_controlnet') and args.use_controlnet:
            ms_dim = getattr(args, 'ms_channels', 8)
            pan_dim = 1
            self.controlnet_full = ControlNet(
                unet_model=self.unet,
                ms_dim=ms_dim,
                pan_dim=pan_dim
            )
            print(f"已启用完整ControlNet（多尺度空间特征）")
        
        # CLIP文本编码器
        if hasattr(args, 'use_clip') and args.use_clip:
            from transformers import AutoTokenizer, CLIPTextModel
            
            clip_base_path = getattr(args, 'clip_model_path', None)
            if clip_base_path is None:
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                clip_base_path = os.path.join(script_dir, "model", "SD21Base")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    clip_base_path, 
                    subfolder="tokenizer"
                )
                self.text_encoder = CLIPTextModel.from_pretrained(
                    clip_base_path, 
                    subfolder="text_encoder"
                ).to(next(self.unet.parameters()).device)
                
                self.text_encoder.requires_grad_(False)
                self.use_clip = True
                self.default_prompt = getattr(args, 'prompt', "A high resolution satellite image")
                
                print(f"已启用CLIP文本编码器（冻结）")
                print(f"  - 默认提示: \"{self.default_prompt}\"")
            except Exception as e:
                print(f"CLIP模型初始化失败: {e}")
                self.use_clip = False
        
        # RAM Caption生成器
        if hasattr(args, 'use_ram') and args.use_ram:
            self.use_ram = True
            ram_model_path = getattr(args, 'ram_model_path', None)
            
            try:
                device = next(self.unet.parameters()).device
                self.ram_caption_generator = RAMCaptionGenerator(
                    model_path=ram_model_path,
                    device=device
                )
                print(f"已启用RAM Caption生成器")
                print(f"  - 模型路径: {ram_model_path}")
            except Exception as e:
                print(f"RAM模型加载失败: {e}")
                self.use_ram = False
                self.ram_caption_generator = None
        
        # ========================================
        # Part 4: Diffusion
        # ========================================
        _, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        self.inference_timestep = 999
        self.lora_rank = args.lora_rank
        self.training = True
        self._active_device = torch.device("cpu")
        
        print("="*70)
        print("模型初始化完成")
        print("="*70)

    def _ensure_device(self, device):
        """确保子模块位于当前进程的设备，便于多卡并行"""
        if device == getattr(self, "_active_device", None):
            return

        def _move(module):
            if module is not None:
                module.to(device)

        _move(self.unet)
        _move(self.unet_teacher)
        _move(self.vae_encoder)
        _move(self.controlnet_full)
        _move(getattr(self, "text_encoder", None))

        ram_generator = getattr(self, "ram_caption_generator", None)
        if ram_generator is not None:
            if hasattr(ram_generator, "model"):
                ram_generator.model.to(device)
            ram_generator.device = device

        self._active_device = device
    
    def set_step(self, step):
        """设置当前训练步数"""
        self.current_step = step
        if hasattr(self.unet, 'set_epoch'):
            self.unet.set_epoch(step)
    
    def set_train(self, enable_arconv=False, train_vae=False, train_lora=True, 
                  train_controlnet=True, train_clip=False):
        """设置训练模式"""
        self.training = True
        self.unet.train()
        
        trainable_count = {'lora': 0, 'vae': 0, 'arconv': 0, 'controlnet': 0, 'clip': 0, 'gate': 0, 'proj': 0}
        
        # UNet参数
        for n, p in self.unet.named_parameters():
            if "lora_down" in n or "lora_up" in n:
                p.requires_grad = train_lora
                if train_lora:
                    trainable_count['lora'] += 1
            elif any(key in n.lower() for key in ['arconv', 'adaptive', 'offset', 'modulation', 'kernel_gen']):
                p.requires_grad = enable_arconv
                if enable_arconv:
                    trainable_count['arconv'] += 1
            elif 'gate' in n and 'attn_gate' not in n:
                p.requires_grad = True
                trainable_count['gate'] += 1
            elif 'text_proj' in n:
                p.requires_grad = True
                trainable_count['proj'] += 1
            else:
                p.requires_grad = False
        
        # VAE参数
        if self.vae_encoder is not None:
            self.vae_encoder.train()
            for p in self.vae_encoder.parameters():
                p.requires_grad = train_vae
                if train_vae:
                    trainable_count['vae'] += 1
        
        # ControlNet参数
        if self.controlnet_full is not None:
            self.controlnet_full.train()
            for p in self.controlnet_full.parameters():
                p.requires_grad = train_controlnet
                if train_controlnet:
                    trainable_count['controlnet'] += 1
        
        # CLIP参数
        if self.use_clip and self.text_encoder is not None:
            if train_clip:
                self.text_encoder.train()
                for p in self.text_encoder.parameters():
                    p.requires_grad = True
                    trainable_count['clip'] += 1
            else:
                self.text_encoder.eval()
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
        
        if not hasattr(self, '_last_train_state') or self._last_train_state != (enable_arconv, train_vae, train_lora, train_controlnet, train_clip):
            print(f"可训练参数: LoRA={trainable_count['lora']}, VAE={trainable_count['vae']}, "
                  f"ARConv={trainable_count['arconv']}, ControlNet={trainable_count['controlnet']}, "
                  f"CLIP={trainable_count['clip']}, Gate={trainable_count['gate']}, Proj={trainable_count['proj']}")
            self._last_train_state = (enable_arconv, train_vae, train_lora, train_controlnet, train_clip)
    
    def select_clip_semantic_label(self, ram_caption, top_k=3, max_ram_tags=12):
        """使用CLIP对RAM生成的caption进行语义过滤（从 retrain 复制）"""
        labels = [
            "urban area", "residential zone", "industrial zone", "forest", "grassland",
            "mountain", "river", "lake", "sea", "beach", "harbor", "farmland",
            "road network", "bare soil", "wetland", "desert",
            "bridge", "building", "highway", "stadium",
            "coastline", "cliff", "hill slope", "dense vegetation", "open water",
            "cloudy", "coastal city", "harbor town"
        ]
        
        if isinstance(ram_caption, str):
            ram_tags = [tag.strip() for tag in ram_caption.split(',')]
            ram_tags = ram_tags[:max_ram_tags]
            ram_caption_short = ', '.join(ram_tags)
        else:
            ram_caption_short = ram_caption
        
        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            labels + [ram_caption_short],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            text_embeds = self.text_encoder(text_inputs.input_ids)[0]
            text_embeds = text_embeds.mean(dim=1)
            text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-6)
            
            label_embeds = text_embeds[:-1]
            caption_embed = text_embeds[-1:]
            scores = (label_embeds * caption_embed).sum(dim=-1)
            
        topk_idx = torch.topk(scores, k=min(top_k, len(labels))).indices.cpu().numpy()
        topk_labels = [labels[i] for i in topk_idx]
        
        return topk_labels
    
    def encode_prompt(self, prompt_batch):
        """编码文本提示为文本嵌入（从 retrain 复制）"""
        if not self.use_clip or self.tokenizer is None or self.text_encoder is None:
            return None
        
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, 
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device)
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        
        if prompt_embeds_list:
            prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
            prompt_embeds = F.layer_norm(prompt_embeds, prompt_embeds.shape[-1:])
            return prompt_embeds
        return None
    
    def forward(self, lms, pan, ms, gt=None, prompt=None):
        """前向传播"""
        device = lms.device
        batch_size = lms.shape[0]

        self._ensure_device(device)

        # VAE编码器：提取全局场景特征
        vae_features = None
        if self.vae_encoder is not None:
            mu, logvar = self.vae_encoder(pan)
            vae_features = mu
        
        # CLIP文本编码器：提取文本语义特征
        text_embeds = None
        if self.use_clip and self.text_encoder is not None:
            if prompt is None and self.use_ram and self.ram_caption_generator is not None:
                with torch.no_grad():
                    captions = []
                    for i in range(batch_size):
                        pan_i = pan[i]
                        caption_raw = self.ram_caption_generator.generate_caption(pan_i)
                        clip_labels = self.select_clip_semantic_label(caption_raw)
                        caption_filtered = ', '.join(clip_labels)
                        captions.append(caption_filtered)
                    prompt = captions
            
            if prompt is None:
                prompt = getattr(self, 'default_prompt', "A high resolution satellite image")
            
            if isinstance(prompt, str):
                prompt = [prompt] * batch_size
            elif len(prompt) == 1 and batch_size > 1:
                prompt = prompt * batch_size
            
            text_embeds = self.encode_prompt(prompt)
        
        # 生成时间步和噪声
        if self.training:
            timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        else:
            timesteps = torch.full((batch_size,), self.inference_timestep, device=device, dtype=torch.long)
        
        if self.training and gt is not None:
            gt_residual = gt - lms
            noise = torch.randn_like(gt_residual)
            x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        else:
            x_t = torch.zeros_like(lms)
        
        # ControlNet：提取多尺度空间特征
        control_features = None
        if self.controlnet_full is not None:
            control_features = self.controlnet_full(pan, x_t, timesteps)
        
        # UNet前向传播
        residual_pred = self.unet.forward_impl(
            lms, pan, ms, x_t, timesteps, 
            scene_token=vae_features,
            control_features=control_features,
            encoder_hidden_states=text_embeds,
            epoch=self.current_step
        )
        
        if torch.isnan(residual_pred).any():
            residual_pred = torch.nan_to_num(residual_pred, nan=0.0)
        
        if self.args.predict_xstart:
            residual = residual_pred
        else:
            residual = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)
        
        output = lms + residual
        output = output.clamp(0, 1)
        
        return output, residual_pred, vae_features
    
    def l1_loss(self, lms, pan, ms, gt, prompt=None):
        """
        L1重建损失（直接监督）
        """
        batch_size = lms.shape[0]
        device = lms.device

        self._ensure_device(device)

        # VAE编码器
        vae_features = None
        mu = None
        logvar = None
        if self.vae_encoder is not None:
            mu, logvar = self.vae_encoder(pan)
            vae_features = mu
        
        # CLIP文本编码器
        text_embeds = None
        if self.use_clip and self.text_encoder is not None:
            if prompt is None and self.use_ram and self.ram_caption_generator is not None:
                with torch.no_grad():
                    captions = []
                    for i in range(batch_size):
                        pan_i = pan[i]
                        caption_raw = self.ram_caption_generator.generate_caption(pan_i)
                        clip_labels = self.select_clip_semantic_label(caption_raw)
                        caption_filtered = ', '.join(clip_labels)
                        captions.append(caption_filtered)
                    prompt = captions
            
            if prompt is None:
                prompt = getattr(self, 'default_prompt', "A high resolution satellite image")
            
            if isinstance(prompt, str):
                prompt = [prompt] * batch_size
            elif len(prompt) == 1 and batch_size > 1:
                prompt = prompt * batch_size
            
            text_embeds = self.encode_prompt(prompt)
        
        timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        
        gt_residual = gt - lms
        noise = torch.randn_like(gt_residual)
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        
        # ControlNet
        control_features = None
        if self.controlnet_full is not None:
            control_features = self.controlnet_full(pan, x_t, timesteps)
        
        # UNet前向传播
        residual_pred = self.unet.forward_impl(
            lms, pan, ms, x_t, timesteps, 
            scene_token=vae_features,
            control_features=control_features,
            encoder_hidden_states=text_embeds,
            epoch=self.current_step
        )
        
        if self.args.predict_xstart:
            residual_final = residual_pred
        else:
            residual_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)
        
        # 计算L1重建损失
        l1_loss = F.l1_loss(residual_final, gt_residual, reduction="mean")
        total_loss = l1_loss
        
        loss_dict = {
            'l1_loss': l1_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        # KL散度损失
        if self.use_kl_loss and mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            total_loss = total_loss + self.lambda_kl * kl_loss
            loss_dict['kl_loss'] = kl_loss.item()
        
        # 感知损失
        if self.use_perceptual_loss and hasattr(self, 'perceptual_loss'):
            try:
                pred_image = lms + residual_final
                pred_image = pred_image.clamp(0, 1)
                gt = gt.clamp(0, 1)
                
                perceptual_loss = self.perceptual_loss(pred_image, gt)
                
                if perceptual_loss > 5.0:
                    perceptual_loss = torch.log(perceptual_loss + 1.0)
                
                weighted_perceptual_loss = self.lambda_perceptual * perceptual_loss
                total_loss = total_loss + weighted_perceptual_loss
                loss_dict['perceptual_loss'] = perceptual_loss.item()
            except Exception as e:
                print(f"感知损失计算错误: {e}")
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, vae_features, loss_dict
    
    def diff_loss(self, lms, pan, ms, gt, prompt=None):
        """
        扩散一致性损失：参考单步蒸馏流程，固定时间步训练残差预测。
        """
        batch_size = lms.shape[0]
        device = lms.device

        self._ensure_device(device)

        # VAE编码（若启用则使用均值作为scene token）
        vae_features = None
        if self.vae_encoder is not None:
            mu, _ = self.vae_encoder(pan)
            vae_features = mu

        # CLIP文本编码（若启用）
        text_embeds = None
        if self.use_clip and self.text_encoder is not None:
            if prompt is None and self.use_ram and self.ram_caption_generator is not None:
                with torch.no_grad():
                    captions = []
                    for i in range(batch_size):
                        pan_i = pan[i]
                        caption_raw = self.ram_caption_generator.generate_caption(pan_i)
                        clip_labels = self.select_clip_semantic_label(caption_raw)
                        caption_filtered = ', '.join(clip_labels)
                        captions.append(caption_filtered)
                    prompt = captions

            if prompt is None:
                prompt = getattr(self, 'default_prompt', "A high resolution satellite image")

            if isinstance(prompt, str):
                prompt = [prompt] * batch_size
            elif len(prompt) == 1 and batch_size > 1:
                prompt = prompt * batch_size

            text_embeds = self.encode_prompt(prompt)

        # 单步蒸馏默认在最高噪声步（t=999）训练
        timesteps = torch.full((batch_size,), 999, device=device, dtype=torch.long)
        gt_residual = gt - lms
        noise = torch.randn_like(gt_residual)
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)

        if not hasattr(self, "_diff_loss_sampling_logged"):
            try:
                sample_ts = timesteps[: min(3, timesteps.shape[0])].detach().cpu().tolist()
                print(f"[SSDiff_Unified] diff_loss shared timestep sample: {sample_ts}, noise std: {noise.std().item():.4f}")
            except Exception:
                pass
            self._diff_loss_sampling_logged = True

        control_features = None
        if self.controlnet_full is not None:
            control_features = self.controlnet_full(pan, x_t, timesteps)

        residual_pred = self.unet.forward_impl(
            lms, pan, ms, x_t, timesteps,
            scene_token=vae_features,
            control_features=control_features,
            encoder_hidden_states=text_embeds,
            epoch=self.current_step
        )

        if self.args.predict_xstart:
            residual_final = residual_pred
        else:
            residual_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)

        loss_diff = F.l1_loss(residual_final, gt_residual, reduction="mean")

        loss_dict = {
            'diff_loss': loss_diff.item(),
            'total_loss': loss_diff.item(),
        }

        return loss_diff, vae_features, loss_dict
    
    def vsd_loss(self, lms, pan, ms, gt, prompt=None):
        """
        VSD蒸馏损失（从Teacher学习）
        """
        batch_size = lms.shape[0]
        device = lms.device

        self._ensure_device(device)

        # VAE编码器
        vae_features = None
        if self.vae_encoder is not None:
            vae_features = self.vae_encoder(pan)[0]
        
        # CLIP文本编码器
        text_embeds = None
        if self.use_clip and self.text_encoder is not None:
            if prompt is None and self.use_ram and self.ram_caption_generator is not None:
                with torch.no_grad():
                    captions = []
                    for i in range(batch_size):
                        pan_i = pan[i]
                        caption_raw = self.ram_caption_generator.generate_caption(pan_i)
                        clip_labels = self.select_clip_semantic_label(caption_raw)
                        caption_filtered = ', '.join(clip_labels)
                        captions.append(caption_filtered)
                    prompt = captions
            
            if prompt is None:
                prompt = getattr(self, 'default_prompt', "A high resolution satellite image")
            
            if isinstance(prompt, str):
                prompt = [prompt] * batch_size
            elif len(prompt) == 1 and batch_size > 1:
                prompt = prompt * batch_size
            
            text_embeds = self.encode_prompt(prompt)
        
        # 随机时间步
        timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        
        gt_residual = gt - lms
        noise = torch.randn_like(gt_residual)
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        
        # ControlNet
        control_features = None
        if self.controlnet_full is not None:
            control_features = self.controlnet_full(pan, x_t, timesteps)
        
        with torch.no_grad():
            # Teacher预测（不使用条件模块）
            residual_teacher = self.unet_teacher.forward_impl(
                lms, pan, ms, x_t, timesteps, 
                scene_token=None,
                control_features=None,
                encoder_hidden_states=None,
                epoch=0
            )
            
            if self.args.predict_xstart:
                residual_teacher_final = residual_teacher
            else:
                residual_teacher_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_teacher)
            
            output_teacher = (lms + residual_teacher_final).clamp(0, 1)
        
        # Student预测（使用条件模块）
        residual_student = self.unet.forward_impl(
            lms, pan, ms, x_t, timesteps, 
            scene_token=vae_features,
            control_features=control_features,
            encoder_hidden_states=text_embeds,
            epoch=self.current_step
        )
        
        if self.args.predict_xstart:
            residual_student_final = residual_student
        else:
            residual_student_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_student)
        
        output_student = (lms + residual_student_final).clamp(0, 1)
        
        # VSD损失计算
        weighting_factor = torch.abs(gt - output_teacher).mean(dim=[1, 2, 3], keepdim=True) + 1e-8
        grad = (output_student - output_teacher) / weighting_factor
        loss_vsd = F.mse_loss(gt, (gt - grad).detach(), reduction="mean")
        
        loss_dict = {
            'vsd_loss': loss_vsd.item(),
            'total_loss': loss_vsd.item(),
        }
        
        return loss_vsd, vae_features, loss_dict

    def distribution_matching_loss(self, lms, pan, ms, gt, prompt=None):
        """
        分布匹配损失：让Student在随机噪声步与Teacher输出保持一致。
        """
        batch_size = lms.shape[0]
        device = lms.device

        self._ensure_device(device)

        # VAE编码作为scene token
        vae_features = None
        if self.vae_encoder is not None:
            mu, _ = self.vae_encoder(pan)
            vae_features = mu

        # CLIP文本编码（若启用）
        text_embeds = None
        if self.use_clip and self.text_encoder is not None:
            if prompt is None and self.use_ram and self.ram_caption_generator is not None:
                with torch.no_grad():
                    captions = []
                    for i in range(batch_size):
                        pan_i = pan[i]
                        caption_raw = self.ram_caption_generator.generate_caption(pan_i)
                        clip_labels = self.select_clip_semantic_label(caption_raw)
                        caption_filtered = ', '.join(clip_labels)
                        captions.append(caption_filtered)
                    prompt = captions

            if prompt is None:
                prompt = getattr(self, 'default_prompt', "A high resolution satellite image")

            if isinstance(prompt, str):
                prompt = [prompt] * batch_size
            elif len(prompt) == 1 and batch_size > 1:
                prompt = prompt * batch_size

            text_embeds = self.encode_prompt(prompt)

        timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        gt_residual = gt - lms
        noise = torch.randn_like(gt_residual)
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)

        if not hasattr(self, "_distribution_loss_sampling_logged"):
            try:
                sample_ts = timesteps[: min(3, timesteps.shape[0])].detach().cpu().tolist()
                print(f"[SSDiff_Unified] distribution_matching shared timestep sample: {sample_ts}, noise std: {noise.std().item():.4f}")
            except Exception:
                pass
            self._distribution_loss_sampling_logged = True

        control_features = None
        if self.controlnet_full is not None:
            control_features = self.controlnet_full(pan, x_t, timesteps)

        with torch.no_grad():
            residual_teacher = self.unet_teacher.forward_impl(
                lms, pan, ms, x_t, timesteps,
                scene_token=None,
                control_features=None,
                encoder_hidden_states=None,
                epoch=0
            )

            if self.args.predict_xstart:
                residual_teacher_final = residual_teacher
            else:
                residual_teacher_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_teacher)

            output_teacher = (lms + residual_teacher_final).clamp(0, 1)

        residual_student = self.unet.forward_impl(
            lms, pan, ms, x_t, timesteps,
            scene_token=vae_features,
            control_features=control_features,
            encoder_hidden_states=text_embeds,
            epoch=self.current_step
        )

        if self.args.predict_xstart:
            residual_student_final = residual_student
        else:
            residual_student_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_student)

        output_student = (lms + residual_student_final).clamp(0, 1)

        weighting_factor = torch.abs(gt - output_teacher).mean(dim=[1, 2, 3], keepdim=True) + 1e-8
        grad = (output_student - output_teacher) / weighting_factor

        loss_vsd = F.mse_loss(gt, (gt - grad).detach(), reduction="mean")

        loss_dict = {
            'distribution_matching_loss': loss_vsd.item(),
            'total_loss': loss_vsd.item(),
        }

        return loss_vsd, vae_features, loss_dict
    
    def save_model(self, save_path):
        """保存模型"""
        state_dict = {
            'lora_target_modules': self.lora_target_modules,
            'lora_rank': self.lora_rank,
            'unet_state_dict': {},
            'vae_state_dict': {},
            'controlnet_state_dict': {},
            'clip_state_dict': {},
        }
        
        # 保存UNet参数（LoRA + 门控 + ARConv）
        for name, param in self.unet.named_parameters():
            if 'lora' in name or 'gate' in name or 'text_proj' in name:
                state_dict['unet_state_dict'][name] = param.cpu()
            elif any(key in name.lower() for key in ['arconv', 'adaptive', 'offset', 'modulation', 'p_conv', 'l_conv', 'w_conv', 'm_conv', 'b_conv', 'convs.']):
                state_dict['unet_state_dict'][name] = param.cpu()
        
        # 保存VAE参数
        if self.vae_encoder is not None:
            state_dict['vae_state_dict'] = self.vae_encoder.state_dict()
        
        # 保存ControlNet参数
        if self.controlnet_full is not None:
            state_dict['controlnet_state_dict'] = self.controlnet_full.state_dict()
        
        # 保存CLIP参数（如果微调）
        if self.use_clip:
            state_dict['use_clip'] = True
            state_dict['default_prompt'] = getattr(self, 'default_prompt', "A high resolution satellite image")
            if self.text_encoder is not None and any(p.requires_grad for p in self.text_encoder.parameters()):
                state_dict['clip_state_dict'] = self.text_encoder.state_dict()
        
        torch.save(state_dict, save_path)
        print(f"模型已保存到: {save_path}")
