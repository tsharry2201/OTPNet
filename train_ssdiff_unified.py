"""
SSDiff统一训练脚本
支持三种训练模式：
1. L1模式：直接监督学习
2. VSD模式：知识蒸馏
3. Mixed模式：L1 + VSD混合训练
"""
import os
import sys
import argparse
import datetime
import socket
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
import wandb

sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main')

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler

from ssdiff_unified import SSDiff_Unified_gen, switch_conv2d_to_arconv_with_interpolation
from configs.option_DPM_pansharpening import parser_args as ssdiff_parser_args
from pancollection.common.psdata import PansharpeningSession as DataSession
from train_utils_progressive import (
    get_training_stage, 
    print_stage_transition,
)


def parse_args(input_args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train SSDiff Unified Model')
    
    # 基础路径
    parser.add_argument("--pretrained_ssdiff_path", type=str, required=True)
    parser.add_argument("--lora_checkpoint_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/ssdiff_unified")
    
    # 训练模式
    parser.add_argument("--loss_mode", type=str, default="mixed",
                       choices=["l1", "vsd", "mixed"],
                       help="训练模式: l1(直接监督), vsd(蒸馏), mixed(混合)")
    parser.add_argument("--lambda_l1", type=float, default=1.0,
                       help="L1损失权重（mixed模式）")
    parser.add_argument("--lambda_vsd", type=float, default=1.0,
                       help="VSD损失权重（mixed模式）")
    parser.add_argument("--lambda_distribution", type=float, default=1.0,
                       help="分布匹配损失权重（蒸馏一致性）")
    parser.add_argument("--lambda_diff", type=float, default=1.0,
                       help="扩散一致性损失权重（单步蒸馏）")
    
    # 训练参数
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    
    # 学习率调度
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    
    # 优化器参数
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # LoRA配置
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--train_lora", type=lambda x: str(x).lower() == 'true', default=True)
    
    # VAE配置
    parser.add_argument("--use_vae", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--vae_latent_dim", type=int, default=256)
    parser.add_argument("--train_vae", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--vae_lr", type=float, default=1e-5)
    parser.add_argument("--use_kl_loss", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--lambda_kl", type=float, default=0.001)
    parser.add_argument("--use_perceptual_loss", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--lambda_perceptual", type=float, default=0.1)
    
    # ControlNet配置
    parser.add_argument("--use_controlnet", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--ms_channels", type=int, default=8)
    parser.add_argument("--train_controlnet", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--controlnet_lr", type=float, default=5e-5)
    
    # CLIP配置
    parser.add_argument("--use_clip", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="A high resolution satellite image")
    parser.add_argument("--train_clip", type=lambda x: str(x).lower() == 'true', default=False)
    
    # RAM配置
    parser.add_argument("--use_ram", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--ram_model_path", type=str, default=None)
    
    # ARConv配置
    parser.add_argument("--use_arconv", type=str, default="True")
    parser.add_argument("--arconv_hw_range", type=str, default="[1,5]")
    parser.add_argument("--arconv_warmup_steps", type=int, default=1000)
    parser.add_argument("--arconv_activate_steps", type=int, default=3000)
    parser.add_argument("--arconv_fixstep", type=int, default=4000)
    
    # 扩散模型参数
    parser.add_argument("--predict_xstart", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--use_ddim", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--timestep_respacing", type=str, default="1000")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    
    # 其他参数
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="train_ssdiff_unified")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--set_grads_to_none", action="store_true")
    
    # SSDiff特定参数
    parser.add_argument("--ms_dim", type=int, default=8)
    parser.add_argument("--pan_dim", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=64)
    
    # 恢复训练
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_step", type=int, default=0)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


def main(args):
    """主训练函数"""
    
    # 初始化Accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # 获取SSDiff配置
    ssdiff_args = ssdiff_parser_args()
    for key, value in vars(args).items():
        setattr(ssdiff_args, key, value)
    
    if hasattr(args, 'train_batch_size'):
        ssdiff_args.samples_per_gpu = args.train_batch_size
    
    # 参数类型转换
    if hasattr(ssdiff_args, 'use_arconv'):
        ssdiff_args.use_arconv = ssdiff_args.use_arconv.lower() == 'true'
    if hasattr(ssdiff_args, 'arconv_hw_range'):
        if isinstance(ssdiff_args.arconv_hw_range, str):
            ssdiff_args.arconv_hw_range = eval(ssdiff_args.arconv_hw_range)
    if hasattr(args, 'use_arconv'):
        args.use_arconv = args.use_arconv.lower() == 'true'
    
    # 创建模型
    print("\n" + "="*70)
    print(f"创建SSDiff统一训练模型 - 模式: {args.loss_mode.upper()}")
    print("="*70)
    print(f"  VAE编码器: {'启用' if ssdiff_args.use_vae else '禁用'}")
    print(f"  ControlNet: {'启用' if ssdiff_args.use_controlnet else '禁用'}")
    print(f"  CLIP文本编码器: {'启用' if ssdiff_args.use_clip else '禁用'}")
    print(f"  RAM Caption生成器: {'启用' if ssdiff_args.use_ram else '禁用'}")
    print(f"  ARConv: {'启用' if ssdiff_args.use_arconv else '禁用'}")
    if args.loss_mode == "mixed":
        print(f"  损失权重: L1={args.lambda_l1}, VSD={args.lambda_vsd}")
    print("="*70)
    
    # ARConv渐进式策略
    original_use_arconv = ssdiff_args.use_arconv
    if original_use_arconv:
        ssdiff_args.use_arconv = False
    
    model_gen = SSDiff_Unified_gen(ssdiff_args)
    model_gen.set_train(
        enable_arconv=False,
        train_vae=args.train_vae,
        train_lora=args.train_lora,
        train_controlnet=args.train_controlnet,
        train_clip=args.train_clip if hasattr(args, 'train_clip') else False
    )
    
    ssdiff_args.use_arconv = original_use_arconv
    print("模型创建完成")
    
    # 收集可训练参数
    lora_params = []
    vae_params = []
    controlnet_params = []
    arconv_params = []
    gate_params = []
    proj_params = []
    
    for n, p in model_gen.unet.named_parameters():
        if p.requires_grad:
            if "lora_down" in n or "lora_up" in n:
                lora_params.append(p)
            elif any(key in n.lower() for key in ['arconv', 'adaptive', 'offset', 'modulation', 'kernel_gen']):
                arconv_params.append(p)
            elif 'gate' in n and 'attn_gate' not in n:
                gate_params.append(p)
            elif 'text_proj' in n:
                proj_params.append(p)
    
    if model_gen.vae_encoder is not None:
        for p in model_gen.vae_encoder.parameters():
            if p.requires_grad:
                vae_params.append(p)
    
    if model_gen.controlnet_full is not None:
        for p in model_gen.controlnet_full.parameters():
            if p.requires_grad:
                controlnet_params.append(p)
    
    print(f"\n可训练参数统计:")
    print(f"  LoRA: {len(lora_params)}")
    print(f"  VAE: {len(vae_params)}")
    print(f"  ControlNet: {len(controlnet_params)}")
    print(f"  ARConv: {len(arconv_params)}")
    print(f"  Gate: {len(gate_params)}")
    print(f"  Projection: {len(proj_params)}")
    
    # 创建参数组
    param_groups = []
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.learning_rate, 'name': 'lora'})
    if vae_params:
        vae_lr = getattr(args, 'vae_lr', args.learning_rate * 0.2)
        param_groups.append({'params': vae_params, 'lr': vae_lr, 'name': 'vae'})
    if controlnet_params:
        controlnet_lr = getattr(args, 'controlnet_lr', args.learning_rate)
        param_groups.append({'params': controlnet_params, 'lr': controlnet_lr, 'name': 'controlnet'})
    if arconv_params:
        param_groups.append({'params': arconv_params, 'lr': args.learning_rate * 0.1, 'name': 'arconv'})
    if gate_params:
        gate_lr = args.learning_rate * 2.0
        param_groups.append({'params': gate_params, 'lr': gate_lr, 'name': 'gate'})
    if proj_params:
        param_groups.append({'params': proj_params, 'lr': args.learning_rate, 'name': 'projection'})
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # 创建数据加载器
    print("创建数据加载器...")
    session = DataSession(ssdiff_args)
    train_dataloader, _, _ = session.get_dataloader(ssdiff_args.dataset['train'], False, None)
    print(f"训练批次数: {len(train_dataloader)}")
    
    # Prepare
    model_gen, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model_gen, optimizer, train_dataloader, lr_scheduler
    )
    
    # 初始化trackers
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    
    # 恢复checkpoint
    global_step = 0
    if args.resume_from_checkpoint is not None:
        print(f"从checkpoint恢复: {args.resume_from_checkpoint}")
        global_step = args.resume_step
    
    # 训练循环
    arconv_switched = False
    current_stage = "warmup"
    
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(train_dataloader):
            # 更新训练阶段
            prev_stage = current_stage
            current_stage = get_training_stage(global_step, args)
            
            if prev_stage != current_stage:
                print_stage_transition(global_step, prev_stage, current_stage, args)
            
            # ARConv切换
            if args.use_arconv and not arconv_switched and global_step >= args.arconv_warmup_steps:
                print(f"\n切换到ARConv（步数: {global_step}）")
                unwrapped_model_gen = accelerator.unwrap_model(model_gen)
                switch_conv2d_to_arconv_with_interpolation(unwrapped_model_gen.unet, args)
                arconv_switched = True
                model_gen = accelerator.prepare(unwrapped_model_gen)
                
                # 重新创建优化器（略，逻辑同 train_ssdiff_vae.py）
                print("ARConv切换完成")
            
            # 获取原始模型（处理DDP包装）
            unwrapped_model = accelerator.unwrap_model(model_gen)
            
            # 更新当前步数
            unwrapped_model.set_step(global_step)
            
            # 设置训练模式
            enable_arconv = current_stage != "warmup"
            unwrapped_model.set_train(
                enable_arconv=enable_arconv,
                train_vae=args.train_vae,
                train_lora=args.train_lora,
                train_controlnet=args.train_controlnet,
                train_clip=args.train_clip if hasattr(args, 'train_clip') else False
            )
            
            with accelerator.accumulate(model_gen):
                # 获取数据
                pan = batch['pan']
                lms = batch['lms']
                ms = batch['ms']
                gt = batch['gt']
                
                if gt.shape[1] > gt.shape[2]:
                    import einops
                    gt = einops.rearrange(gt, 'b h w c -> b c h w')
                
                # 📊 数据范围检查（仅在第一个batch打印）
                if global_step == 0 and accelerator.is_main_process:
                    print("\n" + "="*70)
                    print("📊 数据范围检查")
                    print("="*70)
                    print(f"PAN   - min: {pan.min():.4f}, max: {pan.max():.4f}, mean: {pan.mean():.4f}")
                    print(f"LMS   - min: {lms.min():.4f}, max: {lms.max():.4f}, mean: {lms.mean():.4f}")
                    print(f"MS    - min: {ms.min():.4f}, max: {ms.max():.4f}, mean: {ms.mean():.4f}")
                    print(f"GT    - min: {gt.min():.4f}, max: {gt.max():.4f}, mean: {gt.mean():.4f}")
                    residual_check = gt - lms
                    print(f"GT-LMS - min: {residual_check.min():.4f}, max: {residual_check.max():.4f}, mean: {residual_check.mean():.4f}, std: {residual_check.std():.4f}")
                    print("="*70)
                    if gt.max() > 1.5 or gt.min() < -0.5:
                        print("⚠️  警告：数据范围异常！应该在[0,1]范围内")
                        print("⚠️  这会导致loss异常大！")
                    else:
                        print("✅ 数据范围正常")
                    print("="*70 + "\n")
                
                # 计算损失
                total_loss = 0.0
                loss_dict = {}
                
                if args.loss_mode == "l1":
                    # 仅L1损失
                    loss, _, loss_dict = unwrapped_model.l1_loss(lms, pan, ms, gt)
                    total_loss = loss
                
                elif args.loss_mode == "vsd":
                    # 仅VSD损失
                    loss, _, loss_dict = unwrapped_model.vsd_loss(lms, pan, ms, gt)
                    total_loss = loss
                
                elif args.loss_mode == "mixed":
                    # 混合损失
                    loss_l1, _, loss_dict_l1 = unwrapped_model.l1_loss(lms, pan, ms, gt)
                    loss_vsd, _, loss_dict_vsd = unwrapped_model.vsd_loss(lms, pan, ms, gt)
                    
                    total_loss = args.lambda_l1 * loss_l1 + args.lambda_vsd * loss_vsd
                    
                    loss_dict = {
                        'l1_loss': loss_dict_l1.get('l1_loss', 0.0),
                        'vsd_loss': loss_dict_vsd.get('vsd_loss', 0.0),
                    }
                    if 'kl_loss' in loss_dict_l1:
                        loss_dict['kl_loss'] = loss_dict_l1['kl_loss']
                    if 'perceptual_loss' in loss_dict_l1:
                        loss_dict['perceptual_loss'] = loss_dict_l1['perceptual_loss']

                # 蒸馏相关的附加损失
                use_distribution_loss = args.loss_mode in ("vsd", "mixed") and args.lambda_distribution > 0
                if use_distribution_loss:
                    loss_dist, _, loss_dict_dist = unwrapped_model.distribution_matching_loss(lms, pan, ms, gt)
                    total_loss = total_loss + args.lambda_distribution * loss_dist
                    loss_dict['distribution_matching_loss'] = loss_dict_dist.get(
                        'distribution_matching_loss', loss_dist.detach().item()
                    )
                    loss_dict['distribution_matching_loss_weighted'] = (
                        args.lambda_distribution * loss_dist
                    ).detach().item()

                use_diffusion_loss = args.loss_mode in ("vsd", "mixed") and args.lambda_diff > 0
                if use_diffusion_loss:
                    loss_diff, _, loss_dict_diff = unwrapped_model.diff_loss(lms, pan, ms, gt)
                    total_loss = total_loss + args.lambda_diff * loss_diff
                    loss_dict['diff_loss'] = loss_dict_diff.get(
                        'diff_loss', loss_diff.detach().item()
                    )
                    loss_dict['diff_loss_weighted'] = (
                        args.lambda_diff * loss_diff
                    ).detach().item()
                
                loss_dict['total_loss'] = total_loss.detach().item()
                
                # 反向传播
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_gen.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            
            # 更新进度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    # 记录日志
                    progress_logs = {"loss": total_loss.detach().item()}
                    progress_bar.set_postfix(**progress_logs)
                    
                    # 保存checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(
                            args.output_dir, 
                            "checkpoints", 
                            f"model_{global_step}.pkl"
                        )
                        accelerator.unwrap_model(model_gen).save_model(outf)
                        print(f"Checkpoint保存在步数 {global_step}")
                    
                    # WandB日志
                    if global_step % 10 == 0:
                        wandb_logs = {
                            "train/total_loss": total_loss.item(),
                            "train/step": global_step,
                        }
                        for key, value in loss_dict.items():
                            wandb_logs[f"train/{key}"] = value
                        
                        wandb.log(wandb_logs, step=global_step)
                    
                    accelerator.log(loss_dict, step=global_step)
                
                if global_step >= args.max_train_steps:
                    print(f"\n训练完成! 达到最大步数: {args.max_train_steps}")
                    return
        
        if global_step >= args.max_train_steps:
            wandb.finish()
            break
    
    print("训练完成!")


if __name__ == "__main__":
    args = parse_args()
    
    # 初始化wandb
    run_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    
    wandb.init(
        config=vars(args),
        project="ssdiff-unified-training",
        entity="tszharry-xi-an-jiaotong-university-",
        notes=socket.gethostname(),
        name=f"unified_{args.loss_mode}_{args.lora_rank}",
        dir=run_dir,
        job_type="training",
        mode="offline",
        reinit=True
    )
    
    main(args)
