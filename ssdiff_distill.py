"""
SSDiff一步蒸馏模型
模仿OSEDiff的训练范式，将多步SSDiff蒸馏为单步模型
🔥 优化：使用轻量级Scene Token替代CLIP，提供场景条件信息
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
import copy

# 添加SSDiff路径
sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main')
from utils.script_util import create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults
from model.ARConv import ARConv


def _extract_effective_weight(conv_layer):
    """
    提取Conv层的有效权重
    
    - 如果是普通Conv2d：直接返回权重
    - 如果是LoRALayer：返回融合权重（base + LoRA effect）
    
    Args:
        conv_layer: Conv2d或LoRALayer
    
    Returns:
        weight: 有效的卷积权重 [outc, inc, h, w]
    """
    # 检查是否是LoRALayer（通过检查是否有lora_down属性）
    if hasattr(conv_layer, 'base_layer') and hasattr(conv_layer, 'lora_down'):
        print(f"   检测到LoRALayer，计算融合权重...")
        # 获取base权重
        base_weight = conv_layer.base_layer.weight.data.clone()  # [outc, inc, 3, 3]
        
        # 计算LoRA delta
        # LoRA: delta = lora_up(lora_down(I)) * scaling
        # 对于卷积，我们需要计算等效的权重修正
        lora_down_weight = conv_layer.lora_down.weight.data  # [rank, inc, 3, 3]
        lora_up_weight = conv_layer.lora_up.weight.data      # [outc, rank, 1, 1]
        scaling = conv_layer.scaling
        
        # 计算LoRA delta weight
        # 简化：对于kernel_size=3的conv，LoRA的影响可以近似为权重叠加
        # 更精确的做法：lora_delta = conv(lora_up_weight, lora_down_weight)
        # 但由于lora_up是1x1卷积，我们可以直接做矩阵乘法
        rank, inc, kh, kw = lora_down_weight.shape
        outc, rank2, _, _ = lora_up_weight.shape
        
        # Reshape for matmul: [outc, rank] @ [rank, inc*kh*kw]
        lora_down_flat = lora_down_weight.reshape(rank, inc * kh * kw)  # [rank, inc*9]
        lora_up_flat = lora_up_weight.squeeze(-1).squeeze(-1)  # [outc, rank]
        
        # Delta weight: [outc, inc*kh*kw]
        lora_delta_flat = torch.matmul(lora_up_flat, lora_down_flat) * scaling
        lora_delta = lora_delta_flat.reshape(outc, inc, kh, kw)
        
        # 融合权重
        fused_weight = base_weight + lora_delta
        
        print(f"      Base权重范围: [{base_weight.min():.4f}, {base_weight.max():.4f}]")
        print(f"      LoRA delta范围: [{lora_delta.min():.4f}, {lora_delta.max():.4f}]")
        print(f"      融合权重范围: [{fused_weight.min():.4f}, {fused_weight.max():.4f}]")
        
        return fused_weight
    else:
        # 普通Conv2d
        return conv_layer.weight.data.clone()


def switch_conv2d_to_arconv_with_interpolation(model, args):
    """
    动态切换：将模型中的Conv2d（含LoRA）替换为ARConv，使用插值扩展初始化
    
    适用场景：
    - 前N步：Student使用Conv2d训练（继承预训练权重+LoRA）
    - N步后：切换到ARConv，将Conv2d权重通过插值扩展到ARConv的9个卷积核
    
    Args:
        model: Student UNet模型
        args: 配置参数
    """
    print("\n" + "="*80)
    print("🔄 开始切换：Conv2d → ARConv（插值扩展初始化）")
    print("="*80)
    
    # 递归替换所有ResBlock中的Conv2d为ARConv
    def replace_conv_in_module(module, module_name=""):
        from model.SSNet import ResBlock
        
        if isinstance(module, ResBlock):
            # 检查是否已经是ARConv
            if isinstance(module.conv0, ARConv):
                print(f"⏭️  {module_name}: 已经是ARConv，跳过")
                return
            
            # 🔥 提取有效权重（融合Conv2d base + LoRA delta）
            conv0_weight = _extract_effective_weight(module.conv0)  # [outc, inc, 3, 3]
            conv1_weight = _extract_effective_weight(module.conv1)  # [outc, inc, 3, 3]
            
            # 创建新的ARConv
            in_channels = conv0_weight.shape[1]      # inc
            hidden_channels = conv0_weight.shape[0]   # outc of conv0
            out_channels = conv1_weight.shape[0]      # outc of conv1
            
            new_conv0 = ARConv(in_channels, hidden_channels, 3, 1, 1)
            new_conv1 = ARConv(hidden_channels, out_channels, 3, 1, 1)
            
            # 使用插值扩展初始化ARConv的9个卷积核
            _init_arconv_from_conv2d_interpolation(new_conv0, conv0_weight)
            _init_arconv_from_conv2d_interpolation(new_conv1, conv1_weight)
            
            # 替换
            module.conv0 = new_conv0
            module.conv1 = new_conv1
            module.use_arconv = True
            
            # 🔥 确保arconv_hw_range是列表格式
            if hasattr(args, 'arconv_hw_range'):
                hw_range = args.arconv_hw_range
                # 如果是字符串，转换为列表
                if isinstance(hw_range, str):
                    hw_range = eval(hw_range)
                module.arconv_hw_range = hw_range
            else:
                module.arconv_hw_range = [1, 9]  # 默认值
            
            print(f"✅ {module_name}: Conv2d+LoRA → ARConv (插值扩展，hw_range={module.arconv_hw_range})")
    
        # 递归处理子模块
        for name, child in module.named_children():
            replace_conv_in_module(child, f"{module_name}.{name}" if module_name else name)
    
    replace_conv_in_module(model, "unet")
    
    print("="*80)
    print("🎉 切换完成！ARConv已从Conv2d+LoRA权重初始化")
    print("="*80 + "\n")


def _init_arconv_from_conv2d_interpolation(arconv_model, conv2d_weight):
    """
    🔥 更严格的ARConv恒等起步初始化
    
    策略：
    - 切换瞬间仅保留3×3路径：混合权重对3×3置1，其它核置0
    - offset全0：无空间偏移
    - modulation初值1.0（而不是tanh≈0.96），避免幅值收缩
    
    Args:
        arconv_model: ARConv模块
        conv2d_weight: Conv2d权重 [outc, inc, 3, 3]
    """
    kernel_sizes = [(3,3), (3,5), (5,3), (3,7), (7,3), (5,5), (5,7), (7,5), (7,7)]
    
    # 🔥 新策略：先用插值初始化所有kernel，但仅激活3×3
    for i, (h, w) in enumerate(kernel_sizes):
        if h == 3 and w == 3:
            # 3x3直接复制
            arconv_model.convs[i].weight.data = conv2d_weight.clone()
        else:
            # 其他尺寸：使用插值扩展（为后续训练准备）
            expanded_weight = F.interpolate(
                conv2d_weight, 
                size=(h, w), 
                mode='bilinear', 
                align_corners=True
            )
            arconv_model.convs[i].weight.data = expanded_weight
    
    # 🔥 关键改进：更严格的"恒等起步"
    
    # 1. m_conv（modulation）：初始化为1.0（无调制）
    #    out = x_offset * m + bias，m=1时相当于 out = x_offset + bias
    #    修改：不使用tanh≈0.96（会收缩幅值），直接初始化输出为接近0（经tanh后接近1）
    nn.init.zeros_(arconv_model.m_conv[6].weight)
    if arconv_model.m_conv[6].bias is not None:
        # 修改：使用更大的bias让tanh输出更接近1
        nn.init.constant_(arconv_model.m_conv[6].bias, 3.0)  # Tanh(3.0) ≈ 0.995
    
    # 2. b_conv（bias）：初始化为0（无额外bias）
    nn.init.zeros_(arconv_model.b_conv[6].weight)
    if arconv_model.b_conv[6].bias is not None:
        nn.init.zeros_(arconv_model.b_conv[6].bias)
    
    # 3. p_conv（offset position）：初始化为0（无偏移）
    nn.init.zeros_(arconv_model.p_conv[4].weight)
    if arconv_model.p_conv[4].bias is not None:
        nn.init.zeros_(arconv_model.p_conv[4].bias)
    
    # 4. l_conv和w_conv（kernel size selection）：
    #    🔥 修改：初始化为更接近3的值（Sigmoid输出≈0）
    #    l = Sigmoid(output) * (hw_range[1] - 1) + 1
    #    Sigmoid(-5.0) ≈ 0.007，则 l ≈ 0.007*4+1 ≈ 1（对于hw_range=[1,5]）
    #    但我们希望初始为3，所以Sigmoid应该≈0.5，即bias=0
    nn.init.zeros_(arconv_model.l_conv[4].weight)
    if arconv_model.l_conv[4].bias is not None:
        # 对于hw_range=[1,5]：Sigmoid(0)=0.5 → 0.5*4+1=3
        nn.init.constant_(arconv_model.l_conv[4].bias, 0.0)
    
    nn.init.zeros_(arconv_model.w_conv[4].weight)
    if arconv_model.w_conv[4].bias is not None:
        nn.init.constant_(arconv_model.w_conv[4].bias, 0.0)
    
    print("      🎯 ARConv恒等起步初始化策略:")
    print("         - 卷积核: 从Conv2d插值扩展")
    print("         - Modulation: 初始化为≈1.0（无幅值收缩）")
    print("         - Bias: 初始化为0（无偏移）")
    print("         - Offset: 初始化为0（无空间偏移）")
    print("         - Kernel size: 初始化为3x3（稳定起点，hw_range=[1,5]）")
    

def initialize_ssdiff_unet_with_lora(args, pretrained_path=None):
    """
    初始化SSDiff的UNet并添加LoRA层
    
    Args:
        args: 配置参数
        pretrained_path: 预训练模型路径
    
    Returns:
        unet: 带LoRA的UNet模型
        lora_target_modules: LoRA目标模块列表
    """
    # 创建Student模型（配置跟随args）
    student_args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    student_args_dict['use_arconv'] = args.use_arconv  # 🔥 Student的ARConv设置跟随args
    student_args_dict['use_scene_token'] = args.use_scene_token  # 🔥 Student的Scene Token设置跟随args
    model, _ = create_model_and_diffusion(**student_args_dict)
    
    # 加载预训练权重
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"Loading pretrained SSDiff from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # 使用strict=False，因为我们添加了scene_embed层
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  缺失的键（新添加的层，将随机初始化）: {len(missing_keys)} 个")
            # 打印scene_embed相关的缺失键
            scene_keys = [k for k in missing_keys if 'scene_embed' in k]
            if scene_keys:
                print(f"   Scene embed层: {len(scene_keys)} 个参数")
        print("✅ Pretrained SSDiff loaded successfully!")
    
    # 冻结所有参数
    model.requires_grad_(False)
    model.train()
    
    # 找到所有可以添加LoRA的层
    lora_target_modules = []
    for name, module in model.named_modules():
        # 为Conv2d和Linear层添加LoRA
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 跳过输入输出层和一些特殊层
            if any(skip in name for skip in ['time_embed', 'label_emb']):
                continue
            lora_target_modules.append(name)
    
    # 只保留部分关键层（避免过度参数化）
    # 优先选择attention和残差块中的层
    filtered_modules = []
    for name in lora_target_modules:
        if any(key in name for key in ['attn', 'in_layers', 'out_layers', 'skip_connection']):
            filtered_modules.append(name)
    
    if len(filtered_modules) == 0:
        filtered_modules = lora_target_modules  # 如果过滤后为空，使用全部
    
    print(f"Found {len(filtered_modules)} modules for LoRA")
    print(f"Sample modules: {filtered_modules[:5]}")
    
    # 手动为Conv2d层添加LoRA参数
    # 使用简单的LoRA实现，避免PEFT库的复杂性
    class LoRALayer(nn.Module):
        def __init__(self, base_layer, rank=4, alpha=8):
            super().__init__()
            self.base_layer = base_layer
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
            # 为Conv2d创建LoRA参数
            if isinstance(base_layer, nn.Conv2d):
                self.lora_down = nn.Conv2d(
                    base_layer.in_channels, 
                    rank, 
                    kernel_size=base_layer.kernel_size,
                    stride=base_layer.stride,
                    padding=base_layer.padding,
                    bias=False
                )
                self.lora_up = nn.Conv2d(
                    rank, 
                    base_layer.out_channels, 
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                )
                # 初始化
                nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
                nn.init.zeros_(self.lora_up.weight)
        
        def forward(self, x):
            base_out = self.base_layer(x)
            if hasattr(self, 'lora_down'):
                # 确保 LoRA 层使用与输入相同的 dtype
                x_dtype = x.dtype
                lora_down_out = self.lora_down(x.to(self.lora_down.weight.dtype))
                lora_out = self.lora_up(lora_down_out) * self.scaling
                # 转换回原始 dtype
                lora_out = lora_out.to(x_dtype)
                return base_out + lora_out
            return base_out
    
    # 为选定的模块添加LoRA
    lora_count = 0
    for name, module in model.named_modules():
        if name in filtered_modules and isinstance(module, nn.Conv2d):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            
            # 创建LoRA包装层
            lora_layer = LoRALayer(module, rank=args.lora_rank, alpha=args.lora_rank * 2)
            setattr(parent, child_name, lora_layer)
            lora_count += 1
    
    print(f"✅ Added LoRA to {lora_count} Conv2d layers")
    return model, filtered_modules


class SceneTokenExtractor(nn.Module):
    """
    🔥 轻量级场景特征提取器
    从PAN图像中自动提取场景条件token，替代CLIP
    
    优势：
    1. 任务特定：直接学习遥感图像的场景特征
    2. 轻量级：只有~100K参数，计算开销极小
    3. 端到端可训练：与蒸馏任务联合优化
    4. 无需额外标注：自动从图像提取
    """
    def __init__(self, input_channels=1, token_dim=256):
        super().__init__()
        self.token_dim = token_dim
        
        # 多尺度特征提取
        self.encoder = nn.Sequential(
            # Stage 1: 64x64 -> 32x32
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Stage 2: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            
            # Stage 3: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Project to token space
            nn.Linear(256, token_dim),
            nn.LayerNorm(token_dim)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pan):
        """
        Args:
            pan: [B, 1, H, W] 全色图像
        
        Returns:
            scene_token: [B, token_dim] 场景特征向量
        """
        return self.encoder(pan)


class SSDiff_gen(nn.Module):
    """
    单步生成器模型
    类似于OSEDiff_gen，但针对SSDiff的全景锐化任务
    🔥 优化：使用轻量级Scene Token提供场景条件信息
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.current_step = 0  # 🔥 追踪当前训练步数（用于ARConv的epoch参数）
        
        # 🔥 初始化Scene Token提取器（轻量级，可训练）
        if hasattr(args, 'use_scene_token') and args.use_scene_token:
            token_dim = getattr(args, 'scene_token_dim', 256)
            print(f"🎨 Initializing Scene Token Extractor (dim={token_dim})...")
            self.scene_extractor = SceneTokenExtractor(
                input_channels=1,  # PAN通道数
                token_dim=token_dim
            )
            print("✅ Scene Token Extractor initialized")
        else:
            self.scene_extractor = None
            print("ℹ️  Scene Token disabled")
        
        # 🔥 OSEDiff风格修改：添加Teacher模型（冻结）
        # Teacher：原始预训练的SSDiff（无ARConv，无Scene Token，无LoRA）
        print("📚 Loading Teacher model (frozen, no ARConv, no Scene Token)...")
        # Teacher不使用ARConv和Scene Token（因为预训练模型没有）
        teacher_args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
        teacher_args_dict['use_arconv'] = False  # 🔥 Teacher不使用ARConv
        teacher_args_dict['use_scene_token'] = False  # 🔥 Teacher不使用Scene Token
        self.unet_teacher, _ = create_model_and_diffusion(**teacher_args_dict)
        # 加载teacher权重（使用strict=False，因为预训练模型没有scene_embed层）
        teacher_state = torch.load(args.pretrained_ssdiff_path, map_location='cpu')
        missing_keys, unexpected_keys = self.unet_teacher.load_state_dict(teacher_state, strict=False)
        
        # 打印缺失和多余的键（用于调试）
        if missing_keys:
            print(f"⚠️  Teacher缺失的键（这些是新添加的，正常）: {missing_keys[:5]}...")  # 只打印前5个
        if unexpected_keys:
            print(f"⚠️  Teacher多余的键: {unexpected_keys}")
        
        # 冻结teacher
        self.unet_teacher.eval()
        for p in self.unet_teacher.parameters():
            p.requires_grad = False
        print("✅ Teacher model loaded and frozen")
        
        # Student：添加LoRA的可训练模型
        print("🎓 Loading Student model (with LoRA)...")
        self.unet, self.lora_target_modules = initialize_ssdiff_unet_with_lora(
            args, 
            pretrained_path=args.pretrained_ssdiff_path
        )
        print("✅ Student model loaded with LoRA")
        
        # 创建diffusion（用于单步去噪，使用顶部已导入的函数）
        _, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # 打印predict_xstart设置（训练时）
        print(f"🔍 [SSDiff_gen 训练] predict_xstart = {args.predict_xstart}")
        
        # 🔥 OSEDiff风格修改：不再固定timestep，训练时动态生成
        # 推理时使用t=999，训练时使用随机timestep [100, 999)
        self.inference_timestep = 999
        
        self.lora_rank = args.lora_rank
        self.training = True  # 添加training标志
    
    def set_step(self, step):
        """设置当前训练步数（用于ARConv的epoch参数和渐进式训练）"""
        self.current_step = step
        # 将step传递给unet（ARConv会根据epoch判断是否固定）
        if hasattr(self.unet, 'set_epoch'):
            self.unet.set_epoch(step)
    
    def set_train(self, enable_arconv=False, freeze_scene_token=False):
        """
        设置训练模式，LoRA层、Scene Extractor和ARConv可训练
        
        Args:
            enable_arconv: 是否启用ARConv训练（渐进式解冻用）
                          注意：fixstep后ARConv的offset/modulation会固定，
                          但卷积权重仍继续训练
            freeze_scene_token: 是否冻结Scene Token Extractor（前300步冻结）
        """
        self.training = True
        self.unet.train()
        
        trainable_count = {'lora': 0, 'scene': 0, 'scene_gate': 0, 'arconv': 0}
        
        # UNet的参数设置
        for n, p in self.unet.named_parameters():
            # LoRA层始终可训练
            if "lora_down" in n or "lora_up" in n:
                p.requires_grad = True
                trainable_count['lora'] += 1
            # 🔥 Scene Gate参数（门控α）始终可训练
            elif "scene_gate" in n:
                p.requires_grad = True
                trainable_count['scene_gate'] += 1
            # 🔥 ARConv根据阶段决定是否可训练
            # 包括：offset_conv, modulation_conv, weight等所有参数
            elif any(key in n.lower() for key in ['arconv', 'adaptive', 'offset', 'modulation', 'kernel_gen']):
                p.requires_grad = enable_arconv
                if enable_arconv:
                    trainable_count['arconv'] += 1
            else:
                p.requires_grad = False
        
        # 🔥 Scene Extractor：前300步冻结，之后解冻（学习率为主干的0.1x）
        if self.scene_extractor is not None:
            self.scene_extractor.train()
            for p in self.scene_extractor.parameters():
                p.requires_grad = not freeze_scene_token
                if not freeze_scene_token:
                    trainable_count['scene'] += 1
        
        # 只在第一次调用或状态改变时打印
        if not hasattr(self, '_last_arconv_state') or self._last_arconv_state != enable_arconv:
            status = "训练中（offset+modulation+weight）" if enable_arconv and self.current_step < 4000 else \
                     "训练中（仅weight，offset/modulation已固定）" if enable_arconv else "冻结"
            scene_status = "冻结" if freeze_scene_token else "训练中"
            print(f"🔥 可训练参数: LoRA={trainable_count['lora']}, "
                  f"Scene={trainable_count['scene']} ({scene_status}), SceneGate={trainable_count['scene_gate']}, "
                  f"ARConv={trainable_count['arconv']} ({status})")
            self._last_arconv_state = enable_arconv
    
    def forward(self, lms, pan, ms, gt=None):
        """
        前向传播（残差学习版本，与原SSDiff对齐）
        🔥 优化：支持Scene Token conditioning
        
        Args:
            lms: 低分辨率多光谱图像 [B, 8, H, W]
            pan: 全色图像 [B, 1, H, W]
            ms: 上采样的多光谱图像 [B, 8, H, W]
            gt: Ground truth (训练时使用) [B, 8, H, W]
        
        Returns:
            output: 预测的高分辨率多光谱图像 [B, 8, H, W]
            residual_pred: UNet预测的残差
            scene_token: Scene token features (如果启用)
        """
        device = lms.device
        batch_size = lms.shape[0]
        
        # 🔥 提取scene token (从PAN提取)
        scene_token = None
        if self.scene_extractor is not None:
            scene_token = self.scene_extractor(pan)  # [B, token_dim]
        
        # 🔥 训练时使用随机timestep
        if self.training:
            # 训练：随机timestep [100, 999)，覆盖多种噪声水平
            timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        else:
            # 推理：固定t=999单步
            timesteps = torch.full((batch_size,), self.inference_timestep, device=device, dtype=torch.long)
        
        # 🔥 核心修改：与原SSDiff完全一致，使用q_sample_xt
        if self.training and gt is not None:
            # 训练时：计算真残差，使用原SSDiff的扩散过程
            gt_residual = gt - lms  # 真残差
            
            # 使用原SSDiff的q_sample_xt进行加噪（与原SSDiff完全一致）
            noise = torch.randn_like(gt_residual)
            x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        else:
            # 推理时：输入为0（初始噪声残差的近似）
            x_t = torch.zeros_like(lms)
        
        # UNet预测残差（使用forward_impl方法）
        # forward_impl(self, lms, pan, ms, x_t, timesteps, scene_token, epoch)
        # 参数说明：lms(64x64), pan(64x64), ms(16x16会被upsample), x_t(噪声残差, 64x64)
        # 注意：x_t现在是噪声残差，与原SSDiff的输入一致
        # epoch参数用于ARConv判断是否固定卷积核
        residual_pred = self.unet.forward_impl(lms, pan, ms, x_t, timesteps, scene_token=scene_token, epoch=self.current_step)
        
        # 检查 UNet 输出
        if torch.isnan(residual_pred).any():
            print(f"[SSDiff_gen] residual_pred contains NaN after forward_impl!")
            print(f"  Input ranges - lms: [{lms.min():.4f}, {lms.max():.4f}], x_t: [{x_t.min():.4f}, {x_t.max():.4f}]")
            # 将 NaN 替换为 0
            residual_pred = torch.nan_to_num(residual_pred, nan=0.0)
        
        # 单步去噪得到残差
        if self.args.predict_xstart:
            # 如果模型预测x0，直接使用（这里x0就是残差）
            residual = residual_pred
        else:
            # 从噪声预测计算x0（残差）
            residual = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)
        
        # 最终输出 = LMS + 残差
        output = lms + residual
        
        # 检查最终输出
        if torch.isnan(output).any():
            print(f"[SSDiff_gen] output contains NaN!")
            output = torch.nan_to_num(output, nan=0.0)
        
        # 裁剪到有效范围
        output = output.clamp(0, 1)
        
        return output, residual_pred, scene_token
    
    def distribution_matching_loss(self, lms, pan, ms, gt):
        """
        🔥 OSEDiff风格的VSD (Variational Score Distillation) 损失
        🔥 优化：使用Scene Token增强语义一致性
        
        核心思想：Student学习Teacher在不同噪声水平下的预测分布，而不是直接学GT
        
        Args:
            lms: 低分辨率多光谱图像 [B, 8, H, W]
            pan: 全色图像 [B, 1, H, W]
            ms: 上采样的多光谱图像 [B, 8, H, W]
            gt: Ground truth [B, 8, H, W]
        
        Returns:
            loss_vsd: VSD损失标量
            scene_token: Scene token features (如果启用)
        """
        batch_size = lms.shape[0]
        device = lms.device
        
        # 🔥 提取scene token (从PAN提取)
        scene_token = None
        if self.scene_extractor is not None:
            scene_token = self.scene_extractor(pan)
        
        # 随机timestep（与forward中训练时的范围一致）
        timesteps = torch.randint(100, 999, (batch_size,), device=device, dtype=torch.long)
        
        # 🔥 核心修改：使用原SSDiff的q_sample_xt（完全一致）
        gt_residual = gt - lms  # 计算真残差
        noise = torch.randn_like(gt_residual)
        # 使用原SSDiff的扩散过程进行加噪
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        
        with torch.no_grad():
            # Teacher预测（冻结，不计算梯度）- Teacher不使用scene token，也不需要epoch
            residual_teacher = self.unet_teacher.forward_impl(lms, pan, ms, x_t, timesteps, scene_token=None, epoch=0)
            
            if self.args.predict_xstart:
                residual_teacher_final = residual_teacher
            else:
                residual_teacher_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_teacher)
            output_teacher = (lms + residual_teacher_final).clamp(0, 1)
        
        # Student预测（需要梯度）- Student使用scene token和当前step
        residual_student = self.unet.forward_impl(lms, pan, ms, x_t, timesteps, scene_token=scene_token, epoch=self.current_step)
        
        # 🔍 调试：打印Student配置
        if not hasattr(self, '_debug_student_printed'):
            print(f"   - use_arconv: {self.unet.resblock0.use_arconv}")
            print(f"   - use_scene_token: {hasattr(self.unet, 'scene_embed') and self.unet.scene_embed is not None}")
            self._debug_student_printed = True
        if self.args.predict_xstart:
            residual_student_final = residual_student
        else:
            residual_student_final = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_student)
        output_student = (lms + residual_student_final).clamp(0, 1)
        
        # VSD损失计算（OSEDiff风格的实现）
        # 加权因子：基于GT和Teacher预测的差异
        weighting_factor = torch.abs(gt - output_teacher).mean(dim=[1, 2, 3], keepdim=True) + 1e-8
        
        # 🔍 调试：打印关键值
        if self.current_step % 100 == 0:
            student_teacher_diff = torch.abs(output_student - output_teacher).mean().item()
            gt_teacher_diff = torch.abs(gt - output_teacher).mean().item()
            print(f"   - |Student - Teacher|: {student_teacher_diff:.6f}")
            print(f"   - |GT - Teacher|: {gt_teacher_diff:.6f}")
            print(f"   - weighting_factor: {weighting_factor.mean().item():.6f}")
        
        # 梯度：Student和Teacher的差异，经过weighting标准化
        grad = (output_student - output_teacher) / weighting_factor
        
        # 🔍 调试：打印grad值
        if self.current_step % 100 == 0:
            grad_mean = torch.abs(grad).mean().item()
            grad_max = torch.abs(grad).max().item()
            print(f"   - grad_mean: {grad_mean:.6f}, grad_max: {grad_max:.6f}")
        
        # VSD损失：让Student接近Teacher的分布
        # 使用stop_gradient技巧：gt - grad作为target，但grad不传梯度
        loss_vsd = F.mse_loss(gt, (gt - grad).detach(), reduction="mean")
        
        return loss_vsd, scene_token
    
    def save_model(self, save_path):
        """保存模型（LoRA权重 + ARConv权重 + Scene Extractor权重 + Scene Gate）"""
        state_dict = {
            'lora_target_modules': self.lora_target_modules,
            'lora_rank': self.lora_rank,
            'unet_state_dict': {},
            'scene_extractor_state_dict': None,
            'use_arconv': self.args.use_arconv if hasattr(self.args, 'use_arconv') else False,
            'use_scene_token': self.scene_extractor is not None,
        }
        
        # 保存LoRA参数、ARConv参数和Scene Gate
        saved_counts = {'lora': 0, 'arconv': 0, 'scene_gate': 0}
        
        for name, param in self.unet.named_parameters():
            # 保存LoRA权重
            if 'lora' in name:
                state_dict['unet_state_dict'][name] = param.cpu()
                saved_counts['lora'] += 1
            # 🔥 保存Scene Gate（场景token的门控参数）
            elif 'scene_gate' in name:
                state_dict['unet_state_dict'][name] = param.cpu()
                saved_counts['scene_gate'] += 1
            # 🔥 保存ARConv相关权重（所有与自适应卷积相关的参数）
            elif any(key in name.lower() for key in ['arconv', 'adaptive', 'offset', 'modulation', 'p_conv', 'l_conv', 'w_conv', 'm_conv', 'b_conv', 'convs.']):
                state_dict['unet_state_dict'][name] = param.cpu()
                saved_counts['arconv'] += 1
        
        # 🔥 保存Scene Extractor权重（如果存在）
        if self.scene_extractor is not None:
            state_dict['scene_extractor_state_dict'] = self.scene_extractor.state_dict()
        
        # 打印保存统计
        print(f"\n💾 模型保存统计:")
        print(f"   LoRA参数: {saved_counts['lora']}")
        print(f"   ARConv参数: {saved_counts['arconv']} {'✅' if saved_counts['arconv'] > 0 else '⚠️ 未找到ARConv参数！'}")
        print(f"   Scene Gate参数: {saved_counts['scene_gate']}")
        if self.scene_extractor is not None:
            print(f"   Scene Extractor: ✅ 已保存")
        print(f"   总参数: {len(state_dict['unet_state_dict'])}")
        
        torch.save(state_dict, save_path)
        print(f"💾 Model saved to: {save_path}\n")


class SSDiff_reg(nn.Module):
    """
    正则化模型
    类似于OSEDiff_reg，包含固定的UNet和可更新的UNet
    🔥 优化：集成Scene Token Extractor
    """
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        
        # 🔥 初始化Scene Token提取器
        if hasattr(args, 'use_scene_token') and args.use_scene_token:
            token_dim = getattr(args, 'scene_token_dim', 256)
            print(f"🎨 [SSDiff_reg] Initializing Scene Token Extractor (dim={token_dim})...")
            self.scene_extractor = SceneTokenExtractor(
                input_channels=1,  # PAN通道数
                token_dim=token_dim
            )
            self.scene_extractor.to(accelerator.device)
            print("✅ [SSDiff_reg] Scene Token Extractor initialized")
        else:
            self.scene_extractor = None
        
        # 创建固定的UNet（作为教师模型，无ARConv，无Scene Token）
        fix_args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
        fix_args_dict['use_arconv'] = False  # 🔥 unet_fix不使用ARConv（因为预训练模型没有）
        fix_args_dict['use_scene_token'] = False  # 🔥 unet_fix不使用Scene Token（因为预训练模型没有）
        self.unet_fix, _ = create_model_and_diffusion(**fix_args_dict)
        if args.pretrained_ssdiff_path and os.path.exists(args.pretrained_ssdiff_path):
            state_dict = torch.load(args.pretrained_ssdiff_path, map_location='cpu')
            # 使用strict=False，因为预训练模型没有scene_embed层
            missing_keys, _ = self.unet_fix.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️  unet_fix缺失的键（新添加的层）: {len(missing_keys)} 个")
        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()
        
        # 创建可更新的UNet（带LoRA）
        self.unet_update, self.lora_target_modules = initialize_ssdiff_unet_with_lora(
            args,
            pretrained_path=args.pretrained_ssdiff_path
        )
        
        # 创建diffusion（使用顶部已导入的函数）
        _, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # 打印predict_xstart设置（正则化训练）
        print(f"🔍 [SSDiff_reg 训练] predict_xstart = {args.predict_xstart}")
        
        # 设置权重类型
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
    
    def set_train(self):
        """设置训练模式"""
        self.unet_update.train()
        for n, p in self.unet_update.named_parameters():
            if "lora_down" in n or "lora_up" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        # Scene Extractor可训练
        if self.scene_extractor is not None:
            self.scene_extractor.train()
            for p in self.scene_extractor.parameters():
                p.requires_grad = True
    
    def diff_loss(self, lms, pan, ms, gt):
        """
        扩散损失：让LoRA学习残差（与原始SSDiff保持一致）
        改进方案：从MS开始预测残差（GT - LMS）
        使用Scene Token提供额外的场景条件信息
        """
        device = gt.device
        bsz = gt.shape[0]
        
        # 提取scene token
        scene_token = None
        if self.scene_extractor is not None:
            scene_token = self.scene_extractor(pan)
        
        # 计算真实残差
        gt_residual = gt - lms
        
        # 固定在timestep=999（与主模型一致）
        timesteps = torch.full((bsz,), 999, device=device, dtype=torch.long)
        
        # 🔥 核心修改：使用原SSDiff的q_sample_xt（完全一致）
        noise = torch.randn_like(gt_residual)
        # 使用原SSDiff的扩散过程进行加噪
        x_t = self.diffusion.q_sample_xt(gt_residual, timesteps, noise=noise)
        
        # 预测残差（使用forward_impl方法，加入scene_token）
        # 参数：lms(64x64), pan(64x64), ms(16x16会被upsample), x_t(64x64), scene_token
        residual_pred = self.unet_update.forward_impl(lms, pan, ms, x_t, timesteps, scene_token=scene_token)
        
        # 计算损失：预测残差与真实残差的差异（使用L1，与原始SSDiff一致）
        if self.args.predict_xstart:
            # 如果预测x0，这里x0就是残差
            loss = F.l1_loss(residual_pred.float(), gt_residual.float(), reduction="mean")
        else:
            # 如果预测噪声，需要转换为x0（残差）再比较
            residual_pred_x0 = self.diffusion._predict_xstart_from_eps(x_t, timesteps, residual_pred)
            loss = F.l1_loss(residual_pred_x0.float(), gt_residual.float(), reduction="mean")
        
        return loss
    
    def distribution_matching_loss(self, lms, pan, ms, x_pred):
        """
        分布匹配损失：让单步模型的输出接近多步SSDiff的分布
        使用Scene Token提供额外的场景条件信息
        
        Args:
            lms, pan, ms: 条件输入
            x_pred: 学生模型（单步）的预测
        
        Returns:
            loss: 分布匹配损失
        """
        device = x_pred.device
        bsz = x_pred.shape[0]
        
        # 提取scene token
        scene_token = None
        if self.scene_extractor is not None:
            scene_token = self.scene_extractor(pan)
        
        # 检查输入数据
        if torch.isnan(lms).any():
            print(f"[distribution_matching_loss] lms contains NaN!")
        if torch.isnan(pan).any():
            print(f"[distribution_matching_loss] pan contains NaN!")
        if torch.isnan(ms).any():
            print(f"[distribution_matching_loss] ms contains NaN!")
        if torch.isnan(x_pred).any():
            print(f"[distribution_matching_loss] x_pred contains NaN!")
        
        # 随机采样中间时间步
        timesteps = torch.randint(20, 980, (bsz,), device=device).long()
        
        # 🔥 核心修改：对残差添加噪声（与原SSDiff一致）
        x_pred_residual = x_pred - lms  # 预测的残差
        noise = torch.randn_like(x_pred_residual)
        # 使用diffusion的q_sample_xt对残差添加噪声
        noisy_x = self.diffusion.q_sample_xt(x_pred_residual, timesteps, noise=noise)
        
        if torch.isnan(noisy_x).any():
            print(f"[distribution_matching_loss] noisy_x contains NaN after q_sample_xt!")
        
        with torch.no_grad():
            # 学生模型（可更新）的预测（使用forward_impl + scene_token）
            # 参数：lms(64x64), pan(64x64), ms(16x16会被upsample), noisy_x(64x64), scene_token
            noise_pred_update = self.unet_update.forward_impl(lms, pan, ms, noisy_x, timesteps, scene_token=scene_token)
            x0_pred_update = self.diffusion._predict_xstart_from_eps(
                noisy_x, timesteps, noise_pred_update
            )
            
            # 教师模型（固定）的预测（使用forward_impl，不使用scene_token）
            # 参数：lms(64x64), pan(64x64), ms(16x16会被upsample), noisy_x(64x64)
            noise_pred_fix = self.unet_fix.forward_impl(
                lms.to(self.weight_dtype), 
                pan.to(self.weight_dtype),
                ms.to(self.weight_dtype),
                noisy_x.to(self.weight_dtype), 
                timesteps,
                scene_token=None
            )
            x0_pred_fix = self.diffusion._predict_xstart_from_eps(
                noisy_x, timesteps, noise_pred_fix.float()
            )
        
        weighting_factor = torch.abs(x_pred - x0_pred_fix).mean(
            dim=[1, 2, 3], keepdim=True
        ) + 1e-5
        # 计算梯度
        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        
        # VSD损失
        loss = F.mse_loss(x_pred, (x_pred - grad).detach())
        
        return loss


class SSDiff_test(nn.Module):
    """
    测试/推理模型
    支持两种模式：
    1. 原始SSDiff多步采样（use_distillation=False）
    2. 蒸馏后的单步采样（use_distillation=True）
    🔥 优化：支持Scene Token conditioning（可选）
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 🔥 初始化Scene Token Extractor（可选）
        if hasattr(args, 'use_scene_token') and args.use_scene_token:
            token_dim = getattr(args, 'scene_token_dim', 256)
            print(f"🎨 [SSDiff_test] Initializing Scene Token Extractor (dim={token_dim})...")
            self.scene_extractor = SceneTokenExtractor(
                input_channels=1,  # PAN通道数
                token_dim=token_dim
            )
            self.scene_extractor.to(self.device)
            self.scene_extractor.eval()
            print("✅ [SSDiff_test] Scene Token Extractor initialized")
        else:
            self.scene_extractor = None
        
        # 对于蒸馏模式，不使用SpacedDiffusion，直接使用完整的1000步空间
        if args.use_distillation:
            # 临时移除 timestep_respacing，使用完整的 Gaussian Diffusion
            original_respacing = args.timestep_respacing
            args.timestep_respacing = ""  # 空字符串表示不使用spacing
            print(f"📊 Using full 1000-step space for distillation (训练时使用timestep=999)")
        
        # 创建模型和diffusion
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # 恢复原始设置
        if args.use_distillation:
            args.timestep_respacing = original_respacing
            
        # 打印predict_xstart设置（测试时）
        print(f"🔍 [SSDiff_test 测试] predict_xstart = {args.predict_xstart}")
        
        # 加载预训练权重
        if hasattr(args, 'model_path') and args.model_path:
            print(f"Loading model from: {args.model_path}")
            state_dict = torch.load(args.model_path, map_location='cpu')
            
            if args.use_distillation and 'unet_state_dict' in state_dict:
                # 蒸馏模型：先加载基础权重，再加载LoRA
                if hasattr(args, 'pretrained_ssdiff_path'):
                    base_state = torch.load(args.pretrained_ssdiff_path, map_location='cpu')
                    # 使用strict=False，因为我们添加了scene_embed层
                    missing_keys, _ = self.model.load_state_dict(base_state, strict=False)
                    if missing_keys and len(missing_keys) > 0:
                        print(f"⚠️  缺失 {len(missing_keys)} 个键（scene_embed等新层）")
                    print("✅ Loaded base SSDiff weights")
                
                # 添加LoRA层
                if 'lora_target_modules' in state_dict:
                    # 从checkpoint中读取lora_rank
                    if 'lora_rank' in state_dict:
                        args.lora_rank = state_dict['lora_rank']
                        print(f"📊 Using lora_rank={args.lora_rank} from checkpoint")
                    else:
                        # 默认值
                        args.lora_rank = 4
                        print(f"⚠️  lora_rank not found in checkpoint, using default: {args.lora_rank}")
                    
                    # 添加LoRA层并加载权重（在基础SSDiff权重之上）
                    self.model, _ = initialize_ssdiff_unet_with_lora(
                        args, pretrained_path=args.pretrained_ssdiff_path
                    )
                    
                    # 加载LoRA权重、ARConv权重和Scene Gate
                    loaded_counts = {'lora': 0, 'arconv': 0, 'scene_gate': 0}
                    
                    for name, param in self.model.named_parameters():
                        if name in state_dict['unet_state_dict']:
                            param.data.copy_(state_dict['unet_state_dict'][name])
                            if 'lora' in name:
                                loaded_counts['lora'] += 1
                            elif 'scene_gate' in name:
                                loaded_counts['scene_gate'] += 1
                            elif any(key in name.lower() for key in ['arconv', 'adaptive', 'offset', 'modulation', 'p_conv', 'l_conv', 'w_conv', 'm_conv', 'b_conv', 'convs.']):
                                loaded_counts['arconv'] += 1
                    
                    print(f"\n✅ 模型加载统计:")
                    print(f"   LoRA参数: {loaded_counts['lora']}")
                    print(f"   ARConv参数: {loaded_counts['arconv']} {'✅' if loaded_counts['arconv'] > 0 else '⚠️ 未找到！'}")
                    print(f"   Scene Gate参数: {loaded_counts['scene_gate']}")
                    
                    if args.use_arconv and loaded_counts['arconv'] == 0:
                        print(f"⚠️  警告：模型使用ARConv但checkpoint中没有ARConv权重")
                        print(f"    将使用插值初始化（从Conv2d权重扩展）")
                
                # 🔥 加载Scene Extractor权重（如果存在）
                if 'scene_extractor_state_dict' in state_dict and state_dict['scene_extractor_state_dict'] is not None:
                    if self.scene_extractor is not None:
                        self.scene_extractor.load_state_dict(state_dict['scene_extractor_state_dict'])
                        print("✅ Loaded Scene Extractor weights")
                    else:
                        print("⚠️  Checkpoint has Scene Extractor weights but model doesn't use scene_token")
                elif self.scene_extractor is not None:
                    print("⚠️  Model uses scene_token but checkpoint doesn't have Scene Extractor weights (will use random init)")
            else:
                # 原始SSDiff模型
                # 使用strict=False，因为我们添加了scene_embed层
                missing_keys, _ = self.model.load_state_dict(state_dict, strict=False)
                if missing_keys and len(missing_keys) > 0:
                    print(f"⚠️  缺失 {len(missing_keys)} 个键（scene_embed等新层）")
                print("✅ Loaded original SSDiff weights")
        
        # 🔥 简化模型加载，避免复杂的权重类型设置
        self.model.to(self.device)
        
        # 设置 epoch > 1000，确保 ARConv 使用训练好的固定卷积核（reserved_NXY）
        self.model.set_epoch(10001)
        print("✅ Set epoch to 10001 for using fixed ARConv kernel size")
        
        self.model.eval()
    
    @torch.no_grad()
    def forward(self, lms, pan, ms):
        """
        推理前向传播
        
        Args:
            lms: 低分辨率多光谱 [B, 8, H, W]
            pan: 全色图像 [B, 1, H, W]  
            ms: 上采样多光谱 [B, 8, H, W]
        
        Returns:
            output: 锐化后的多光谱图像 [B, 8, H, W]
        """
        # 🔥 确保推理模式
        self.model.training = False
        self.model.eval()
        
        # 数据类型转换（去掉混合精度）
        lms = lms.to(self.device)
        pan = pan.to(self.device)
        ms = ms.to(self.device)
        
        model_kwargs = {"lms": lms, "pan": pan, "ms": ms}
        
        if self.args.use_distillation:
            # 单步蒸馏模式 - 直接调用forward_impl，不使用采样过程
            # 与训练时的逻辑完全一致：直接前向传播，无需ddim_sample
            
            # 使用forward_chop处理大图像
            # 需要定义一个简单的包装函数，不使用采样
            
            # 用于记录第一个patch的调试信息
            first_patch = [True]
            
            def direct_forward_fn(model_output, t, lms_input, **kwargs):
                # 训练时直接使用forward_impl的输出作为残差
                # model_output是forward_impl的输出（残差预测）
                # lms_input是切分后的patch（64x64）
                
                # 打印调试信息（仅第一次）
               
                # 如果predict_xstart=True，model_output就是预测的x0（残差）
                # 否则需要从eps转换为x0
                if self.args.predict_xstart:
                    pred_xstart = model_output
                else:
                    # 从kwargs稳健获取当前patch对应的x_t；若无则用0
                    x_t = kwargs.get('noise', None)
                    if x_t is None:
                        x_t = torch.zeros_like(lms_input)
                    
                    # 确保x_t与lms_input形状匹配
                    if x_t.shape != lms_input.shape:
                        print(f"⚠️ 形状不匹配! x_t: {x_t.shape}, lms: {lms_input.shape}")
                        # 重新创建与当前patch匹配的x_t
                        x_t = torch.zeros_like(lms_input)
                    
                    # 构造与当前patch批大小匹配的t_batch（优先使用回调传入的t）
                    if isinstance(t, torch.Tensor):
                        if t.dim() == 0:
                            t_batch = t.to(device=lms_input.device, dtype=torch.long).expand(lms_input.shape[0])
                        else:
                            t_batch = t.to(device=lms_input.device, dtype=torch.long)
                    else:
                        t_batch = torch.full((lms_input.shape[0],), 999, device=lms_input.device, dtype=torch.long)
                    pred_xstart = self.diffusion._predict_xstart_from_eps(x_t, t_batch, model_output)
                
                # 最终输出 = lms_patch + 残差
                output = lms_input + pred_xstart
                
                # 打印统计信息
                print(f"   LMS patch: [{lms_input.min():.4f}, {lms_input.max():.4f}], mean={lms_input.mean():.4f}")
                print(f"   Model output (residual): [{model_output.min():.4f}, {model_output.max():.4f}], mean={model_output.mean():.4f}")
                print(f"   Predicted residual: [{pred_xstart.min():.4f}, {pred_xstart.max():.4f}], mean={pred_xstart.mean():.4f}")
                print(f"   Output before clamp: [{output.min():.4f}, {output.max():.4f}], mean={output.mean():.4f}")
                
                return {"sample": output, "pred_xstart": pred_xstart}
            
            # 与训练/推理逻辑保持一致：单步蒸馏推理时使用 x_t = 0
            xt = torch.zeros_like(lms)
            
            # 再次确认测试时predict_xstart设置
            print(f"🔍 [SSDiff_test 单步蒸馏推理] predict_xstart = {self.args.predict_xstart}")
            
            # 可选：添加对比实验 - 使用带噪声的x_t（与训练更接近）
            if hasattr(self.args, 'test_with_noise') and self.args.test_with_noise:
                print("🔬 使用带噪声的x_t进行测试（更接近训练分布）")
                # 对整张图生成一致的噪声，避免patch边界问题
                noise = torch.randn_like(lms)
                # 时间步固定t=999
                t_full = torch.full((lms.shape[0],), 99, device=self.device, dtype=torch.long)
                # 对零残差加噪
                xt = self.diffusion.q_sample_xt(torch.zeros_like(lms), t_full, noise=noise)
                print(f"   带噪x_t范围: [{xt.min():.4f}, {xt.max():.4f}], 均值={xt.mean():.4f}, 标准差={xt.std():.4f}")
                        
            # 🔥 提取scene token（如果启用）
            scene_token = None
            if self.scene_extractor is not None:
                with torch.no_grad():
                    scene_token = self.scene_extractor(pan)
                    print(f"✅ Scene token提取完成: {scene_token.shape}")
            
            # 🔥 统一使用 forward_chop 处理（支持大图像patch切分）
            batch_size = lms.shape[0]
            timesteps = torch.full((batch_size,), 99, device=self.device, dtype=torch.long)
            
            print(f"使用自定义的 forward_chop_distill 进行单步蒸馏推理")
            
            # 🔥 使用我们在 SSNet.py 中新添加的 forward_chop_distill 方法
            # 这个方法直接处理输入，不需要经过复杂的 module.py 逻辑
            output = self.model.forward_chop_distill(
                lms, pan, ms, xt,
                sample_fn=direct_forward_fn,
                scene_token=scene_token,
                noise=xt,
            )
        else:
            # 原始多步采样模式
            sample_fn = (
                self.diffusion.p_sample_loop 
                if not self.args.use_ddim 
                else self.diffusion.ddim_sample_loop
            )
            
            output = sample_fn(
                self.model,
                shape=ms.shape,
                model_kwargs=model_kwargs,
                clip_denoised=self.args.clip_denoised,
                progress=False
            )
        
        # 裁剪到有效范围
        output = output.clamp(0, 1)
        
        return output