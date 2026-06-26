# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Enformer 模型的 TensorFlow 实现。

论文标题: "Effective gene expression prediction from sequence by integrating
long-range interactions"（通过整合长程相互作用从序列中有效预测基因表达）

==============================================================================
Enformer 完整架构（对应论文 Methods 和 Extended Data Fig. 1）:
==============================================================================

输入: 独热编码 DNA 序列，长度 196,608 bp (SEQUENCE_LENGTH)
      形状: [B, 196608, 4]  (A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1])

架构分为三大部分:

1. STEM（茎干）— 初始卷积和池化:
     Conv1D(15) → Residual(Conv1D(1)) → Attention Pooling(stride=2)
     输出: [B, 98304, C/2]

2. CONV_TOWER（卷积塔）— 6 个卷积块，每个块包含:
     ConvBlock(5) → Residual(Conv1D(1)) → Attention Pooling(stride=2)
     通道数从 C/2 指数增长到 C
     每块 stride=2 使空间维度减半: 98304→49152→24576→12288→6144→3072→1536
     输出: [B, 1536, C]  (每个位置代表 128 bp)

3. TRANSFORMER（Transformer 塔）— 11 个 Transformer 块:
     每个块包含 MHA 子层 + MLP 子层（带残差连接）
     输出: [B, 1536, C]

4. CROP + FINAL_POINTWISE（裁剪 + 最终点卷积）:
     TargetLengthCrop1D: 两端各裁剪 320 个位置 → [B, 896, C]
     最终 ConvBlock(1) + Dropout + GELU

5. HEADS（物种特异性预测头）:
     人类头: Linear(5313) + Softplus → [B, 896, 5313]
     小鼠头: Linear(1643) + Softplus → [B, 896, 1643]

==============================================================================
训练配置（对应论文 "Model training and evaluation" 部分）:
==============================================================================
- 损失函数: Poisson 负对数似然 (Poisson NLL)
    L = -Σ_i [y_i · log(ŷ_i) - ŷ_i]
    其中 ŷ_i = softplus(线性层输出)，保证预测值为正
- 优化器: Adam (lr=0.0005, β1=0.9, β2=0.999, ε=1×10⁻⁸)
- 学习率调度: 前 5,000 步从 0 线性预热 (linear warmup)
- 梯度裁剪: 全局范数阈值 0.2
    grads = tf.clip_by_global_norm(grads, clip_norm=0.2)
- 批次大小: 64（每 TPU v3 核心 1 个，共 64 核心同时训练）
- 批次策略: 人类/小鼠交替批次 (alternating batches)
    step % 2 == 0 → human batch, step % 2 == 1 → mouse batch
- Batch Norm: CrossReplicaBatchNorm, momentum=0.9
- 数据增强: 随机平移 ≤3 bp + 反向互补 (reverse complement)
- 验证: 每 1,000 步评估 CAGE TSS Spearman 相关系数
- 训练步数: 150,000（约 3 天）
- 微调: 仅人类数据, lr=0.0001, 额外 30,000 步
- 测试时增强: 8 个随机增强序列的平均预测
==============================================================================
"""
import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import attention_module
import numpy as np
import sonnet as snt
import tensorflow as tf

# ---- 全局常量: 定义基因组序列的输入输出维度 ----
# SEQUENCE_LENGTH = 196,608 bp: 输入 DNA 序列的总长度。
# 这是论文相对于 Basenji2 (131,072 bp) 的重要改进之一:
# 1.5 倍更长的输入序列使模型能看到更远的调控元件。
SEQUENCE_LENGTH = 196_608
# BIN_SIZE = 128 bp: 每个预测位置对应的 DNA 碱基对数。
# 经过 7 次 stride=2 的池化后 (2^7 = 128)，卷积塔输出的每个位置
# 代表 128 bp 的 DNA 区间。这个分辨率大致对应一个典型的
# 调控元件长度，适合聚合被预测的实验数据。
BIN_SIZE = 128
# TARGET_LENGTH = 896: 输出序列的预测位置数。
# 计算方法: 1536 - 2 × 320 = 896
# 其中 1536 是 Transformer 塔的序列长度，
# 320 是每端裁剪的位置数。
# 对应基因组区间: 896 × 128 bp = 114,688 bp
TARGET_LENGTH = 896


class Enformer(snt.Module):
  """Enformer 主模型，完整的基因表达预测网络。

  该模型将 DNA 序列映射到基因组 track 预测值。架构流程:

  输入 x ∈ R^{B × 196608 × 4} (独热编码 DNA)
    ↓
  STEM: Conv1D(15, C/2) → Residual → AttentionPooling(2)
    输出: [B, 98304, C/2]
    ↓
  CONV_TOWER: 6 × [ConvBlock(5) → Residual → AttentionPooling(2)]
    通道数指数增长: C/2 → ... → C
    输出: [B, 1536, C]  (每个位置 = 128 bp)
    ↓
  TRANSFORMER: 11 × [MHA(8个头) + MLP(C→2C→C)]
    输出: [B, 1536, C]
    ↓
  CROP: 两端各去掉 320 个位置
    输出: [B, 896, C]
    ↓
  FINAL_POINTWISE: Conv1D(1, 2C) → Dropout → GELU
    输出: [B, 896, 2C]
    ↓
  HEADS: 两个物种特异性头
    人类: Linear(5313) + Softplus → [B, 896, 5313]
    小鼠: Linear(1643) + Softplus → [B, 896, 1643]

  其中 Softplus(x) = log(1 + exp(x)) 确保输出为正，
  以匹配 Poisson NLL 损失的要求（预测值必须 > 0）。
  """

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               pooling_type: str = 'attention',
               name: str = 'enformer'):
    """初始化 Enformer 模型。

    Args:
      channels: 卷积滤波器数量，也是模型的整体"宽度"C。
                默认 1536 = value_size(192) × num_heads(8)。
                这个参数决定了模型的容量:
                - 主模型: C=1536 (value_size=192)
                - 消融实验: C=768  (value_size=96, 通道数减半)
                通道数影响模型参数总量和计算成本。
      num_transformer_layers: Transformer 层数，默认 11。
                              这是论文的核心设计选择之一:
                              使用 Transformer 块代替 Basenji2 的
                              膨胀卷积来捕获长程相互作用。
                              11 层允许模型逐步建立复杂的
                              调控关系层次结构。
      num_heads: 注意力头数 H，默认 8。
                 每个头独立学习不同的注意力模式，
                 例如不同头可能分别关注:
                 - 启动子-增强子相互作用
                 - 绝缘体边界
                 - 局部 motif 聚类
      pooling_type: 池化函数类型。
                    'attention': 注意力池化（论文默认），
                      公式: h_j = Σ_i exp(x_i · w_j) · x_ij / Σ_i exp(x_i · w_j)
                      初始化 w≈2I 使其行为类似 max pooling。
                    'max': 标准最大池化。
                    论文发现注意力池化比最大池化略好。
      name: Sonnet 模块名称。
    """
    super().__init__(name=name)
    # pylint: disable=g-complex-comprehension,g-long-lambda,cell-var-from-loop

    # ---- 物种特异性预测头: 每个物种有不同的输出 track 数 ----
    # 人类: 5,313 tracks
    #   = 2,131 TF ChIP-seq + 1,860 组蛋白修饰 ChIP-seq
    #     + 684 DNase-seq/ATAC-seq + 638 CAGE
    # 小鼠: 1,643 tracks
    #   = 308 TF ChIP-seq + 750 组蛋白修饰 ChIP-seq
    #     + 228 DNase-seq/ATAC-seq + 357 CAGE
    # 详见论文 Supplementary Table 2, 3。
    heads_channels = {'human': 5313, 'mouse': 1643}

    # ---- 全局 dropout 率 ----
    # 论文 Extended Data Fig. 1a 标注为 0.4。
    # 应用于:
    #   - Transformer 块中的 MHA 输出后
    #   - MLP 的两个线性层后
    #   - 最终点卷积后 (dropout_rate / 8 = 0.05)
    # 注意: MHA 内部有独立的 attention_dropout_rate (0.05) 和
    #   positional_dropout_rate (0.01)，不与此混淆。
    dropout_rate = 0.4

    # 确保通道数能被头数整除
    assert channels % num_heads == 0, ('channels needs to be divisible '
                                       f'by {num_heads}')

    # ---- 全局 MHA 参数 (所有 11 个 Transformer 块共享配置) ----
    # 每个块创建独立的 MHA 实例，但参数配置相同。
    whole_attention_kwargs = {
        # 注意力权重 dropout: softmax 之后随机丢弃注意力连接
        'attention_dropout_rate': 0.05,
        # 初始化器: None 表示使用 VarianceScaling(scale=2.0) 即 He 初始化
        'initializer': None,
        # 键/查询大小 K=64。每个头的查询和键维度。
        'key_size': 64,
        # 头数 H=8。每个位置有 8 种不同的注意力模式。
        'num_heads': num_heads,
        # 相对位置特征数 Cr = V = C/H = 192。
        # 论文: 位置特征数 = value_size，使得每个值维度对应一个位置特征。
        'num_relative_position_features': channels // num_heads,
        # 位置编码 dropout: 随机丢弃 1% 的位置编码特征
        'positional_dropout_rate': 0.01,
        # 位置编码基函数: 3 类（指数衰减、中心掩码、Gamma 分布）
        # 详见 attention_module.py 和 Extended Data Fig. 5b。
        'relative_position_functions': [
            'positional_features_exponential',
            'positional_features_central_mask',
            'positional_features_gamma'
        ],
        # 使用 Transformer-XL 风格的相对位置编码
        'relative_positions': True,
        # 缩放点积注意力: q = q / √K
        'scaling': True,
        # 值大小 V = C/H。论文主模型 V=192，消融实验 V=96。
        'value_size': channels // num_heads,
        # 零初始化最终线性层: 使训练初期 MHA 输出≈0，
        # 模型先依赖卷积塔特征，再逐渐学习注意力模式。
        'zero_initialize': True
    }

    # ---- 构建模型 trunk (主干网络) ----
    # 使用 tf.name_scope 确保所有 trunk 层的变量名有统一前缀。
    trunk_name_scope = tf.name_scope('trunk')
    trunk_name_scope.__enter__()

    # ---- 卷积块工厂函数 ----
    # 在 Sequential 中使用 lambda 以在 tf.name_scope 下构建模块。
    # 每个卷积块 = BatchNorm → GELU → Conv1D
    # 这是标准的"预激活"(pre-activation) 卷积块设计，
    # 有助于训练深层网络时的梯度流动。
    def conv_block(filters, width=1, w_init=None, name='conv_block', **kwargs):
      return Sequential(lambda: [
          # CrossReplicaBatchNorm: 跨 TPU 核心同步的批归一化
          # create_scale/create_offset: 学习 γ 和 β 参数
          # scale_init=Ones(): γ 初始化为 1
          # ExponentialMovingAverage(0.9): 推理时使用 0.9 动量的移动平均
          snt.distribute.CrossReplicaBatchNorm(
              create_scale=True,
              create_offset=True,
              scale_init=snt.initializers.Ones(),
              moving_mean=snt.ExponentialMovingAverage(0.9),
              moving_variance=snt.ExponentialMovingAverage(0.9)),
          # GELU 激活函数（而非标准 ReLU）
          # GELU(x) = x · Φ(x) ≈ x · sigmoid(1.702x)
          gelu,
          # 1D 卷积: 宽度为 width，输出 filters 个通道
          # padding='SAME' 保持序列长度不变
          snt.Conv1D(filters, width, w_init=w_init, **kwargs)
      ], name=name)

    # =================================================================
    # STEM（茎干）: 初始特征提取
    # =================================================================
    # 输入: [B, 196608, 4]
    # 步骤:
    #   1. Conv1D(C/2, 15): 宽卷积核 (宽度=15 bp) 捕获初始 motif 模式
    #      196608 bp → 196608 (padding='SAME')
    #      通道: 4 → C/2 = 768
    #      宽度 15 意味着每个卷积核覆盖 15 个碱基对，
    #      足以捕获大多数 TF 结合 motif (~6-20 bp)。
    #   2. Residual(Conv1D(C/2, 1)): 逐点卷积残差块
    #      1×1 卷积不改变空间维度，仅混合通道信息。
    #   3. Attention Pooling(2): 步长 2 的注意力池化
    #      196608 → 98304 (减半)
    # 输出: [B, 98304, C/2]
    stem = Sequential(lambda: [
        snt.Conv1D(channels // 2, 15),
        Residual(conv_block(channels // 2, 1, name='pointwise_conv_block')),
        pooling_module(pooling_type, pool_size=2),
    ], name='stem')

    # =================================================================
    # CONV_TOWER（卷积塔）: 6 个卷积块 + 池化，逐步压缩空间维度
    # =================================================================
    # 通道数从 C/2 指数增长到 C:
    #   块0: 768 → 块1: 期望~896 → ... → 块5: 1536
    #   exponential_linspace_int 在对数空间均匀分布 6 个值。
    #
    # 每个卷积塔块:
    #   1. ConvBlock(5): 宽度为 5 的卷积，捕获局部的 motif 组合模式
    #      padding='SAME' 保持序列长度
    #   2. Residual(Conv1D(1)): 逐点卷积残差，增强梯度流动
    #   3. Attention Pooling(2): 步长 2 池化，空间维度减半
    #
    # 空间维度变化:
    #   98304 → 49152 → 24576 → 12288 → 6144 → 3072 → 1536
    #   共 7 次减半 (stem 1 次 + tower 6 次)
    #   最终每个位置代表 2^7 × BIN_SIZE = 128 bp
    #
    # 输出: [B, 1536, C]
    filter_list = exponential_linspace_int(start=channels // 2, end=channels,
                                           num=6, divisible_by=128)
    conv_tower = Sequential(lambda: [
        Sequential(lambda: [
            conv_block(num_filters, 5),
            Residual(conv_block(num_filters, 1, name='pointwise_conv_block')),
            pooling_module(pooling_type, pool_size=2),
            ],
                   name=f'conv_tower_block_{i}')
        for i, num_filters in enumerate(filter_list)], name='conv_tower')

    # =================================================================
    # TRANSFORMER（Transformer 塔）: 11 个 Transformer 块
    # =================================================================
    # 这是论文与 Basenji2 的核心区别之一:
    #   使用 Transformer 自注意力代替膨胀卷积来捕获长程相互作用。
    #
    # 每个 Transformer 块（论文中此部分对应 11 个堆叠的 attention_module.TransformerBlock）:
    #   1. MHA 子层:
    #      LN → MultiheadAttention(H=8, K=64, V=C/H) → Dropout(0.4) → 残差
    #      - 8 个注意力头，每个头独立学习"关注模式"
    #      - 相对位置编码使模型学习基于距离的交互偏好
    #   2. MLP 子层:
    #      LN → Linear(C→2C) → Dropout(0.4) → ReLU → Linear(2C→C) → Dropout(0.4) → 残差
    #
    # 注意: 这里使用 Sequential + Residual 重构了 attention_module.TransformerBlock，
    #   但功能等价。LN 的 scale_init=Ones() 不同于默认设置。
    #
    # 为什么是 11 层？这是论文通过消融实验确定的最优深度。
    # 更多的层能建模更复杂的调控关系层次，但增加计算成本。
    #
    # 输出: [B, 1536, C]（形状与输入相同）

    # MLP 子层的工厂函数
    def transformer_mlp():
      return Sequential(lambda: [
          snt.LayerNorm(axis=-1, create_scale=True, create_offset=True),
          snt.Linear(channels * 2),       # C → 2C 升维
          snt.Dropout(dropout_rate),      # rate=0.4
          tf.nn.relu,                     # ReLU 非线性激活
          snt.Linear(channels),           # 2C → C 降维
          snt.Dropout(dropout_rate)],     # rate=0.4
          name='mlp')

    # 11 个 Transformer 块堆叠
    transformer = Sequential(lambda: [
        Sequential(lambda: [
            # MHA 子层 (带残差)
            Residual(Sequential(lambda: [
                snt.LayerNorm(axis=-1,
                              create_scale=True, create_offset=True,
                              scale_init=snt.initializers.Ones()),
                attention_module.MultiheadAttention(**whole_attention_kwargs,
                                                    name=f'attention_{i}'),
                snt.Dropout(dropout_rate)], name='mha')),
            # MLP 子层 (带残差)
            Residual(transformer_mlp())], name=f'transformer_block_{i}')
        for i in range(num_transformer_layers)], name='transformer')

    # =================================================================
    # CROP（裁剪层）: 去掉两端边界的预测
    # =================================================================
    # 输入 [B, 1536, C] → 输出 [B, 896, C]
    # 每端裁剪 (1536 - 896) / 2 = 320 个位置。
    #
    # 论文解释 (Methods):
    # "The cropping layer trims 320 positions on each side to avoid computing
    #  the loss on the far ends because these regions are disadvantaged because
    #  they can observe regulatory elements only on one side (toward the
    #  sequence center) and not the other (the region beyond the sequence
    #  boundaries)."
    #
    # 翻译: 裁剪层在每端去掉 320 个位置，以避免在远端计算损失，
    # 因为这些区域只能观察到一侧（朝向序列中心）的调控元件，
    # 而看不到另一侧（超出序列边界的区域）。
    #
    # 这确保了模型在对称信息条件下进行预测，提高预测的可靠性。
    # 裁剪后对应的基因组区间: 896 × 128 bp = 114,688 bp。
    crop_final = TargetLengthCrop1D(TARGET_LENGTH, name='target_input')

    # =================================================================
    # FINAL_POINTWISE（最终点卷积）: Transformer 后的最终特征变换
    # =================================================================
    # Conv1D(1, 2C) → Dropout(0.05) → GELU
    # - 1×1 卷积（逐点）: 对每个位置独立进行通道变换
    # - 2C 通道: 扩张通道数以为预测头提供更多信息
    # - dropout_rate / 8 = 0.4 / 8 = 0.05: 较小的 dropout 率
    # - GELU: 平滑的非线性激活，保持负值的小梯度
    final_pointwise = Sequential(lambda: [
        conv_block(channels * 2, 1),
        snt.Dropout(dropout_rate / 8),
        gelu], name='final_pointwise')

    # ---- 组装主干网络 ----
    # Trunk: 从 DNA 序列到共享特征表示的完整前向路径
    self._trunk = Sequential([stem,
                              conv_tower,
                              transformer,
                              crop_final,
                              final_pointwise],
                             name='trunk')
    trunk_name_scope.__exit__(None, None, None)

    # =================================================================
    # HEADS（物种特异性预测头）
    # =================================================================
    # 两个独立的预测头，将共享的 trunk 特征映射到物种特异性 track 预测:
    # - 人类头: Linear(2C→5313) + Softplus
    # - 小鼠头: Linear(2C→1643) + Softplus
    #
    # Softplus(x) = log(1 + exp(x)):
    #   - 输出恒为正（满足 Poisson NLL 损失要求）
    #   - 比 ReLU 更平滑，对负值有非零梯度
    #   - 可解释为对数空间中的预测 counts
    #
    # 两个头共享 trunk 参数，仅在最后分开预测。
    # 训练时交替使用人类和小鼠批次（alternating batches）:
    #   - 人类 batch: 梯度更新 trunk + human head
    #   - 小鼠 batch: 梯度更新 trunk + mouse head
    # 这使得 trunk 学习到跨物种共享的调控语法。
    with tf.name_scope('heads'):
      self._heads = {
          head: Sequential(
              lambda: [snt.Linear(num_channels), tf.nn.softplus],
              name=f'head_{head}')
          for head, num_channels in heads_channels.items()
      }
    # pylint: enable=g-complex-comprehension,g-long-lambda,cell-var-from-loop

  @property
  def trunk(self):
    """返回主干网络模块。

    Trunk 将独热编码 DNA 序列 [B, 196608, 4] 映射到共享特征表示。
    在多物种训练中，trunk 参数由两个物种的数据共同更新，
    学习跨物种的调控语法。
    """
    return self._trunk

  @property
  def heads(self):
    """返回物种特异性预测头字典。

    包含 'human' 和 'mouse' 两个头，每个头将 trunk 特征
    映射到物种特异性的基因组 track 预测值。
    """
    return self._heads

  def __call__(self, inputs: tf.Tensor,
               is_training: bool) -> Dict[str, tf.Tensor]:
    """Enformer 前向传播。

    完整流程:
      输入 x ∈ R^{B×196608×4}
        → trunk: [B, 196608, 4] → [B, 896, 2C]
        → human head: [B, 896, 2C] → [B, 896, 5313]
        → mouse head: [B, 896, 2C] → [B, 896, 1643]

    Args:
      inputs: 独热编码 DNA 序列 x ∈ R^{B × 196608 × 4}。
              B = batch_size (训练时 B=64，每 TPU 核心 1 个样本)。
              最后一维 = 4 (A, C, G, T 四种碱基)。
              N (未知碱基) 编码为 [0,0,0,0]（中性值）。
      is_training: 是否为训练模式。
                   训练时: 启用所有 dropout、BatchNorm 使用批次统计量。
                   推理时: 关闭 dropout、BatchNorm 使用移动平均统计量。
                   验证时使用测试时增强: 对 8 个随机增强序列取平均。

    Returns:
      Dict[str, tf.Tensor]: 包含两个键的字典:
        'human': [B, 896, 5313] — 人类基因组 track 预测
        'mouse': [B, 896, 1643] — 小鼠基因组 track 预测

      每个 track 预测的是对应 128-bp bin 中实验 reads 的 counts。
      对于 CAGE track，后续可通过在 TSS 附近 3 个 bin 求和来获得
      基因表达预测值（详见论文 "Model training and evaluation" 部分）。

    训练时的损失计算 (在外部训练脚本中):
      loss_human = PoissonNLL(human_outputs, human_targets)
      loss_mouse = PoissonNLL(mouse_outputs, mouse_targets)
      total_loss = loss_human if human_batch else loss_mouse
      梯度通过整个网络反向传播（trunk + 对应物种的头）。
    """
    # Step 1: 通过主干网络提取共享特征
    # trunk 包含: stem → conv_tower → transformer → crop → final_pointwise
    trunk_embedding = self.trunk(inputs, is_training=is_training)

    # Step 2: 通过物种特异性头预测基因组 track
    # 两个头都计算，但在训练时只对当前 batch 物种的损失进行反向传播
    return {
        head: head_module(trunk_embedding, is_training=is_training)
        for head, head_module in self.heads.items()
    }

  @tf.function(input_signature=[
      tf.TensorSpec([None, SEQUENCE_LENGTH, 4], tf.float32)])
  def predict_on_batch(self, x):
    """用于 SavedModel 导出的推理方法。

    使用 @tf.function 编译为静态图，加速推理。
    input_signature 指定了输入的形状和类型:
      - [None, 196608, 4]: batch 维度可变，序列长度固定 196608，
        4 个碱基通道。

    用于 TensorFlow Serving 部署时的推理接口。
    推理时 is_training=False:
      - 关闭所有 dropout
      - BatchNorm 使用训练期间积累的移动平均统计量

    Args:
      x: 独热编码 DNA 序列 [B, 196608, 4]。

    Returns:
      与 __call__ 相同的字典输出。
    """
    return self(x, is_training=False)


class TargetLengthCrop1D(snt.Module):
  """裁剪序列以匹配期望的目标长度。

  论文 Methods 中描述的裁剪层:
  "The cropping layer trims 320 positions on each side to avoid computing
   the loss on the far ends..."

  为什么需要裁剪:
  Transformer 的自注意力机制允许每个位置关注整个序列。
  然而，序列两端的位置是不对称的:
  - 中心位置: 可以看到左侧和右侧等量的上下文
  - 边界位置: 只能看到一侧的上下文（另一侧超出序列范围）

  这种不对称性会导致边界位置的预测质量较差。
  裁剪掉两端各 320 个位置后，剩余的 896 个位置都具有
  对称的上下文窗口，预测更加可靠。

  具体计算:
    输入长度 = 1536 (Transformer 塔输出)
    目标长度 = 896
    裁剪量 = (1536 - 896) / 2 = 320 (每端)
  """

  def __init__(self,
               target_length: Optional[int],
               name: str = 'target_length_crop'):
    """初始化裁剪层。

    Args:
      target_length: 目标输出长度。论文中为 896。
                     如果为 None，不进行裁剪（直接返回输入）。
      name: 模块名称。
    """
    super().__init__(name=name)
    self._target_length = target_length

  def __call__(self, inputs):
    """执行对称裁剪。

    Args:
      inputs: [B, L, C] 形状的输入张量。
              L = 1536 (Transformer 塔输出序列长度)。

    Returns:
      [B, TARGET_LENGTH, C] = [B, 896, C] 的裁剪后张量。

    Raises:
      ValueError: 如果输入长度小于目标长度。
    """
    if self._target_length is None:
      return inputs

    # 计算每端需要裁剪的位置数
    # (1536 - 896) // 2 = 320
    trim = (inputs.shape[-2] - self._target_length) // 2

    if trim < 0:
      raise ValueError('inputs longer than target length')
    elif trim == 0:
      return inputs
    else:
      # inputs[..., trim:-trim, :]
      # 从第 320 个位置开始，到倒数第 320 个位置结束
      # 保留中间的 1536 - 2×320 = 896 个位置
      return inputs[..., trim:-trim, :]


class Sequential(snt.Module):
  """扩展的 snt.Sequential，自动将 is_training 传递给接受它的层。

  这是 Sonnet v1 风格 Sequential 的兼容层。
  关键特性:
  - 自动检测每个子模块的 __call__ 签名
  - 如果子模块的 __call__ 接受 is_training 参数，则自动传入
  - 如果子模块不接受（如激活函数），则不传入

  这使得可以将普通函数（如 tf.nn.relu, gelu）和
  Sonnet 模块混合在同一个 Sequential 中。

  使用 lambda 包装层列表以保证在正确的 name_scope 下构建:
    Sequential(lambda: [Layer1(), Layer2()])
  等价于在 name_scope 内顺序构建 Layer1 和 Layer2。
  """

  def __init__(self,
               layers: Optional[Union[Callable[[], Iterable[snt.Module]],
                                      Iterable[Callable[..., Any]]]] = None,
               name: Optional[Text] = None):
    """初始化 Sequential 模块。

    Args:
      layers: 层的列表或返回层列表的可调用对象（lambda）。
              使用 lambda 包装器确保层在正确的 tf.name_scope 中构建，
              这对于变量命名和检查点恢复非常重要。
      name: 模块名称。
    """
    super().__init__(name=name)
    if layers is None:
      self._layers = []
    else:
      # layers 包裹在 lambda 函数中以共享命名空间。
      # 如果传入的是可调用对象（lambda），先调用来获取实际层列表。
      if hasattr(layers, '__call__'):
        layers = layers()
      # 过滤掉 None 层（允许条件性包含/排除层）
      self._layers = [layer for layer in layers if layer is not None]

  def __call__(self, inputs: tf.Tensor, is_training: bool, **kwargs):
    """顺序执行所有层。

    Args:
      inputs: 输入张量 [B, L, C]。
      is_training: 训练模式标志，传递给接受此参数的子模块。
      **kwargs: 传递给子模块的其他参数。

    Returns:
      经过所有层处理后的输出张量。
    """
    outputs = inputs
    for _, mod in enumerate(self._layers):
      # 检查模块的 __call__ 方法是否接受 is_training 参数
      if accepts_is_training(mod):
        outputs = mod(outputs, is_training=is_training, **kwargs)
      else:
        outputs = mod(outputs, **kwargs)
    return outputs


def pooling_module(kind, pool_size):
  """池化模块包装器。

  根据类型字符串选择池化方式:
  - 'attention': 注意力池化（论文默认）
  - 'max': 标准最大池化

  Args:
    kind: 池化类型 ('attention' 或 'max')。
    pool_size: 池化窗口大小，论文中为 2。

  Returns:
    池化模块实例。

  Raises:
    ValueError: 如果池化类型无效。
  """
  if kind == 'attention':
    # 注意力池化: 论文中的关键创新之一
    # per_channel=True: 每个通道有独立的注意力权重
    # w_init_scale=2.0: 初始化为约 2×I，使行为类似 max pooling
    return SoftmaxPooling1D(pool_size=pool_size, per_channel=True,
                            w_init_scale=2.0)
  elif kind == 'max':
    # 标准最大池化: padding='same' 保持序列长度不变
    return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')
  else:
    raise ValueError(f'Invalid pooling kind: {kind}.')


class SoftmaxPooling1D(snt.Module):
  """带可学习权重的注意力池化操作。

  论文 Methods 中的公式:
    给定输入窗口 x ∈ R^{L_p × C}（L_p 个位置，C 个通道），
    对于每个通道 j，输出 h_j ∈ R 通过以下公式计算:

      h_j = Σ_i exp(x_i · w_j) · x_{ij} / Σ_i exp(x_i · w_j)

  其中:
    - i 索引池化窗口内的序列位置
    - w ∈ R^{C×K} 是可学习权重矩阵（per_channel=True 时 K=C,
      否则 K=1，所有通道共享权重）
    - x · w 是输入 x 和权重 w 的点积，决定每个位置的"重要性"

  初始化策略:
    w 初始化为 w_init_scale × I（接近单位矩阵）。
    当 w_init_scale = 2.0 且 per_channel=False 时:
      由于 exp(x·w) ≈ exp(2x)，较大的值被指数放大更多，
      行为近似于 max pooling。
    当 w_init_scale = 0.0 时:
      exp(0) = 1，退化为 average pooling。

  论文提到:
    "We initialize w to 2 × I, where I is the identity matrix to
     prioritize the larger value, making the operation similar to max
     pooling. This initialization gave slightly better performance
     than did random initialization or initialization with zeros,
     representing average pooling."

  翻译: 我们将 w 初始化为 2×I，其中 I 是单位矩阵，以优先考虑
  较大的值，使操作类似于最大池化。这种初始化比随机初始化或
  零初始化（代表平均池化）略好的性能。

  窗口大小 L_p = 2, 步长 = 2（非重叠窗口）。
  这意味着每两个相邻的 128-bp bin 被合并为一个，
  对应 256 bp 的感受野。
  """

  def __init__(self,
               pool_size: int = 2,
               per_channel: bool = False,
               w_init_scale: float = 0.0,
               name: str = 'softmax_pooling'):
    """初始化 Softmax 池化。

    Args:
      pool_size: 池化窗口大小 L_p，与 Max/AvgPooling 中的含义相同。
                 论文中为 2，即每两个连续位置合并为一个。
      per_channel: 如果为 True（论文默认），每个通道独立计算 softmax 权重。
                   这允许不同通道有不同的"关注模式"。
                   例如: 某些通道可能关注最大值（类似 max pooling），
                   其他通道可能关注平均值（类似 avg pooling）。
                   如果为 False，所有通道共享同一组权重。
      w_init_scale: 权重初始化的缩放因子。
                    0.0 → 等效于平均池化（所有权重相等）
                    ~2.0 + per_channel=False → 等效于最大池化
                    论文使用 2.0 和 per_channel=True 以获得最佳效果。
      name: 模块名称。
    """
    super().__init__(name=name)
    self._pool_size = pool_size      # L_p = 2
    self._per_channel = per_channel  # True (论文)
    self._w_init_scale = w_init_scale  # 2.0
    self._logit_linear = None

  @snt.once
  def _initialize(self, num_features):
    """延迟初始化可学习的权重线性层。

    使用 @snt.once 确保在第一次调用时初始化一次。
    初始化发生在第一次 forward 时（而非 __init__），
    因为此时才知道输入通道数。

    权重初始化:
      Identity(w_init_scale): 初始化为缩放的单位矩阵。
      当 w_init_scale=2.0 时:
        权重矩阵 ≈ 2 × I (如果 per_channel=True 且 num_features=输出大小)
        或 ≈ 2×1 向量 (如果 per_channel=False)

    Args:
      num_features: 输入通道数 C。
    """
    self._logit_linear = snt.Linear(
        output_size=num_features if self._per_channel else 1,
        with_bias=False,  # Softmax 对平移不变: exp(x+c)/Σexp(x+c) = exp(x)/Σexp(x)
        w_init=snt.initializers.Identity(self._w_init_scale))

  def __call__(self, inputs):
    """执行注意力池化前向传播。

    维度变换流程:
      输入:  [B, L, C]
      重塑:  [B, L/2, 2, C]  (将 L 个位置按 pool_size=2 分组)
      注意力: softmax(Linear(inputs), axis=-2) 沿窗口维度计算注意力权重
      输出:  [B, L/2, C]  (加权求和)

    Args:
      inputs: [B, L, C] 形状的输入张量。
              L 是当前序列长度（逐步从 196608 减半到 1536）。

    Returns:
      [B, L/2, C] 形状的池化后张量。
    """
    _, length, num_features = inputs.shape

    # 延迟初始化权重层
    self._initialize(num_features)

    # 重塑输入: [B, L, C] → [B, L//2, 2, C]
    # 将连续的 pool_size=2 个位置分组到同一个窗口中
    inputs = tf.reshape(
        inputs,
        (-1, length // self._pool_size, self._pool_size, num_features))

    # 注意力池化核心:
    # 1. self._logit_linear(inputs): 对每个窗口位置计算注意力 logits
    #    形状: [B, L//2, 2, C] (per_channel=True)
    # 2. tf.nn.softmax(..., axis=-2): 沿窗口维度（轴-2）归一化
    #    使每个窗口中所有位置的权重和为 1
    # 3. inputs * weights: 对每个位置的值乘以注意力权重
    # 4. tf.reduce_sum(..., axis=-2): 沿窗口维度求和
    # 等效于公式: h_j = Σ_i a_i · x_{ij}，其中 a_i = exp(x_i·w_j) / Σ_k exp(x_k·w_j)
    return tf.reduce_sum(
        inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
        axis=-2)


class Residual(snt.Module):
  """残差连接块。

  实现标准的残差连接:
    output = inputs + module(inputs)

  这是 ResNet (He et al., 2016) 风格残差连接的核心组件。
  在 Enformer 中有三种使用场景:
  1. 卷积塔中的逐点卷积残差块:
     x → x + ConvBlock(1)(x)
     (帮助梯度跨越卷积块传播)
  2. Transformer 块中的 MHA 残差:
     x → x + MHA(LN(x))
     (帮助梯度跨越注意力子层传播)
  3. Transformer 块中的 MLP 残差:
     x → x + MLP(LN(x))
     (帮助梯度跨越前馈子层传播)

  残差连接的数学意义:
    没有残差: y = F(x)         →  ∂y/∂x = ∂F/∂x
    有残差:   y = x + F(x)     →  ∂y/∂x = 1 + ∂F/∂x

  恒等项 1 确保梯度不会在深层网络中消失（梯度高速公路），
  使得训练 11 层 Transformer + 7 层卷积塔成为可能。
  """

  def __init__(self, module: snt.Module, name='residual'):
    """初始化残差块。

    Args:
      module: 被包装的子模块 F(·)。
      name: 模块名称。
    """
    super().__init__(name=name)
    self._module = module

  def __call__(self, inputs: tf.Tensor, is_training: bool, *args,
               **kwargs) -> tf.Tensor:
    """残差前向传播: output = inputs + F(inputs)。

    要求 F 的输出形状与输入完全一致（因为要做加法）。
    在 Enformer 中，这一要求由 padding='SAME' 卷积和
    通道数匹配的 Linear 层保证。

    Args:
      inputs: 输入张量 x。
      is_training: 训练模式标志。
      *args, **kwargs: 传递给子模块的额外参数。

    Returns:
      x + F(x) 的结果张量。
    """
    return inputs + self._module(inputs, is_training, *args, **kwargs)


def gelu(x: tf.Tensor) -> tf.Tensor:
  """Gaussian Error Linear Unit (GELU) 激活函数。

  原始论文: https://arxiv.org/abs/1606.08415

  GELU 是 ReLU 的平滑替代品，定义为:
    GELU(x) = x · Φ(x)
  其中 Φ(x) 是标准正态分布的累积分布函数 (CDF)。

  此处使用原始论文第 2 节中的近似:
    GELU(x) ≈ x · sigmoid(1.702 · x)

  为什么 Enformer 使用 GELU 而非 ReLU:
  - GELU 是平滑的（处处可导），有利于梯度传播
  - GELU 在负值区域有非零输出和梯度，避免"死亡神经元"
  - GELU 在 Transformer 模型（如 BERT, GPT）中已被验证优于 ReLU

  Args:
    x: 输入张量。

  Returns:
    应用 GELU 激活后的张量。
  """
  # GELU(x) ≈ x · σ(1.702x)，其中 σ 是 sigmoid 函数。
  # 1.702 是一个拟合常数，使近似在 x∈[-3,3] 范围内误差最小。
  return tf.nn.sigmoid(1.702 * x) * x


def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
  """将 DNA 序列字符串独热编码为数值数组。

  编码方案（论文 Methods 中的定义）:
    A → [1, 0, 0, 0]
    C → [0, 1, 0, 0]
    G → [0, 0, 1, 0]
    T → [0, 0, 0, 1]
    N → [0, 0, 0, 0]  (未知碱基编码为零向量)

  N 编码为零向量具有特殊含义:
    在损失计算中，模型对 N 碱基的预测不会影响梯度，
    因为所有核苷酸通道的输入梯度都是 0（无信息）。

  实现方法:
    使用查找表 (hash table) 将 ASCII 编码的字符映射到独热向量，
    这种方法比字符串方法更快，适合处理长序列。

  Args:
    sequence: DNA 序列字符串，如 "ACGTN..."。仅包含 A, C, G, T, N。
    alphabet: 字母表字符串，默认 'ACGT'。
    neutral_alphabet: 中性字母（未知碱基），默认 'N'。
    neutral_value: 中性字母的编码值，默认 0（零向量）。
    dtype: 输出数组的数据类型，默认 np.float32。

  Returns:
    形状为 [len(sequence), len(alphabet)] = [L, 4] 的独热编码数组。
  """
  def to_uint8(string):
    """将字符串转换为 uint8 数组（用于查找表索引）。"""
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)

  # 创建查找表: 256 × 4 (覆盖所有可能的 uint8 值)
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  # 设置 A, C, G, T 对应的行: 单位矩阵的对应行
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  # 设置 N 对应的行: 零向量
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  # 查找: 将输入序列的每个字符映射到对应的独热向量
  return hash_table[to_uint8(sequence)]


def exponential_linspace_int(start, end, num, divisible_by=1):
  """在对数空间中生成指数增长的整数值。

  用于生成卷积塔中各层的通道数，确保通道数在对数空间中
  均匀分布，使每层增加的通道数与当前通道数成比例。

  数学定义:
    生成 num 个值 [a_0, a_1, ..., a_{num-1}] 满足:
    a_0 ≈ start
    a_{num-1} ≈ end
    a_{i+1} / a_i ≈ const  (等比数列)

  具体计算:
    base = exp(log(end/start) / (num-1))
    a_i = round(start × base^i / divisible_by) × divisible_by

  论文中的使用:
    start = C/2 = 768, end = C = 1536, num = 6, divisible_by = 128
    → [768, 864, 960, 1088, 1216, 1408]
    注意: 实际值可能因 round 到 128 的倍数而略有不同。

  Args:
    start: 起始值。
    end: 结束值。
    num: 生成的值的数量。
    divisible_by: 所有值必须是此数的整数倍。

  Returns:
    指数增长的整数值列表，长度为 num。
  """
  def _round(x):
    """将 x 四舍五入到 divisible_by 的最近整数倍。"""
    return int(np.round(x / divisible_by) * divisible_by)

  # 计算底数: 等比数列的公比
  base = np.exp(np.log(end / start) / (num - 1))
  # 生成等比数列并四舍五入
  return [_round(start * base**i) for i in range(num)]


def accepts_is_training(module):
  """检查模块的 __call__ 方法是否接受 is_training 参数。

  用于 Sequential 中决定是否向子模块传递 is_training 参数。
  通过 Python 的内省机制 (inspect.signature) 检查方法签名。

  为什么需要:
  - Sonnet 模块（如 snt.Linear, snt.Dropout）通常接受 is_training 参数
  - TensorFlow 函数（如 tf.nn.relu）不接受 is_training 参数
  - GELU 等自定义函数也不接受 is_training 参数

  如果对不接受 is_training 的模块传入此参数，会抛出 TypeError。

  Args:
    module: 要检查的模块/函数。

  Returns:
    bool: 如果模块的 __call__ 方法接受 is_training 参数则返回 True。
  """
  return 'is_training' in list(inspect.signature(module.__call__).parameters)


# ==============================================================================
# 整体流程总结
# ==============================================================================
#
# 本文件实现了 Enformer 模型的完整架构。以下是数据流和关键设计决策:
#
# 1. 输入编码:
#    DNA 序列 "ACGT..." → one_hot_encode → [B, 196608, 4]
#
# 2. 卷积塔 (STEM + CONV_TOWER):
#    7 次空间降采样 (196608 → 1536)，同时通道数从 4 增长到 1536。
#    每次降采样使用注意力池化 (而非标准最大/平均池化)：
#      h_j = Σ_i softmax(x_i · w_j) · x_{ij}
#    注意力池化允许模型学习"哪些位置的信息更重要"。
#
# 3. Transformer 塔:
#    11 个堆叠的 Transformer 块，使用相对位置编码。
#    3 类位置基函数 (exponential, central_mask, gamma) 共提供 192 个位置特征。
#    每个位置可以关注任意其他位置，实现真正的长程相互作用建模。
#
# 4. 裁剪 + 最终变换:
#    两端各裁剪 320 个位置 (避免边界不对称性)。
#    最终点卷积扩展通道数为 2C。
#
# 5. 双物种预测头:
#    人类 (5313 tracks) 和小鼠 (1643 tracks)。
#    共享 trunk 参数，交替批次训练。
#
# 6. 训练循环 (在外部训练脚本中):
#    for step in range(150000):
#      if step % 2 == 0:
#        batch = human_dataset.next()
#      else:
#        batch = mouse_dataset.next()
#      with tf.GradientTape() as tape:
#        preds = model(batch.inputs, is_training=True)
#        loss = poisson_nll(preds[species], batch.targets)
#      grads = tape.gradient(loss, model.trainable_variables)
#      grads, _ = tf.clip_by_global_norm(grads, 0.2)
#      optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#    其中 poisson_nll(y_true, y_pred) = Σ y_true·log(y_pred) - y_pred
#    学习率: 前 5000 步从 0 → 0.0005，之后保持 0.0005。
# ==============================================================================
