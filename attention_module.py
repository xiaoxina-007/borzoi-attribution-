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
"""TransformerBlock 和 MultiheadAttention 模块，用于 Enformer 论文中。

论文标题: "Effective gene expression prediction from sequence by integrating
long-range interactions"（通过整合长程相互作用从序列中有效预测基因表达）

作者: Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R. Ledsam,
Agnieszka Grabska-Barwinska, Kyle R. Taylor, Yannis Assael, John Jumper,
Pushmeet Kohli, David R. Kelley*

所属机构:
1 DeepMind, London, UK
2 Calico Life Sciences, South San Francisco, CA, USA
3 Google, Tokyo, Japan
4 These authors contributed equally.
* 通讯作者: avsec@google.com, pushmeet@google.com, drk@calicolabs.com

==============================================================================
Enformer 模型架构总览（参见论文 Methods 部分）:
==============================================================================
Enformer 架构由三部分组成：
(1) 7 个带池化的卷积块（convolutional blocks with pooling）
(2) 11 个 Transformer 块（本文件实现的核心组件）
(3) 裁剪层 + 最终点卷积，分支为两个物种特异性网络头

输入: 独热编码的 DNA 序列，长度 196,608 bp
     (A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1], N=[0,0,0,0])
输出: 人类 5,313 条基因组 track，小鼠 1,643 条 track
      每条 track 长度 896，对应 114,688 bp，以 128-bp bin 聚合

卷积塔将空间维度从 196,608 bp 压缩至 1,536（每个序列位置向量代表 128 bp），
然后 Transformer 块（共 11 个，即本文件中的 TransformerBlock）捕获序列中的
长程相互作用（如启动子-增强子相互作用）。
==============================================================================

训练相关说明（参见论文 "Model training and evaluation" 部分）:
- 损失函数: Poisson 负对数似然损失 (Poisson negative log-likelihood loss)
  L = -Σ_i y_i * log(ŷ_i) + ŷ_i  (其中 y_i 为观测 counts, ŷ_i 为预测值)
- 优化器: Adam (β1=0.9, β2=0.999, ε=1×10⁻⁸)
- 学习率: 0.0005，前 5,000 步从 0 线性预热 (linear warmup)
- 梯度裁剪: 全局范数最大 0.2 (gradient global norm clipping to 0.2)
- 批次大小: 64（每个 TPU v3 核心 1 个样本，共 64 核心）
- 人类/小鼠交替批次训练 (alternating batches)
- 数据增强: 随机平移 ≤3 bp + 反向互补 (reverse complement)
- 微调阶段: 仅在人类数据上以学习率 0.0001 训练 30,000 步
- 模型选择: 基于验证集上 CAGE TSS 基因表达 Spearman 相关系数
==============================================================================

Example:
```
mha = MultiheadAttention(
    value_size=96,
    key_size=64,
    num_heads=8,
    relative_position_functions=['positional_features_sin_cos'])
mha(tf.ones((2, 1024, 96*8)), is_training=True)

# 论文中使用的 Transformer 块
transformer_block = TransformerBlock(
    channels=96 * 8,
    dropout_rate=0.4,
    attention_kwargs=dict(
        value_size=96,
        key_size=64,
        num_heads=8,
        relative_positions=True,
        relative_position_symmetric=False,
        num_relative_position_features=None,
        relative_position_functions=['positional_features_exponential',
                                     'positional_features_central_mask',
                                     'positional_features_gamma'],
        positional_dropout_rate=0.01,
        attention_dropout_rate=0.05,
        )
    )
transformer_block(tf.ones((2, 1024, 96*8)), is_training=True)
```
"""
from typing import Any, Dict, List, Optional

import numpy as np
import sonnet as snt
import tensorflow as tf


class TransformerBlock(snt.Module):
  """完整的 Transformer 模块块，Enformer 中使用 11 个此块堆叠。

  该模块实现了带有残差连接的标准 Transformer 架构：
  输入 x → [LayerNorm → MultiheadAttention → Dropout] → 残差加 x → 输出 h_attn
           → [LayerNorm → Linear→ReLU→Linear → Dropout] → 残差加 h_attn → 最终输出

  论文符号对照:
  - 输入 x ∈ R^{B × L × C}，其中 B=batch, L=序列长度(1536), C=通道数(768或1536)
  - MHA 输出: 通过多头注意力聚合全序列信息
  - MLP 输出: 位置独立的前馈变换，通道数先扩张 2 倍再压缩回 C

  对应于论文中的 "11 transformer blocks"，每个块通过自注意力机制
  捕获跨序列的长程相互作用（如启动子-增强子相互作用）。
  """

  def __init__(
      self,
      channels: int,
      dropout_rate: float,
      attention_kwargs: Dict[str, Any],
      name: str = 'transformer_block',
  ):
    """初始化 TransformerBlock。

    Args:
      channels: 输入/输出通道数 C。论文中主模型 C = 768 (value_size=96 × num_heads=8)
                或 C = 1536 (value_size=192 × num_heads=8，即论文主体模型配置)。
                通道数在卷积塔之后确定，每个位置代表 128 bp 的 DNA 序列特征。
      dropout_rate: 全局 dropout 率，论文主模型为 0.4（见 Extended Data Fig. 1a）。
                    应用于 MHA 输出后和 MLP 的两个线性层后。
                    注意：MHA 内部有独立的 attention_dropout_rate (0.05) 和
                    positional_dropout_rate (0.01)。
      attention_kwargs: 传递给 MultiheadAttention 的参数字典。论文配置:
          value_size=192 (主模型) 或 96 (消融实验), key_size=64, num_heads=8,
          relative_positions=True, relative_position_symmetric=False,
          relative_position_functions=['exponential','central_mask','gamma'],
          positional_dropout_rate=0.01, attention_dropout_rate=0.05。
          这些参数控制了多头注意力的行为，详见 MultiheadAttention 类。
      name: 模块名称，默认为 'transformer_block'。
    """
    super().__init__(name=name)
    # ---- 多头注意力子层 (MHA sub-layer) ----
    # LayerNorm: 对最后一个维度（通道维 C）进行归一化。
    # create_scale=True 和 create_offset=True 表示学习可训练的 γ 和 β 参数:
    #   LayerNorm(x) = γ * (x - μ)/σ + β
    # 在 MHA 之前进行归一化 (Pre-LN 架构)，有助于训练稳定性。
    self.mha_ln = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    # MultiheadAttention: 核心注意力模块，详见 MultiheadAttention 类。
    # 输入 x ∈ R^{B×L×C}，输出 h_attn ∈ R^{B×L×C}（形状不变）。
    self.mha = MultiheadAttention(**attention_kwargs)
    # Dropout: 对 MHA 输出进行 dropout，rate=0.4。
    # 仅在 is_training=True 时生效（即训练前向传播时）。
    self.mha_dropout = snt.Dropout(dropout_rate)

    # ---- 前馈网络子层 (MLP/FNN sub-layer) ----
    # LayerNorm: 在 MLP 之前再次归一化。
    self.mlp_ln = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    # 第一个线性层: 将通道数从 C 扩展到 2C。
    # 论文中的标准 Transformer FFN 设计：先升维以增加表示能力。
    # 权重矩阵 W1 ∈ R^{C × 2C}，偏置 b1 ∈ R^{2C}。
    self.mlp_linear1 = snt.Linear(channels * 2)
    # Dropout: 对第一个线性层的输出进行 dropout。
    self.mlp_dropout1 = snt.Dropout(dropout_rate)
    # 第二个线性层: 将通道数从 2C 压缩回 C。
    # 权重矩阵 W2 ∈ R^{2C × C}，偏置 b2 ∈ R^{C}。
    self.mlp_linear2 = snt.Linear(channels)
    # Dropout: 对 MLP 最终输出进行 dropout。
    self.mlp_dropout2 = snt.Dropout(dropout_rate)

  def __call__(self, inputs: tf.Tensor, is_training: bool) -> tf.Tensor:
    """TransformerBlock 的前向传播。

    整体流程（与标准 Transformer 一致）:
    1. MHA 子层: x → LN → MHA → Dropout → + x(残差) → h_attn
    2. MLP 子层: h_attn → LN → Linear(C→2C) → ReLU → Linear(2C→C) → Dropout → + h_attn(残差)

    残差连接 (Residual Connection) 允许梯度直接反向传播，缓解深层网络的
    梯度消失问题。Pre-LN 架构（LayerNorm 在子层之前）比 Post-LN 训练更稳定。

    Args:
      inputs: 输入张量 x ∈ R^{B × L × C}。
              B = batch_size, 论文主模型 B=64（每 TPU 核 1 个）。
              L = 序列长度 = 1536（196,608 bp / 128 bp per position，经 7 个卷积块
                  和下采样后，每个位置代表 128 bp）。
              C = 通道数 = 768 (value_size=96×8) 或 1536 (value_size=192×8)。
              注意：卷积塔最深层（第7个卷积块）输出 C=1536，经 attention pooling
              后直接进入 Transformer 块。
      is_training: 是否为训练模式。
                   训练时: 启用所有 dropout（MHA内部 + Transformer块级别）。
                   推理/验证时: 关闭所有 dropout，输出为确定性结果。
                   验证时使用测试时增强 (test-time augmentation): 对 8 个随机
                   增强序列（≤3bp 平移 + 反向互补）的预测取平均。

    Returns:
      输出张量 ∈ R^{B × L × C}，与输入形状相同。
      经过 11 个 Transformer 块堆叠后，输出进入裁剪层（trim 320 位置），
      再通过物种特异性点卷积头预测基因组 track。
    """
    # ===== MHA 子层 =====
    # Step 1: LayerNorm 归一化
    x = self.mha_ln(inputs)
    # Step 2: 多头注意力。输入 x ∈ R^{B×L×C}，输出同形状。
    #   内部过程（详见 MultiheadAttention.__call__）:
    #     a) 投影生成 q, k, v: q=x·W^q, k=x·W^k, v=x·W^v
    #     b) 计算注意力矩阵: a = softmax(q·k^T / √K + R)  其中 R 为相对位置编码
    #     c) 加权聚合: output = a·v
    #     d) 拼接多头并通过 embedding_layer 线性组合
    x = self.mha(x, is_training=is_training)
    # Step 3: Dropout 正则化（仅训练时）
    x = self.mha_dropout(x, is_training=is_training)
    # Step 4: 残差连接。将 MHA 输出与原始输入相加:
    #   h_attn = x + inputs
    #   残差连接使得梯度可以绕过 MHA 直接回传，缓解梯度消失。
    x += inputs  # 残差连接
    mha_output = x  # 保存 MHA 子层的输出，后续用于 MLP 的残差连接

    # ===== MLP 子层 (前馈网络) =====
    # Step 1: LayerNorm 归一化
    x = self.mlp_ln(mha_output)
    # Step 2: 第一个线性变换 C → 2C。
    #   将每个位置的表示映射到更高维度以增加模型容量。
    x = self.mlp_linear1(x)
    # Step 3: Dropout
    x = self.mlp_dropout1(x, is_training=is_training)
    # Step 4: ReLU 激活函数。引入非线性，使得 MLP 可以学习复杂的特征变换。
    #   ReLU(x) = max(0, x)，对于负值输出 0，对于正值保持恒等。
    x = tf.nn.relu(x)
    # Step 5: 第二个线性变换 2C → C。将表示压缩回原始通道数。
    x = self.mlp_linear2(x)
    # Step 6: Dropout
    x = self.mlp_dropout2(x, is_training=is_training)
    # Step 7: 残差连接。最终输出 = MLP(x) + h_attn
    return x + mha_output


class MultiheadAttention(snt.Module):
  """多头注意力模块，Enformer 中用于建模序列长程相互作用。

  论文公式（Methods 部分）:
    输入序列 x ∈ R^{L × C} (省略 batch 维度)，每个头有独立的可学习权重:
      w^q ∈ R^{C × K}    查询投影矩阵 (query projection)
      w^k ∈ R^{C × K}    键投影矩阵 (key projection)
      w^v ∈ R^{C × V}    值投影矩阵 (value projection)
    其中 K = key_size, V = value_size。

    对每个头:
      q_i = x_i · w^q     查询: 位置 i 的"当前信息"
      k_j = x_j · w^k     键:   位置 j 的"被查找信息"
      v_j = x_j · w^v     值:   位置 j 将向前传播的信息

    注意力矩阵（含相对位置编码 R_{ij}，Transformer-XL 风格）:
      a_{ij} = softmax( q_i · k_j^T / √K + R_{ij} )

    其中 R_{ij} 为相对位置编码:
      R_{ij} = q_i · r_{i-j}^T + u · k_j^T + v · r_{i-j}^T
      r_{i-j} = w^R · f(i-j)  (f 是相对位置基函数)

    单头输出: Σ_j a_{ij} · v_j  （加权聚合所有位置的值）
    最终输出: 拼接所有头的结果并通过线性层组合

    论文主模型配置: 8 个头, value_size=192, key_size=64。
    消融实验配置: value_size=96, key_size=64, 通道数减半。
  """

  def __init__(self,
               value_size: int,
               key_size: int,
               num_heads: int,
               scaling: bool = True,
               attention_dropout_rate: float = 0.1,
               relative_positions: bool = False,
               relative_position_symmetric: bool = False,
               relative_position_functions: Optional[List[str]] = None,
               num_relative_position_features: Optional[int] = None,
               positional_dropout_rate: float = 0.1,
               zero_initialize: bool = True,
               initializer: Optional[snt.initializers.Initializer] = None,
               name: str = None):
    """创建一个 MultiheadAttention 模块。

    Args:
      value_size: 每个头的值嵌入大小 V。
                  论文主模型 V=192（通道数 C = V×H = 1536）。
                  消融实验 V=96（通道数 C = V×H = 768）。
      key_size: 每个头的键和查询嵌入大小 K。
                论文中 K=64。查询和键共享同一维度，二者做点积计算注意力分数。
      num_heads: 每个时间步的独立查询头数 H。
                 论文中 H=8。多头允许模型从不同表示子空间中关注不同位置，
                 类似于为每个位置学习多种不同的"检索策略"。
      scaling: 是否对注意力 logits 进行缩放（即除以 √K）。
               默认为 True。缩放点积注意力 (scaled dot-product attention)
               防止当 K 较大时点积值过大导致 softmax 梯度饱和。
               公式: a = softmax(q·k^T / √K + R)
      attention_dropout_rate: 注意力权重矩阵的 dropout 率。
                              论文中为 0.05。在 softmax 之后应用于注意力权重 a，
                              随机丢弃某些注意力连接，防止过拟合。
                              仅在训练时生效。
      relative_positions: 是否使用 Transformer-XL 风格的相对位置注意力。
                          论文中为 True。相对位置编码 R_{ij} 以参数化方式
                          建模两个位置之间应基于其相对距离相互影响的程度。
                          不使用绝对位置编码，因为基因组建模更关注相对距离。
      relative_position_symmetric: 如果为 True，仅使用基函数的对称版本
                                   f(|x|)；如果为 False（论文设置），同时使用
                                   对称版本 f(|x|) 和反对称版本 sign(x)·f(|x|)
                                   以引入方向性信息。
                                   生物学意义: 增强子对上游/下游启动子的影响
                                   可能不对称，方向性建模很重要。
      relative_position_functions: 用于相对位置偏差的基函数名称列表。
                                   论文使用 3 种基函数类（见 Extended Data Fig. 5b）:
                                   1. 'positional_features_exponential':
                                      f_i(r) = exp(-log(2) · r / r_{1/2,i})
                                      指数衰减，半衰期在对数空间线性分布
                                   2. 'positional_features_central_mask':
                                      f_i(r) = 1 如果 r ≤ 2^i, 否则 0
                                      中心掩码，用于捕获短程相互作用
                                   3. 'positional_features_gamma':
                                      f_i(r) = Gamma(r | α=μ_i²/σ², β=μ_i/σ²)
                                      Gamma 分布，提供灵活的距离衰减曲线
      num_relative_position_features: 相对位置特征的总数。
                                      如果为 None（论文默认），自动设为
                                      value_size = V。
                                      论文中 V=192，因此有 192 个相对位置基函数。
                                      这些基函数在 3 个基函数类之间均分，
                                      每类 64 个特征（32 对称 + 32 反对称）。
      positional_dropout_rate: 位置编码的 dropout 率（如果使用相对位置）。
                               论文中为 0.01。应用于位置编码 f(i-j)，
                               在训练时随机丢弃部分位置特征以正则化。
      zero_initialize: 如果为 True，最终线性层（embedding_layer）将被零初始化。
                       这确保了训练初期注意力模块输出接近零，
                       使得模型最初主要依赖残差路径（即卷积塔的特征），
                       然后逐渐学习注意力变换。有助于训练稳定性。
      initializer: 投影层的初始化器。如果未指定，使用 VarianceScaling (scale=2.0)，
                   即 He 初始化，适合 ReLU 类激活函数。
      name: 模块名称。
    """
    super().__init__(name=name)
    self._value_size = value_size          # V: 每个头的值维度
    self._key_size = key_size              # K: 每个头的查询/键维度
    self._num_heads = num_heads            # H: 头数
    self._attention_dropout_rate = attention_dropout_rate
    self._scaling = scaling
    self._relative_positions = relative_positions
    self._relative_position_symmetric = relative_position_symmetric
    self._relative_position_functions = relative_position_functions
    if num_relative_position_features is None:
      # num_relative_position_features 需要能被
      # 相对位置基函数数量 × 2（对称 + 反对称版本）整除。
      # 论文中: 3 个基函数类 × 2 = 6, V=192, 每个版本得 192/6 = 32 个特征。
      divisible_by = 2 * len(self._relative_position_functions)
      self._num_relative_position_features = (
          (self._value_size // divisible_by) * divisible_by)
    else:
      self._num_relative_position_features = num_relative_position_features
    self._positional_dropout_rate = positional_dropout_rate

    self._initializer = initializer
    if self._initializer is None:
      # He 初始化 (VarianceScaling with scale=2.0): 方差 = scale / fan_in
      # 适合 ReLU 激活，保持前向传播时各层方差稳定。
      self._initializer = snt.initializers.VarianceScaling(scale=2.0)

    # 键/查询投影的总输出维度 = K × H
    # 每个头输出 K 维，共 H 个头
    key_proj_size = self._key_size * self._num_heads
    # 值投影的总输出维度（也是最终嵌入维度）= V × H = C（通道数）
    embedding_size = self._value_size * self._num_heads

    # ---- 查询投影层 w^q (论文符号) ----
    # 将输入 x ∈ R^{L×C} 投影到 q ∈ R^{L×(K·H)}
    # 权重矩阵 W_q ∈ R^{C × (K·H)}，无偏置。
    # 物理含义: "查询"代表每个位置的"当前信息/兴趣"，
    #          表示该位置想要"查找"什么类型的信息。
    self._q_layer = snt.Linear(
        key_proj_size,
        name='q_layer',
        with_bias=False,
        w_init=self._initializer)

    # ---- 键投影层 w^k (论文符号) ----
    # 将输入 x ∈ R^{L×C} 投影到 k ∈ R^{L×(K·H)}
    # 权重矩阵 W_k ∈ R^{C × (K·H)}，无偏置。
    # 物理含义: "键"代表每个位置的"被检索标签"，
    #          表示该位置"具有"什么类型的信息可供其他位置查找。
    self._k_layer = snt.Linear(
        key_proj_size,
        name='k_layer',
        with_bias=False,
        w_init=self._initializer)

    # ---- 值投影层 w^v (论文符号) ----
    # 将输入 x ∈ R^{L×C} 投影到 v ∈ R^{L×(V·H)}
    # 权重矩阵 W_v ∈ R^{C × (V·H)}，无偏置。
    # 物理含义: "值"代表每个位置实际传播的信息内容，
    #          被注意力权重加权后传递给其他位置。
    self._v_layer = snt.Linear(
        embedding_size,
        name='v_layer',
        with_bias=False,
        w_init=self._initializer)

    # ---- 最终线性组合层 (输出投影) ----
    # 将多头拼接后的结果线性映射回 C 维。
    # 权重矩阵 W_o ∈ R^{(V·H) × C} = R^{C × C}。
    # zero_initialize=True 时使用零初始化，使训练初期注意力模块
    # 对残差路径的影响最小。
    w_init = snt.initializers.Zeros() if zero_initialize else self._initializer
    self._embedding_layer = snt.Linear(
        embedding_size,
        name='embedding_layer',
        w_init=w_init)

    # ---- 相对位置编码相关层 (Transformer-XL 风格) ----
    # 如果使用相对位置，创建额外的参数层。
    # 论文中相对位置编码公式:
    #   R_{ij} = q_i · r_{i-j}^T + u · k_j^T + v · r_{i-j}^T
    # 其中:
    #   r_{i-j} = w^R · f(i-j) 是通过 r_k_layer 计算的相对位置嵌入
    #   u (r_w_bias) 是位置无关的键偏好嵌入，用于评估对特定键内容的基础偏好
    #   v (r_r_bias) 是位置无关的相对距离偏好嵌入，用于评估对特定距离的基础偏好
    if self._relative_positions:
      # ---- 相对位置投影层 w^R (论文符号) ----
      # 将位置基函数 f(i-j) 投影到相对键嵌入 r_{i-j}
      # f(i-j) ∈ R^{Cr}（Cr = num_relative_position_features = V = 192）
      # r_{i-j} ∈ R^{K·H}（与 q, k 同样维度）
      self._r_k_layer = snt.Linear(
          key_proj_size,
          name='r_k_layer',
          with_bias=False,
          w_init=self._initializer)

      # ---- 内容相关的位置偏置 u (论文中的 u 向量) ----
      # 形状: [1, H, 1, K]（广播到 [B, H, T', K]）
      # 物理含义: u 是一个全局参数，建模查询对键内容的"基础偏好"，
      #          与具体位置无关。它允许模型学习：无论位置如何，
      #          某些"类型"的键信息天然比其他类型更重要。
      # 论文公式: u · k_j^T 是 u_k^T 项，表示内容相关的偏置。
      self._r_w_bias = tf.Variable(
          self._initializer([1, self._num_heads, 1, self._key_size],
                            dtype=tf.float32),
          name='r_w_bias')

      # ---- 位置相关的位置偏置 v (论文中的 v 向量) ----
      # 形状: [1, H, 1, K]（广播到 [B, H, T', K]）
      # 物理含义: v 是一个全局参数，建模查询对相对距离的"基础偏好"，
      #          与键的内容无关。例如，模型可以学习"通常更关注近处的位置"。
      # 论文公式: v · r_{i-j}^T 是 v_r^T 项，表示位置相关的偏置。
      self._r_r_bias = tf.Variable(
          self._initializer([1, self._num_heads, 1, self._key_size],
                            dtype=tf.float32),
          name='r_r_bias')

  def _multihead_output(self, linear, inputs):
    """对输入应用标准线性变换并返回多头输出。

    将线性变换的输出从 [B, T, H*KV] 重塑为 [B, H, T, KV]，
    使得每个头可以独立处理。

    维度变换流程:
      输入:  [B, T, C]  或 [1, 2T-1, Cr]（位置编码时）
      线性:  [B, T, H·KV]  对所有位置并行应用同一线性层
      reshape: [B, T, H, KV]  将拼接的多头输出拆分为独立头
      transpose: [B, H, T, KV]  将头维度提前以便批量矩阵乘法

    Args:
      linear: snt.Linear 层（q_layer, k_layer, v_layer 或 r_k_layer）。
      inputs: 输入张量。
              对于 q/k/v: 形状 [B, T, C]，来自卷积塔的第 T=1536 个位置。
              对于 r_k: 形状 [1, 2T-1, Cr]，位置编码。

    Returns:
      输出张量形状 [B, H, T, KV]:
        B = batch_size (对于位置编码 r_k，B=1)
        H = num_heads (论文中为 8)
        T = 序列时间步数 (1536, 或 2T-1 对于相对位置)
        KV = key_size (K) 或 value_size (V)，取决于投影类型
    """
    # snt.BatchApply: 对最后 (T) 维度外的所有 batch 维度应用线性层
    # 保持了 T 维度独立处理，即每个位置并行进行相同的线性变换
    output = snt.BatchApply(linear)(inputs)  # [B, T, H * KV]
    num_kv_channels = output.shape[-1] // self._num_heads
    # 将 H * KV 通道拆分为 H 个独立的头
    # reshape 后的形状: [B, T, H, KV]
    output = snt.reshape(output,
                         output_shape=[-1, self._num_heads, num_kv_channels])
    # [B, T, H, KV] -> [B, H, T, KV]
    # 将头维度 H 移到 T 之前，以便后续批量矩阵乘法 [B, H, T, K] × [B, H, K, T]
    return tf.transpose(output, [0, 2, 1, 3])

  def __call__(self,
               inputs,
               is_training=False):
    """MultiheadAttention 的前向传播。

    注意: 这是 Enformer 模型的核心创新之一。传统 NLP Transformer 的 MHA
    直接处理 token 序列；在 Enformer 中，卷积塔先将 DNA 核苷酸嵌入为
    128-bp 分辨率的丰富特征，然后 MHA 在此特征空间上捕获长程相互作用。

    前向传播步骤（与论文 Methods 公式对应）:

    1. 投影生成 q, k, v:
       q = x · w^q,  k = x · w^k,  v = x · w^v

    2. 缩放查询（可选）:
       q = q / √K    (scaled dot-product attention)

    3. 如果使用相对位置编码（论文默认 True）:
       3a. 计算位置编码: r_{i-j} = w^R · f(i-j)
       3b. 内容注意力 logits: q · k^T + u · k^T  （通过 _r_w_bias）
       3c. 相对位置 logits: q · r_{i-j}^T + v · r_{i-j}^T  （通过 _r_r_bias）
       3d. 对相对 logits 应用 relative_shift 对齐维度
       3e. 总 logits = 内容 + 相对位置

       如果不使用相对位置:
       logits = q · k^T

    4. 注意力权重: a = softmax(logits)  （沿最后一维）

    5. Dropout 应用于注意力权重（仅训练时）

    6. 加权聚合: output = a · v  （每个查询位置对所有值位置的加权和）

    7. 重塑并通过最终线性层组合多头输出

    Args:
      inputs: 输入 x ∈ R^{B × L × C}。
              B = batch_size（论文主模型 B=64）。
              L = 序列长度 = 1536。
                  在卷积塔中: 196,608 bp → (7次 stride=2 池化) → 1536 个位置。
                  每个位置代表 128 bp 的 DNA 序列区间。
              C = 通道数 = embedding_size = V × H。
                  论文主模型: V=192, H=8, C=1536。
                  消融实验: V=96, H=8, C=768。
                  这里 C 包含卷积塔从原始 DNA 序列中提取的丰富特征，
                  包括 motif 模式、染色质状态等信息。
      is_training: 是否为训练模式。训练时启用 dropout，推理时关闭。
                   在训练循环中:
                   - 人类/小鼠交替批次: 每个 step 仅从一种物种取 batch
                   - forward pass 时 is_training=True
                   - 验证时 is_training=False，使用测试时增强
                     (8个随机增强序列的平均预测)

    Returns:
      输出张量 ∈ R^{B × L × C}，与输入形状相同。
      经过 11 个 Transformer 块后，输出进入裁剪层和物种特异性预测头。
    """
    # 初始化投影层。embedding_size = V × H = 总通道数 C。
    embedding_size = self._value_size * self._num_heads
    # 序列长度 L = 1536。
    seq_len = inputs.shape[1]

    # ===== 步骤 1: 计算 q, k, v 作为输入的多头投影 =====
    # 每个投影对所有位置并行进行相同的线性变换
    # q: [B, H, T, K] — 查询: 每个位置 i 想要查找什么信息
    #    q_i 表示位置 i 的"查询内容"
    q = self._multihead_output(self._q_layer, inputs)  # [B, H, T, K]
    # k: [B, H, T, K] — 键: 每个位置 j 具有什么信息可供查找
    #    k_j 表示位置 j 的"被匹配标签"
    k = self._multihead_output(self._k_layer, inputs)  # [B, H, T, K]
    # v: [B, H, T, V] — 值: 每个位置 j 将传播什么信息
    #    v_j 表示位置 j 的实际信息内容
    v = self._multihead_output(self._v_layer, inputs)  # [B, H, T, V]

    # ===== 步骤 2: 缩放点积注意力 =====
    # 将查询除以 √K 以防止点积值过大。
    # 原理: 当 q 和 k 的各分量是独立随机变量（均值 0，方差 1）时，
    # q·k^T 的方差为 K，除以 √K 使方差归一化为 1，
    # 防止 softmax 进入饱和区（梯度过小）。
    if self._scaling:
      q *= self._key_size**-0.5  # 等价于 q = q / √K

    # ===== 步骤 3: 计算注意力 logits =====
    if self._relative_positions:
      # ---- 步骤 3a: 计算相对位置编码 ----
      # distances = [-1535, -1534, ..., 0, ..., 1534, 1535]
      # 共 2T-1 = 3071 个不同相对距离值。
      # 这是序列中任意两个位置之间可能的相对距离范围。
      distances = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
      # positional_encodings: [1, 2T-1, Cr]
      # 其中 Cr = num_relative_position_features = V = 192
      # 使用 3 类基函数对每个相对距离值编码:
      #   1) exponential: 指数衰减（长程相互作用建模）
      #   2) central_mask: 中心掩码（短程相互作用建模）
      #   3) gamma: Gamma 分布（灵活的距离衰减曲线）
      # 每类基函数产生 32 个对称 + 32 个反对称 = 64 个特征
      # 3 类 × 64 = 192 = Cr = V
      # 这些基函数共同构成对任意相对距离的丰富表示。
      positional_encodings = positional_features_all(
          positions=distances,
          feature_size=self._num_relative_position_features,
          seq_length=seq_len,
          feature_functions=self._relative_position_functions,
          symmetric=self._relative_position_symmetric)
      # [1, 2T-1, Cr]

      # 位置编码 dropout（仅训练时，rate=0.01）
      # 在训练时随机丢弃 1% 的位置特征以正则化。
      if is_training:
        positional_encodings = tf.nn.dropout(
            positional_encodings, rate=self._positional_dropout_rate)

      # ---- 步骤 3b: 将位置编码投影到相对键空间 ----
      # r_k: [1, H, 2T-1, K]
      # 计算 r_{i-j} = w^R · f(i-j)，即位置基函数的线性投影。
      # 这里 B=1 因为位置编码对所有 batch 样本是共享的。
      r_k = self._multihead_output(self._r_k_layer, positional_encodings)

      # ---- 步骤 3c: 计算内容注意力 logits ----
      # 论文公式: q_i · k_j^T + u · k_j^T
      # 在代码中通过将 q + u (r_w_bias) 与 k 做矩阵乘法实现。
      # q + u: 查询加上全局的内容偏好偏置（u 对每个头、每个 key 维度
      #        提供独立的基础偏好值，广播到所有位置和 batch）
      # [B, H, T', T] — 每个查询位置 T' 对所有键位置 T 的内容注意力分数
      content_logits = tf.matmul(q + self._r_w_bias, k, transpose_b=True)

      # ---- 步骤 3d: 计算相对位置注意力 logits ----
      # 论文公式: q_i · r_{i-j}^T + v · r_{i-j}^T
      # 在代码中通过将 q + v (r_r_bias) 与 r_k 做矩阵乘法实现。
      # 矩阵乘法结果: [B, H, T', 2T-1]
      # 每一列对应一个相对距离 d = j - i (从 -(T-1) 到 T-1)
      # 例如列 T-1 对应相对距离 0（自己对自己）
      relative_logits = tf.matmul(
          q + self._r_r_bias, r_k, transpose_b=True)

      # ---- 步骤 3e: 对齐相对 logits 维度 ----
      # relative_shift 函数将 [B, H, T', 2T-1] 转换为 [B, H, T', T]
      # 使得位置 (i,j) 的 logit 对应正确的相对距离 i-j。
      # 具体变换逻辑见 relative_shift 函数注释。
      #  [B, H, T', T]
      relative_logits = relative_shift(relative_logits)

      # ---- 步骤 3f: 合并内容和位置注意力的 logits ----
      # 总 logits = 内容注意力 + 相对位置注意力
      # R_{ij} = q_i·k_j^T/√K + q_i·r_{i-j}^T + u·k_j^T + v·r_{i-j}^T
      # 其中 q_i·k_j^T + u·k_j^T 来自 content_logits
      #      q_i·r_{i-j}^T + v·r_{i-j}^T 来自 relative_logits
      # 两种注意力的和使得模型既能基于序列内容（如 motif 匹配）做注意力，
      # 也能基于纯距离（如"增强子通常影响 10-100kb 内的启动子"）做注意力。
      logits = content_logits + relative_logits
    else:
      # 不使用相对位置时的标准缩放点积注意力
      # [B, H, T', T] — q 和 k 的点积，每行是一个查询对所有键的注意力分数
      logits = tf.matmul(q, k, transpose_b=True)

    # ===== 步骤 4: Softmax 归一化得到注意力权重 =====
    # a_{ij} = softmax(logits_{ij}) = exp(logits_{ij}) / Σ_j exp(logits_{ij})
    # 对每个查询位置 i 的所有键位置 j 做归一化。
    # 含义: 注意力权重 a_{ij} 表示查询位置 i 将多少"注意力"分配给键位置 j，
    #       所有权重之和为 1（概率分布）。
    # 在基因组学语境中: a_{ij} 量化了位置 i（如启动子）对位置 j（如增强子）
    #                   的依赖程度。
    weights = tf.nn.softmax(logits)

    # ===== 步骤 5: 注意力权重 dropout =====
    # 仅训练时启用，rate=0.05。
    # 随机将某些注意力连接置零，强制模型学习冗余的注意力模式，
    # 防止过度依赖少数几个位置对。
    # 这是标准 Transformer 训练中的正则化技术。
    if is_training:
      weights = tf.nn.dropout(weights, rate=self._attention_dropout_rate)

    # ===== 步骤 6: 加权聚合值向量 =====
    # output = a · v 即 Σ_j a_{ij} · v_j
    # [B, H, T', V] — 每个查询位置 i 对所有值 v_j 的加权和。
    # 物理含义: 位置 i 根据其注意力权重 a_{ij}，从所有位置 j 聚合信息。
    #   例如: 启动子位置可能主要从远处的增强子位置聚合信息，
    #         而从无关位置聚合很少的信息（注意力权重接近 0）。
    output = tf.matmul(weights, v)  # [B, H, T', V]

    # ===== 步骤 7: 转置和重塑为最终输出格式 =====
    # [B, H, T', V] → [B, T', H, V]
    # 将头维度移回时间维度之后，为拼接做准备。
    output_transpose = tf.transpose(output, [0, 2, 1, 3])  # [B, T', H, V]

    # ---- 步骤 8: 最终线性层组合多头输出 ----
    # 先将 [B, T', H, V] 重塑为 [B, T', H·V] = [B, T', C]
    # preserve_dims=2 表示保留前 2 个维度（B 和 T'）不变。
    # 然后通过 embedding_layer 线性变换组合各头输出。
    # 输出的形状恢复为 [B, T', C]，与输入形状相同。
    # 注意: 论文中此步骤拼接各头输出并通过线性层 w^o 进行组合:
    #       最终输出 = concat(head_1, ..., head_H) · w^o
    attended_inputs = snt.reshape(
        output_transpose, output_shape=[embedding_size], preserve_dims=2)
    output = self._embedding_layer(attended_inputs)

    return output


def relative_shift(x):
  """对相对 logits 进行位移，如 Transformer-XL 中的实现。

  该函数将相对注意力 logits 从 [B, H, T, 2T-1] 格式（按相对距离索引）
  转换为 [B, H, T, T] 格式（按绝对位置索引），使得位置 (i, j) 的 logit
  对应正确的相对距离 i - j。

  输入维度含义:
    x ∈ [B, H, T, 2T-1]
    - 最后一维大小 = 2T-1，索引 d = 0, 1, ..., 2T-2
    - 索引 d 对应相对距离 d - (T-1)，即范围 [-(T-1), T-1]
    - 例如: d = T-1 对应相对距离 0（自己对自己）
    - 例如: d = 0 对应相对距离 -(T-1)（最远的前向距离）
    - 例如: d = 2T-2 对应相对距离 T-1（最远的后向距离）

  转换过程（用 T=3 举例）:
    输入 (行=查询i, 列=相对距离d):
      d=0     d=1     d=2     d=3     d=4
      r_{-2}  r_{-1}  r_{0}   r_{1}   r_{2}    ← 查询位置 0
      r_{-2}  r_{-1}  r_{0}   r_{1}   r_{2}    ← 查询位置 1
      r_{-2}  r_{-1}  r_{0}   r_{1}   r_{2}    ← 查询位置 2

    期望输出 (行=查询i, 列=键j):
      j=0       j=1       j=2
      r_{0-0}  r_{1-0}  r_{2-0}    →  r_0   r_1   r_2    ← 查询位置 0
      r_{0-1}  r_{1-1}  r_{2-1}    →  r_{-1} r_0   r_1   ← 查询位置 1
      r_{0-2}  r_{1-2}  r_{2-2}    →  r_{-2} r_{-1} r_0   ← 查询位置 2

  实现方法:
    1. 在最后一维左侧填充一列零（避免索引越界）
    2. 交换 T 和 2T 维度
    3. 切片取出正确的相对距离值
    4. 重塑并截断到目标形状

  Args:
    x: [B, H, T, 2T-1] 形状的相对 logits 张量。

  Returns:
    [B, H, T, T] 形状的位移后 logits 张量。
  """
  # Step 1: 在最后一维（相对距离维）左侧填充一列零
  # 填充后的形状: [B, H, T, 2T]
  # 填充零的原因是: 在重塑和切片时，某些位置需要填充值
  to_pad = tf.zeros_like(x[..., :1])
  x = tf.concat([to_pad, x], -1)
  # 获取张量形状: [B, H, T, 2T]
  _, num_heads, t1, t2 = x.shape

  # Step 2: 重塑 [B, H, T, 2T] → [B, H, 2T, T]
  # 交换最后两维以便沿正确的轴进行切片
  x = tf.reshape(x, [-1, num_heads, t2, t1])

  # Step 3: 切片，从第 1 行开始（跳过第 0 行）
  # [B, H, 2T, T] → [B, H, 2T-1, T]
  # 去掉第一行，因为它对应着被填充的零
  x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])

  # Step 4: 重塑回 [B, H, T, 2T-1]
  x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])

  # Step 5: 截断到 [B, H, T, T]
  # 只保留前 T 列（即只保留有意义的相对距离范围）
  # (t2 + 1) // 2 = (2T + 1) // 2 = T（整数除法）
  x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
  return x


# 可用的位置特征函数注册表:
def get_positional_feature_function(name):
  """返回位置特征函数。

  这是一个函数注册表，根据名称字符串返回对应的位置编码基函数。
  论文中使用了前三种（exponential, central_mask, gamma），
  后三种（cosine, linear_masks, sin_cos）是额外的备选方案，
  可用于消融实验或替代配置。

  Args:
    name: 函数名称字符串。

  Returns:
    对应的位置特征函数。

  Raises:
    ValueError: 如果名称不在可用函数列表中。
  """
  available = {
      'positional_features_exponential': positional_features_exponential,
      'positional_features_central_mask': positional_features_central_mask,
      'positional_features_gamma': positional_features_gamma,
      'positional_features_cosine': positional_features_cosine,
      'positional_features_linear_masks': positional_features_linear_masks,
      'positional_features_sin_cos': positional_features_sin_cos,
  }
  if name not in available:
    raise ValueError(f'Function {name} not available in {available.keys()}')
  return available[name]


def positional_features_all(positions: tf.Tensor,
                            feature_size: int,
                            seq_length: Optional[int] = None,
                            bin_size: Optional[int] = None,
                            feature_functions: Optional[List[str]] = None,
                            symmetric=False):
  """计算相对位置编码/特征 f(i-j)（论文中的符号）。

  每个位置特征函数将计算/提供相同比例的特征，总计 feature_size 个特征。
  这与论文 Methods 中的描述完全一致:

  论文原文:
    "We use three different basis function classes for f(i-j)"
    "For each basis function, we use a symmetric f(|x|) and asymmetric
     sign(x) × f(|x|) version to introduce directionality."

  论文配置:
    - 总计 Cr = feature_size = V = 192 个位置特征
    - 3 个基函数类 × 2 (对称+反对称) = 6 个分量
    - 每个分量获得 192 / 6 = 32 个特征

  Args:
    positions: 任意形状的相对位置张量。
               通常为 tf.range(-seq_len+1, seq_len) 即 [-1535, 1535]。
               正值表示键在查询的右侧（下游），负值表示左侧（上游）。
    feature_size: 基函数的总数 Cr。论文中 Cr = V = 192。
    seq_length: 序列长度 L = 1536，表示特征长度尺度。
                各个位置特征可以使用此参数来参数化特征，
                使得特征的参数化与 positions 的具体值无关。
    bin_size: 用于划分序列的 bin 大小（以 bp 为单位）。
              可用于在基因组绝对尺度上计算特征。
              在当前实现中未使用（基函数仅依赖于相对距离）。
    feature_functions: 要使用的不同特征函数的列表。
                       每个函数接收参数: positions, sequence length 和
                       要计算的特征数量。
                       论文默认: ['exponential', 'central_mask', 'gamma']。
    symmetric: 如果为 True，结果特征在相对位置 0 两侧对称
              （即只使用位置的绝对值）。
              如果为 False（论文设置），同时使用对称版本和反对称版本
              （对称版本乘以 sign(positions)），以引入方向性。

  Returns:
    形状为 `positions.shape + (feature_size,)` 的张量。
    例如当 positions = [2T-1] 时，返回 [2T-1, Cr]。
  """
  if feature_functions is None:
    # 论文默认的 3 个基函数类
    feature_functions = ['positional_features_exponential',
                         'positional_features_central_mask',
                         'positional_features_gamma']
  # 分量数 = 基函数类数量 × (对称版本或对称+反对称)
  # 论文: 3 类 × 2 (symmetric=False) = 6 个分量
  num_components = len(feature_functions)  # 每个基函数 1 个
  if not symmetric:
    num_components = 2 * num_components  # 对称 + 反对称

  # 目前不允许奇数大小的嵌入（保证整除）
  if feature_size % num_components != 0:
    raise ValueError(
        f'feature_size has to be divisible by {num_components}')

  # 将函数名称解析为实际函数
  feature_functions = [get_positional_feature_function(f)
                       for f in feature_functions]
  # 每个分量获得的基函数数量
  # 论文: 192 / 6 = 32
  num_basis_per_class = feature_size // num_components

  # 对每个基函数类，使用位置的绝对值 |i-j| 计算特征
  # 然后沿最后一维拼接所有特征
  # 结果: [..., feature_size/2]（如果 symmetric=False）
  embeddings = tf.concat([f(tf.abs(positions), num_basis_per_class,
                             seq_length, bin_size)
                           for f in feature_functions],
                          axis=-1)

  if not symmetric:
    # 反对称版本: sign(positions) * f(|positions|)
    # sign(x) = +1 (x>0), 0 (x=0), -1 (x<0)
    # 这引入了方向信息，使模型能区分上游和下游位置
    # sign(positions)[..., tf.newaxis] 广播到与 embeddings 相同形状
    embeddings = tf.concat([embeddings,
                            tf.sign(positions)[..., tf.newaxis] * embeddings],
                           axis=-1)

  # 断言输出形状正确
  tf.TensorShape(embeddings.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return embeddings


def _prepend_dims(x, num_dims):
  """在张量前面添加 num_dims 个大小为 1 的维度，以支持广播。

  Args:
    x: 输入张量。
    num_dims: 要添加的维度数量。通常等于 positions.shape.rank。

  Returns:
    形状为 [1] * num_dims + x.shape 的张量。
  """
  return tf.reshape(x, shape=[1] * num_dims + x.shape)


def positional_features_exponential(positions: tf.Tensor,
                                     feature_size: int,
                                     seq_length: Optional[int] = None,
                                     bin_size: Optional[int] = None,
                                     min_half_life: Optional[float] = 3.0):
  """创建指数衰减的位置权重。

  论文公式:
    f_i^{exponential}(r) = exp(-log(2) · r / r_{1/2,i})

  其中 r = |i-j| 是绝对相对距离，r_{1/2,i} 是第 i 个特征的半衰期
  （半衰期在对数空间中从 min_half_life=3 到 seq_length=1536 线性分布）。

  物理含义: 每个特征在不同距离尺度上衰减，总共有 feature_size 个不同的
  半衰期，覆盖了从极短程（3 个位置 ≈ 384 bp）到全序列长度
  （1536 个位置 ≈ 196,608 bp）的范围。这使得模型可以灵活地建模
  不同尺度上的长程相互作用。

  例如:
    - 半衰期=3: 距离 3 个位置时衰减到一半 → 建模短程 motif 相互作用
    - 半衰期=768: 距离 768 个位置时衰减到一半 → 建模中程调控相互作用
    - 半衰期=1536: 距离 1536 个位置时衰减到一半 → 建模长程增强子-启动子相互作用

  Args:
    positions: 位置张量（任意形状），通常为绝对相对距离 |i-j|。
    feature_size: 要使用的基函数数量。论文中此值为 32（每个分量）。
    seq_length: 序列长度 L = 1536。用于确定半衰期的最大值范围。
    bin_size: (未使用)。参见 positional_features_all。
    min_half_life: 半衰期网格中的最小指数半衰期。默认 3.0。
                   对应约 3×128 = 384 bp。

  Returns:
    形状为 positions.shape + [feature_size] 的张量。
    例如 positions = [2T-1] 时，返回 [3071, 32]。
  """
  del bin_size  # 未使用。
  if seq_length is None:
    seq_length = tf.reduce_max(tf.abs(positions)) + 1

  # 在对数空间中从 [min_half_life=3, seq_length=1536] 均匀分布 feature_size 个半衰期。
  # max_range = log2(1536) ≈ 10.58
  # 半衰期取值: 2^3, 2^{3+7.58/31}, 2^{3+2·7.58/31}, ..., 2^{10.58}
  # = 8, 9.5, 11.3, ..., 1536
  seq_length = tf.cast(seq_length, dtype=tf.float32)
  max_range = tf.math.log(seq_length) / tf.math.log(2.0)
  half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
  half_life = _prepend_dims(half_life, positions.shape.rank)
  positions = tf.abs(positions)

  # 指数衰减公式:
  # exp(-log(2) / half_life * r)
  # = exp(-r * log(2) / half_life)
  # = 2^(-r / half_life)
  # 当 r = half_life 时，值为 0.5（半衰）。
  outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def positional_features_central_mask(positions: tf.Tensor,
                                      feature_size: int,
                                      seq_length: Optional[int] = None,
                                      bin_size: Optional[int] = None):
  """使用中心掩码的位置特征（仅允许中心范围内的特征）。

  论文公式:
    f_i^{central_mask}(r) = 1  如果 r ≤ 2^i
                            0  否则

  物理含义: 这是一种硬阈值形式的距离编码，每个特征定义了一个"窗口"，
  窗口内的位置获得值 1（被关注），窗口外获得值 0（被忽略）。
  窗口大小以 2 的幂次增长: 2^1, 2^2, ..., 2^{feature_size}。

  例如:
    - i=1: 窗口半径 = 1 → 仅关注距离 ≤1 的相邻位置
    - i=5: 窗口半径 = 32 → 关注约 4kb 范围内的位置
    - i=10: 窗口半径 = 1024 → 关注约 131kb 范围内的位置

  这允许模型学习在不同距离尺度上选择性地关注或忽略位置，
  类似于生物学中不同调控元件在不同距离范围内发挥作用。

  Args:
    positions: 位置张量（任意形状），通常为绝对距离。
    feature_size: 要使用的中心掩码数量。论文中此值为 32。
    seq_length: (未使用)。
    bin_size: (未使用)。

  Returns:
    形状为 positions.shape + [feature_size] 的二值张量。
  """
  del seq_length  # 未使用。
  del bin_size  # 未使用。
  # 中心窗口宽度: 2^1 - 1, 2^2 - 1, 2^3 - 1, ..., 2^32 - 1
  # 每个特征 i 的窗口半径为 2^i - 1
  center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32))
  center_widths = center_widths - 1
  center_widths = _prepend_dims(center_widths, positions.shape.rank)
  # 比较: 如果 center_widths > |positions| 则输出 1，否则 0
  outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis],
                     tf.float32)
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def gamma_pdf(x, concentration, rate):
  """Gamma 概率分布函数: p(x | α=concentration, β=rate)。

  公式:
    Gamma(x | α, β) = β^α / Γ(α) · x^{α-1} · exp(-βx)

  对数空间实现以避免数值下溢/溢出:
    log p = (α-1)·log(x) - βx - [log Γ(α) - α·log(β)]

  其中 log Γ(α) - α·log(β) 是归一化常数（对数空间）。

  Args:
    x: 输入值（距离）。
    concentration: Gamma 分布的形状参数 α。
    rate: Gamma 分布的速率参数 β。

  Returns:
    Gamma PDF 在 x 处的值。
  """
  # 非归一化对数概率: (α-1)·log(x) - βx
  log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
  # 对数归一化常数: log Γ(α) - α·log(β)
  log_normalization = (tf.math.lgamma(concentration) -
                       concentration * tf.math.log(rate))
  # 通过指数恢复标准概率值
  return tf.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(positions: tf.Tensor,
                               feature_size: int,
                               seq_length: Optional[int] = None,
                               bin_size: Optional[int] = None,
                               stddev=None,
                               start_mean=None):
  """使用 Gamma 分布计算的位置特征。

  论文公式:
    f_i^{gamma}(r) = Gamma(r | α=μ_i²/σ², β=μ_i/σ²)

  其中 μ_i 从 seq_length/feature_size 到 seq_length 线性分布，
  σ = seq_length / (2 × feature_size)。

  物理含义: Gamma 分布提供了比指数衰减更灵活的距离编码曲线。
  通过调节形状参数 α 和速率参数 β，可以生成从类似指数的衰减
  到钟形曲线的多种形状。

  - 均值 μ 控制 Gamma 分布的峰值位置：
    μ_1 ≈ 1536/32 ≈ 48 → 峰值在约 48 个位置 (≈ 6 kb)
    μ_32 = 1536 → 峰值在约 1536 个位置 (≈ 197 kb)
  - 标准差 σ = 1536/64 = 24 → 控制分布的展宽
  - α = (μ/σ)², β = μ/σ² → 标准 Gamma 参数化

  每个特征在归一化时除以其最大值，确保值域在 [0, 1]。
  加 1e-8 是为了数值稳定性（防止除以零）。

  Args:
    positions: 位置张量（任意形状）。
    feature_size: 基函数数量。论文中此值为 32。
    seq_length: 序列长度 L = 1536。
    bin_size: (未使用)。
    stddev: Gamma 分布的标准差。默认 seq_length / (2*feature_size) = 24。
    start_mean: 起始均值。默认 seq_length / feature_size ≈ 48。

  Returns:
    形状为 positions.shape + [feature_size] 的归一化 Gamma 概率张量。
  """
  del bin_size  # 未使用。
  if seq_length is None:
    seq_length = tf.reduce_max(tf.abs(positions)) + 1
  if stddev is None:
    # σ = seq_length / (2 * feature_size)
    # 论文: σ = 1536 / (2 × 32) = 24
    stddev = seq_length / (2 * feature_size)
  if start_mean is None:
    # μ 的起始值 = seq_length / feature_size
    # 论文: μ_1 = 1536 / 32 ≈ 48
    start_mean = seq_length / feature_size

  # 均值 μ 从 start_mean 到 seq_length 线性分布
  mean = tf.linspace(start_mean, seq_length, num=feature_size)
  mean = _prepend_dims(mean, positions.shape.rank)

  # Gamma 分布参数:
  # α = (μ/σ)²: 形状参数，越大分布越集中在均值附近
  # β = μ/σ²:  速率参数，控制衰减速度
  concentration = (mean / stddev)**2  # 论文中的 α
  rate = mean / stddev**2             # 论文中的 β

  # 在每个绝对距离 |i-j| 上计算 Gamma PDF 值
  probabilities = gamma_pdf(
      tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis],
      concentration, rate)
  # 加 1e-8 以确保数值稳定性（防止除以零在后续归一化中）
  probabilities += 1e-8  # 确保数值稳定性。
  # 归一化: 沿距离轴将每个特征除以其最大值，使值域在 [0, 1]
  outputs = probabilities / tf.reduce_max(probabilities,
                                           axis=1, keepdims=True)
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def positional_features_cosine(positions: tf.Tensor,
                                feature_size: int,
                                seq_length: Optional[int] = None,
                                bin_size: Optional[int] = None):
  """余弦位置特征（备选方案，论文主模型未使用）。

  使用不同频率的余弦函数编码位置:
    f_i(r) = cos(2π · r / periodicity_i)

  其中 periodicity_i = 1.25 × 2^i，提供了从低频到高频的多尺度周期编码。

  这类似于标准 Transformer 中的正弦位置编码，但应用于相对距离而非
  绝对位置。周期从约 1.25 到 1.25×2^{feature_size-1}。

  Args:
    positions: 位置张量（任意形状）。
    feature_size: 特征数量。
    seq_length: (未使用)。
    bin_size: (未使用)。

  Returns:
    形状为 positions.shape + [feature_size] 的余弦编码张量。
  """
  del bin_size  # 未使用。
  del seq_length  # 未使用。
  # 周期从 1.25 开始，每个特征翻倍
  periodicity = 1.25 * tf.pow(2.0, tf.range(0, feature_size, dtype=tf.float32))
  periodicity = _prepend_dims(periodicity, positions.shape.rank)

  outputs = tf.math.cos(2 * np.pi * positions[..., tf.newaxis] / periodicity)
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def positional_features_linear_masks(positions: tf.Tensor,
                                      feature_size: int,
                                      seq_length: Optional[int] = None,
                                      bin_size: Optional[int] = None):
  """指数增长的点聚焦（备选方案，论文主模型未使用）。

  每个特征在特定距离处激活（值为 1），其他距离为 0:
    f_i(r) = 1  if |r| == i
             0  otherwise

  这提供了精确距离的"独热"编码: 特征 0 编码距离 0，特征 1 编码距离 1，
  依此类推。这是最细粒度的距离编码方式，但缺乏泛化能力。

  Args:
    positions: 位置张量（任意形状）。
    feature_size: 特征数量（也等于最大编码距离）。
    seq_length: (未使用)。
    bin_size: (未使用)。

  Returns:
    形状为 positions.shape + [feature_size] 的独热编码张量。
  """
  del bin_size  # 未使用。
  del seq_length  # 未使用。
  distances = tf.range(0, feature_size, dtype=tf.float32)
  distances = _prepend_dims(distances, positions.shape.rank)
  # 精确距离匹配: 每个特征对应一个特定距离值
  outputs = tf.cast(distances == tf.abs(positions[..., tf.newaxis]),
                     dtype=tf.float32)

  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def positional_features_sin_cos(positions: tf.Tensor,
                                 feature_size: int,
                                 seq_length: Optional[int] = None,
                                 bin_size: Optional[int] = None,
                                 max_time=10000.0):
  """正弦/余弦位置编码（备选方案，论文主模型未使用）。

  这是标准 Transformer (Vaswani et al., 2017) 中的位置编码，
  但应用于相对距离而非绝对位置:
    PE(r, 2j)   = sin(r / 10000^{2j/feature_size})
    PE(r, 2j+1) = cos(r / 10000^{2j/feature_size})

  其中 j = 0, 1, ..., feature_size/2 - 1。

  不同频率的正弦/余弦函数允许模型通过点积自然地学习相对距离:
  PE(r)^T · PE(r+k) 的值仅依赖于位移 k，这是正弦/余弦编码的重要性质。

  Args:
    positions: 位置张量（任意形状）。
    feature_size: 特征数量（必须是偶数）。
    seq_length: (未使用)。
    bin_size: (未使用)。
    max_time: 最大时间尺度参数。默认 10000.0。

  Returns:
    形状为 positions.shape + [feature_size] 的正弦/余弦编码张量。

  Raises:
    ValueError: 如果 feature_size 不是偶数。
  """
  del bin_size  # 未使用。
  del seq_length  # 未使用。
  if feature_size % 2 != 0:
    raise ValueError('feature_size needs to be divisible by 2.')
  # 频率索引: 0, 2, 4, ..., feature_size-2
  i = tf.range(0, feature_size, 2, dtype=tf.float32)
  i = _prepend_dims(i, positions.shape.rank)

  # 拼接正弦和余弦
  # 正弦对应偶数索引 (2j)，余弦对应奇数索引 (2j+1)
  outputs = tf.concat([
      tf.sin(positions[..., tf.newaxis] / max_time**(i / feature_size)),
      tf.cos(positions[..., tf.newaxis] / max_time**(i / feature_size))], -1)

  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


# ==============================================================================
# 整体流程总结
# ==============================================================================
#
# 本文件实现了 Enformer 模型的两个核心组件: TransformerBlock 和
# MultiheadAttention。以下是完整的前向传播流程:
#
# 1. 输入准备 (在 Enformer 主模型中):
#    - DNA 序列独热编码: [B, 196608, 4]
#    - 通过 7 个卷积块 + attention pooling (窗口=2, stride=2):
#      196608 → 98304 → 49152 → 24576 → 12288 → 6144 → 3072 → 1536
#    - 卷积塔输出: x ∈ R^{B × 1536 × C}，其中 C = V×H
#
# 2. Transformer 块堆叠 (11 个, 本文件实现):
#    对于每个 TransformerBlock (共 11 个):
#      a) MHA 子层:
#         - LayerNorm → MultiheadAttention → Dropout → 残差连接
#         - MultiheadAttention 内部:
#           · 投影: q=x·W^q, k=x·W^k, v=x·W^v
#           · 位置编码: 用 3 类基函数 (exponential, central_mask, gamma)
#             计算 f(i-j) ∈ R^{Cr} (Cr=V=192)
#           · 相对位置嵌入: r = w^R·f(i-j)
#           · 注意力: a = softmax(q·k^T/√K + q·r^T + u·k^T + v·r^T)
#           · 聚合: output = a·v
#           · 拼接 + 线性组合
#      b) MLP 子层:
#         - LayerNorm → Linear(C→2C) → ReLU → Linear(2C→C) → Dropout → 残差连接
#
# 3. 输出处理 (在 Enformer 主模型中):
#    - 裁剪层: 两端各去掉 320 个位置 (避免边界效应的不对称性)
#    - 物种特异性点卷积头:
#      · 人类头: 预测 5,313 个基因组 track
#      · 小鼠头: 预测 1,643 个基因组 track
#    - 输出: [B, 896, 5313] (人类) 或 [B, 896, 1643] (小鼠)
#
# 4. 训练循环 (在主训练脚本中):
#    - Loss: Poisson 负对数似然 L = -Σ y·log(ŷ) + ŷ
#    - Optimizer: Adam (lr=0.0005, β1=0.9, β2=0.999, ε=1e-8)
#    - 学习率调度: 前 5,000 步从 0 线性预热到目标值
#    - 梯度裁剪: tf.clip_by_global_norm(grads, max_norm=0.2)
#    - 批次策略: 人类/小鼠交替批次
#    - 数据增强: 随机平移 ≤3bp + 反向互补
#    - 验证: 每 1,000 步评估 CAGE TSS Spearman 相关系数
#    - 微调: 仅人类数据, lr=0.0001, 30,000 步
# ==============================================================================
