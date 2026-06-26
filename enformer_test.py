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
"""通过将随机序列作为输入来测试 Enformer 模型。

==============================================================================
测试文件概述
==============================================================================

本测试是 Enformer 模型最基本的"冒烟测试"(smoke test)，用于验证:
1. 模型可以成功构建和初始化（所有子模块的维度兼容）
2. 前向传播不会抛出异常（张量形状在所有层中正确传播）
3. 人类和小鼠两个预测头的输出维度符合论文声明

测试方法（命令行）:
    $ python enformer_test.py

==============================================================================
测试覆盖的功能验证（与论文 Methods 部分对照）:
==============================================================================

- 模型实例化: channels=1536, num_transformer_layers=11
  对应论文主模型配置: value_size=192 × num_heads=8 = 1536 通道
  这是论文 Extended Data Fig. 1a 中标注的参数。

- 输入处理: 生成长度为 SEQUENCE_LENGTH=196,608 bp 的随机 DNA 序列
  对应论文输入规格: "Enformer takes as input one-hot-encoded DNA
  sequence of length 196,608 bp"
  编码为 [1, 196608, 4] (batch_size=1, 仅用于形状验证)。

- 前向传播: model(inputs, is_training=True)
  训练模式下的完整前向传播，数据流经以下路径:
  STEM(Conv1D+Pooling) → CONV_TOWER(6 blocks) → TRANSFORMER(11 blocks)
  → CROP(trim 320 each side) → FINAL_POINTWISE(Conv1D+GELU)
  → HEADS(Linear+Softplus)
  输出: human [B, 896, 5313], mouse [B, 896, 1643]

- 输出形状验证:
  人类: [1, 896, 5313]
    5313 = 2131 TF ChIP-seq + 1860 histone ChIP-seq
           + 684 DNase-seq/ATAC-seq + 638 CAGE
  小鼠: [1, 896, 1643]
    1643 = 308 TF ChIP-seq + 750 histone ChIP-seq
           + 228 DNase-seq/ATAC-seq + 357 CAGE
  详见论文 Supplementary Table 2 (human) 和 Table 3 (mouse)。

  896 = TARGET_LENGTH = 1536 - 2×320
  对应论文: "The cropping layer trims 320 positions on each side
  to avoid computing the loss on the far ends"

==============================================================================
与完整训练/验证流程的关系:
==============================================================================
本测试只验证前向传播的形状正确性，不涉及:
- 损失计算: Poisson 负对数似然 (论文 "Model training and evaluation")
  L = -Σ_i [y_i · log(ŷ_i) - ŷ_i]，其中 ŷ_i = softplus(线性层输出)
- 梯度反向传播: 需要真实的目标值 y 才能计算 loss → backward
- 参数更新: 需要 optimizer (Adam, lr=0.0005)
- 数据增强: 随机平移 ≤3 bp + 反向互补
- 测试时增强: 8 个增强序列的平均预测

完整的训练流程见 enformer-training.ipynb。
==============================================================================
"""

import random
import unittest

import enformer
import numpy as np


class TestEnformer(unittest.TestCase):
  """Enformer 模型的单元测试类。

  此测试是模型开发过程中的基本验证手段，确保对模型代码的任何修改
  不会破坏前向传播的维度兼容性和基本功能。

  对应论文 Methods 部分描述的标准配置:
  - channels = 1536 (= value_size 192 × num_heads 8)
  - num_transformer_layers = 11

  注意: 本测试不验证预测值的正确性（因为输入是随机序列），
  仅验证模型架构的完整性（无崩溃 + 输出形状正确）。
  """

  def test_enformer(self):
    """测试 Enformer 模型的前向传播（主模型配置）。

    验证流程:
    1. 构建论文主模型配置的 Enformer 实例
    2. 生成随机 DNA 序列（长度 196,608 bp）作为输入
    3. 执行训练模式下的前向传播
    4. 验证人类和小鼠两个预测头的输出形状

    模型构建的内部步骤（代码透明性说明）:
      Enformer.__init__ 中依次构建:
        a) Stem: Conv1D(channels//2=768, kernel=15)
           → Residual(ConvBlock(768, 1))
           → SoftmaxPooling1D(pool_size=2, w_init_scale=2.0)
           输入 [B,196608,4] → 输出 [B,98304,768]

        b) ConvTower: 6×(ConvBlock(filters, 5) → Residual → SoftmaxPooling1D(2))
           filters 从 768 指数增长到 1536:
             块0: [B,98304,768]   → [B,49152, filts[0]]
             块1: [B,49152, filts[0]]  → [B,24576, filts[1]]
             块2: [B,24576, filts[1]]  → [B,12288, filts[2]]
             块3: [B,12288, filts[2]]  → [B,6144,  filts[3]]
             块4: [B,6144,  filts[3]]  → [B,3072,  filts[4]]
             块5: [B,3072,  filts[4]]  → [B,1536,  1536]
           每个位置代表 128 bp (2^7 × 128 = 196608/1536)。

        c) Transformer: 11 × TransformerBlock
           每个块 = Residual(MHA) + Residual(MLP)
           MHA 配置: H=8, K=64, V=192, 3类相对位置基函数
           MLP: C→2C→C (1536→3072→1536)
           形状不变: [B,1536,1536]

        d) Crop: TargetLengthCrop1D(896)
           两端各去掉 (1536-896)//2 = 320 个位置
           [B,1536,1536] → [B,896,1536]

        e) FinalPointwise: ConvBlock(3072, 1) → Dropout(0.05) → GELU
           [B,896,1536] → [B,896,3072]

        f) Heads:
           人类: Linear(3072→5313) + Softplus → [B,896,5313]
           小鼠: Linear(3072→1643) + Softplus → [B,896,1643]
           Softplus(x)=log(1+e^x) 确保输出恒正（Poisson NLL 要求）。

    断言说明:
    - self.assertEqual(outputs['human'].shape, (1, 896, 5313))
      验证人类基因组 track 预测的形状。
      如果失败，可能原因: trunk 或 human head 的维度配置有误。
    - self.assertEqual(outputs['mouse'].shape, (1, 896, 1643))
      验证小鼠基因组 track 预测的形状。
      如果失败，可能原因: trunk 或 mouse head 的维度配置有误。
    """
    # ---- 步骤 1: 构建论文主模型配置的 Enformer 模型 ----
    # channels=1536: 对应 value_size=192 × num_heads=8
    #   这是论文中的主模型配置（非消融实验的 channels=768）。
    # num_transformer_layers=11: 论文通过消融实验确定的最优深度。
    #   比 11 更深的模型收益递减，更浅的模型性能下降。
    #
    # 模型初始化时执行:
    #   - 创建所有 Sonnet 模块和 TensorFlow 变量
    #   - 设置权重初始化器（He 初始化 + 零初始化 MHA 最终层）
    #   - 构建计算图结构（变量仅在首次 forward 时实际分配内存）
    # 注意: 此步骤可能较慢（~几秒），因为涉及大量变量创建。
    model = enformer.Enformer(channels=1536, num_transformer_layers=11)

    # ---- 步骤 2: 生成随机 DNA 序列输入 ----
    # 序列长度 = SEQUENCE_LENGTH = 196,608 bp
    #   这是论文相对于 Basenji2 (131,072 bp) 的 1.5 倍改进。
    #   更长的输入序列使模型能"看到"更远的调控元件，
    #   对于增强子-启动子相互作用（可达 100 kb 以上）至关重要。
    #
    # 随机序列仅包含 A, C, G, T（不含 N/未知碱基）。
    # 独热编码: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]。
    # 形状: [1, 196608, 4] — batch_size=1 用于快速测试。
    #
    # 在实际训练中:
    #   batch_size=64，每个 TPU v3 核心 1 个样本。
    #   人类/小鼠交替批次: step%2==0 → human, step%2==1 → mouse。
    #   数据增强: 随机平移 ≤3 bp + 50% 概率反向互补。
    inputs = _get_random_input()

    # ---- 步骤 3: 执行前向传播（训练模式） ----
    # is_training=True:
    #   - Dropout 层: 处于活跃状态，随机丢弃神经元
    #     (MHA attention_dropout_rate=0.05, positional_dropout_rate=0.01,
    #      Transformer dropout_rate=0.4, final dropout_rate=0.05)
    #   - BatchNorm: 使用当前 batch 的均值和方差统计量
    #     (而非训练期间累积的移动平均)
    #   - 对于 batch_size=1 的随机输入:
    #     BatchNorm 统计量不具代表性，但不影响形状测试。
    #
    # 推理时 (is_training=False):
    #   - 所有 Dropout 关闭（输出变为确定性）
    #   - BatchNorm 使用训练期间累积的移动平均统计量
    #   - 使用测试时增强: 对 8 个随机增强序列取平均
    #
    # 前向传播的数据流 (详细维度追踪):
    #   输入 [1, 196608, 4]
    #   → Stem: Conv1D(768, 15) → [1, 196608, 768]
    #           Residual → [1, 196608, 768]
    #           SoftmaxPooling1D(2) → [1, 98304, 768]
    #   → ConvTower Block 0: Conv(5)+Pool(2) → [1, 49152, filts[0]]
    #   → ConvTower Block 1: Conv(5)+Pool(2) → [1, 24576, filts[1]]
    #   → ConvTower Block 2: Conv(5)+Pool(2) → [1, 12288, filts[2]]
    #   → ConvTower Block 3: Conv(5)+Pool(2) → [1, 6144,  filts[3]]
    #   → ConvTower Block 4: Conv(5)+Pool(2) → [1, 3072,  filts[4]]
    #   → ConvTower Block 5: Conv(5)+Pool(2) → [1, 1536,  1536]
    #       此时每个位置代表 2^7=128 bp，共 1536×128=196,608 bp
    #   → Transformer 0..10: MHA+MLP (形状不变) → [1, 1536, 1536]
    #       11 个块顺序处理，每个块内部:
    #         x → LN → MHA(H=8,K=64,V=192) → Dropout → +x (残差)
    #         x → LN → Linear(3072)→ReLU→Linear(1536) → Dropout → +x
    #   → Crop: trim 320→[1, 896, 1536]
    #   → FinalPointwise: Conv1D(3072,1)→Dropout→GELU → [1, 896, 3072]
    #   → Human head: Linear(3072→5313)+Softplus → [1, 896, 5313]
    #   → Mouse head: Linear(3072→1643)+Softplus → [1, 896, 1643]
    outputs = model(inputs, is_training=True)

    # ---- 步骤 4a: 验证人类预测头输出形状 ----
    # 期望: (batch=1, TARGET_LENGTH=896, 5313 human tracks)
    #
    # 5313 个人类 track 的构成 (论文 Supplementary Table 2):
    #   - 2131 TF ChIP-seq: 转录因子染色质免疫沉淀测序
    #     包含数百个 TF（如 CTCF, POLR2A, EP300 等）在多种细胞类型中的结合
    #   - 1860 histone ChIP-seq: 组蛋白修饰
    #     如 H3K27ac(活性增强子), H3K4me3(活性启动子),
    #        H3K36me3(转录延伸), H3K27me3(抑制) 等
    #   - 684 DNase-seq/ATAC-seq: 染色质可及性
    #     反映开放染色质区域，即潜在的调控元件位置
    #   - 638 CAGE: Cap Analysis of Gene Expression
    #     转录起始位点 (TSS) 的精确位置和表达水平
    #
    # 输出值范围: 由于 Softplus 激活，所有值 > 0。
    #   在 Poisson NLL 损失中，这些值被解释为预测的 reads counts。
    #   对于 CAGE track，TSS 附近的 3 个 bin 求和得到基因表达预测值。
    self.assertEqual(outputs['human'].shape, (1, enformer.TARGET_LENGTH, 5313))

    # ---- 步骤 4b: 验证小鼠预测头输出形状 ----
    # 期望: (batch=1, TARGET_LENGTH=896, 1643 mouse tracks)
    #
    # 1643 个小鼠 track 的构成 (论文 Supplementary Table 3):
    #   - 308 TF ChIP-seq: 远少于人类，反映小鼠数据资源的相对有限
    #   - 750 histone ChIP-seq: 同样少于人类
    #   - 228 DNase-seq/ATAC-seq: 染色质可及性数据
    #   - 357 CAGE: 基因表达数据
    #
    # 人类和小鼠共享 trunk 参数，交替批次训练:
    #   for step in range(150000):
    #     species = 'human' if step % 2 == 0 else 'mouse'
    #     batch = dataset[species].next()
    #     preds = model(batch.inputs, is_training=True)
    #     loss = poisson_nll(preds[species], batch.targets)
    #     grads = tape.gradient(loss, model.trainable_variables)
    #     grads, _ = tf.clip_by_global_norm(grads, 0.2)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #
    # 物种交替训练的好处:
    #   - Trunk 学习到跨物种共享的调控语法
    #   - 数据量更大的物种（人类）有助于改善小鼠预测（迁移学习效应）
    #   - 训练后通过微调阶段 (lr=0.0001, 30k steps, 仅人类数据)
    #     进一步优化人类特异性预测
    self.assertEqual(outputs['mouse'].shape, (1, enformer.TARGET_LENGTH, 1643))


def _get_random_input():
  """生成一个随机的独热编码 DNA 序列，用于模型测试。

  生成过程（对应论文 Methods 中的输入格式说明）:

  1. 随机生成长度为 196,608 的 DNA 序列
     从 {A, C, G, T} 中均匀随机采样每个位置。
     random.choice('ACGT') 等价于多分类分布 p=[0.25,0.25,0.25,0.25]。
     实际基因组中碱基分布不均匀（GC 含量 ~41%），但测试不需要真实分布。

  2. 使用 enformer.one_hot_encode 将其转换为独热编码
     编码规则（论文 Methods）:
       A → [1, 0, 0, 0]
       C → [0, 1, 0, 0]
       G → [0, 0, 1, 0]
       T → [0, 0, 0, 1]
       N → [0, 0, 0, 0]  (未知碱基编码为零向量，此处不使用)
     形状: [196608, 4]

  3. 扩展 batch 维度: [196608, 4] → [1, 196608, 4]
     np.expand_dims(seq, 0) 在第 0 轴前插入维度。
     等价于 seq[np.newaxis, :, :] 或 tf.expand_dims(seq, 0)。

  4. 转换为 float32 类型
     astype(np.float32) 确保与 TensorFlow 默认数据类型一致。
     Sonnet/TensorFlow 模型默认使用 float32，传递 float64 可能导致
     类型转换警告或性能下降。

  Returns:
    np.ndarray: 形状为 [1, 196608, 4] 的独热编码随机 DNA 序列。
    数据类型: np.float32。

  注意:
    此函数生成的随机序列仅用于测试模型形状，预测值无生物学意义。
    实际基因组序列应使用 hg38/mm10 参考基因组提取。
    论文中训练序列来自 "training/validation/test sets" 的分区方式:
      1) 人类和小鼠基因组各分为 1 Mb 区域
      2) 构建二分图，边表示 >100 kb 的同源比对序列
      3) 随机划分连通分量为 train/val/test
    这确保了同源序列不会跨越数据集，避免数据泄漏。
  """
  # 生成随机 DNA 序列字符串
  # 使用列表推导式 + random.choice 逐一生成碱基。
  # 复杂度: O(L), L=196,608 — 每次测试调用需要 ~0.1 秒。
  # 备选方案: np.random.choice(['A','C','G','T'], size=196608)
  #   可能更快，但此处使用 random.choice 以保持与参考实现的兼容性。
  seq = ''.join(
      [random.choice('ACGT') for _ in range(enformer.SEQUENCE_LENGTH)])

  # 独热编码 + 添加 batch 维度 + 类型转换
  # one_hot_encode 使用 uint8 查找表实现，对 196,608 bp 序列非常高效。
  # 查找表大小: 256 × 4 = 1024 字节（极小的内存占用）。
  # 内部流程:
  #   string → ascii bytes → uint8 数组 → 查找表索引 → 独热向量
  return np.expand_dims(enformer.one_hot_encode(seq), 0).astype(np.float32)


if __name__ == '__main__':
  # ---- 运行所有测试 ----
  # unittest.main() 执行流程:
  #   1. 自动发现 TestEnformer 类（继承自 unittest.TestCase）
  #   2. 执行所有以 test_ 开头的方法
  #   3. 对每个测试方法:
  #      a) 调用 setUp()（如果定义）进行测试前置准备
  #      b) 调用 test_xxx() 方法
  #      c) 调用 tearDown()（如果定义）进行清理
  #   4. 汇总结果，打印通过/失败的测试数量
  #
  # 测试通过标准:
  #   - 无未捕获的异常抛出
  #   - 所有 assertEqual/assert* 断言通过
  #   - 所有层的前向维度传播兼容（否则 TensorFlow 会在某层抛出异常）
  #
  # 测试失败的可能原因:
  #   - 模型代码修改导致维度不匹配
  #   - TensorFlow/Sonnet 版本不兼容
  #   - 依赖文件 (enformer.py, attention_module.py) 有语法错误
  #   - 内存不足（需要 ~2 GB 加载完整模型）
  #
  # 注意: 此测试不验证训练收敛性或预测准确性。
  # 训练收敛性需要通过完整的数据加载和训练循环来验证（见 enformer-training.ipynb）。
  unittest.main()
