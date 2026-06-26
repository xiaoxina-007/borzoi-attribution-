#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

# 命令行参数解析
from optparse import OptionParser

# 标准库
import gc          # 垃圾回收，手动释放 GPU 显存
import json        # 读取模型参数 JSON
import os          # 文件路径操作
import time        # 计时
import h5py        # HDF5 格式输出（存储大规模梯度数据）
import numpy as np # 数值计算
import pandas as pd # 表格数据处理（targets_df, gene_df）
import pysam       # 参考基因组随机访问（FastaFile）
import tempfile    # GCS 模式临时目录
import pickle      # options_pkl 序列化
import shutil      # 清理临时目录
import pdb         # 调试断点
import tensorflow as tf  # 深度学习框架（GPU 前向/反向传播）

# Baskerville/Borzoi 项目内部模块
from baskerville.dataset import targets_prep_strand  # 靶标链方向预处理
from baskerville import dna         # DNA 序列操作（one-hot 编码, 反向互补, 平移增强）
from baskerville import gene as bgene  # 基因结构解析（GTF → Transcriptome, output_slice）
from baskerville import seqnn      # Borzoi SeqNN 模型定义 + 梯度计算接口
from baskerville.helpers.gcs_utils import (
    upload_folder_gcs,       # 上传本地目录到 GCS
    download_rename_inputs,  # 从 GCS 下载文件到本地
)
from baskerville.helpers.utils import load_extra_options  # 从 pickle 加载额外选项

"""
borzoi_satg_gene_gpu.py

对 GTF 文件中指定的基因执行梯度显著性分析（GPU 友好版本）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
论文对应关系（Borzoi method — "Input sequence attribution" 节）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本脚本实现论文中描述的 "Expression attribution"（表达量归因）的梯度显著性计算：

  设 ℳ 为 Borzoi 模型，
     𝐱 ∈ {0,1}^{524288×4} 为 one-hot 编码的输入 DNA 序列，
     𝐲 = ℳ(𝐱) ∈ (0,+∞]^{16384×7611} 为覆盖度预测（经过逆变换还原到 count 空间），
     𝒯 = {t₀,...,t_T} 为目标覆盖度 track（组织/实验）索引集合，
     ℬ = {b₀,...,b_B} 为与目标基因外显子重叠的 32 bp bin 索引集合。

  汇总统计量 u（expression attribution）：
    u = log( C + (1/T) × Σ_{t∈𝒯} Σ_{b∈ℬ} 𝐲_{b,t} )

  其中 C 是伪计数（pseudo count），防止低表达基因的梯度爆炸。

  梯度显著性得分 𝐬 ∈ ℝ^{524288×4}（Gradient × Input）：
    𝐬_{i,j} = ∂u(𝐱)/∂𝐱_{i,j} - (1/4) × Σ_{k=1}^{4} ∂u(𝐱)/∂𝐱_{i,k}

  即对输入 one-hot 编码的每个位置 i 和每个碱基通道 j 计算偏导数，
  然后减去该位置四个碱基通道的均值（subtract_avg=True），以消除碱基无关的基线偏移。

  可视化时仅提取参考碱基对应通道的得分：
    𝐬_i^{(vis)} = Σ_{j=1}^{4} 𝐬_{i,j} × 𝐱_{i,j}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练数据变换（论文 "Training data" 节 — "squashed scale"）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

训练时对覆盖度值做了如下变换以压缩动态范围：

  𝐲_{j,t}^{(squashed)} = {
      𝐲_{j,t}^{(3/4)}                              if 𝐲_{j,t}^{(3/4)} ≤ 384,
      384 + √(𝐲_{j,t}^{(3/4)} - 384)               otherwise
  }

然后乘以 track_scale 因子。预测时需要逆变换还原到 count 空间（见 _count_func）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
脚本整体流程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  阶段 1: 解析命令行参数和配置文件
  阶段 2: （可选）从 GCS 下载输入文件到本地
  阶段 3: 读取模型参数（params_model, params_train）和靶标定义（targets_df）
  阶段 4: 加载第一个模型 fold 以获取结构参数（model_stride, model_crop, target_length）
  阶段 5: 解析 GTF 文件，为每个基因确定序列窗口坐标（以 gene midpoint 为中心）
  阶段 6: 初始化 HDF5 输出文件（datasets: seqs, grads, preds, 基因元数据）
  阶段 7: 对每个 fold：
      a) （可选）先做前向预测，计算并存储标量预测值 u
      b) （可选）根据预测值分布的分位数计算 pseudo count C
      c) 对每个 shift（序列平移增强）和每个 rev_comp（正/反向互补链）：
         - 批量提取基因序列、做 one-hot 编码、平移增强、确定输出位置切片
         - 调用 seqnn_model.gradients() 计算 ∂u/∂𝐱
         - 撤销增强变换（unaugment_grads），将梯度累加到 HDF5
  阶段 8: 保存原始序列，梯度除以集成大小（n_shifts × (2 if rc else 1)）得到平均
  阶段 9: 关闭文件，（可选）上传到 GCS
"""


################################################################################
# main
# ###############################################################################
def main():
    """
    【主函数】—— 梯度显著性分析流程的入口。

    该函数实现论文 "Input sequence attribution" 节中 Expression attribution
    的完整计算管线：对 GTF 中每个基因，计算汇总统计量 u = log(C + mean_track Σ_b 𝐲_{b,t})
    对输入序列 𝐱 的梯度 ∂u/∂𝐱，作为每个碱基位置的显著性得分。

    关键设计决策：
    -------------
    1. 缓冲写入机制（buffer_size=1024）：
       为避免一次性将所有基因数据加载到 GPU 显存，将基因分批处理。
       当累积的基因数达到 buffer_size 或处理完所有基因时，
       批量进行一次前向/反向传播，将结果写入 HDF5 后清空缓冲区。

    2. 集成策略（ensemble）：
       多 fold + 多 shift + 正反向互补链的梯度取平均，降低方差。
       最终梯度 = (1 / (n_folds × n_shifts × (2 if rc else 1))) × Σ grads

    3. 多进程支持：
       通过 worker_index 将基因列表切分给不同进程并行处理。
    """
    usage = "usage: %prog [options] <params> <model> <gene_gtf>"
    parser = OptionParser(usage)
    parser.add_option(
        "--fa",
        dest="genome_fasta",
        # default="%s/assembly/ucsc/hg38.fa" % os.environ["HG38"],
        default=None,
        help="参考基因组 FASTA 文件路径（hg38 或 mm10），用于提取基因周围的 DNA 序列 [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="satg_out",
        help="输出目录，HDF5 结果文件将写入此目录 [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="多进程并行时的总进程数，由主调度脚本（multi script）传入。每个 worker 只处理基因列表的一部分 [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=0,
        type="int",
        help="是否集成正向链和反向互补链的预测：1=同时计算正向(fwd)和反向互补(rev)两个方向的梯度并累加 [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="folds",
        default="0",
        type="str",
        help="用于集成的模型 fold 列表，逗号分隔。Borzoi 训练了 4 个独立初始化的模型副本（replicates），"
             "通常使用 '0,1,2,3' 进行集成预测 [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="集成预测时的序列平移量列表，逗号分隔。例如 '0,1,-1' 表示在三个平移位置上分别计算梯度后平均，"
             "用于增强预测的平移鲁棒性 [Default: %default]",
    )
    parser.add_option(
        "--span",
        dest="span",
        default=0,
        type="int",
        help="是否聚合整个 gene span（而非仅外显子区域）的输出：1=将基因的全长（从 TSS 到 TES）"
             "作为输出位置范围 [Default: %default]",
    )
    parser.add_option(
        "--smoothgrad",
        dest="smooth_grad",
        default=0,
        type="int",
        help="是否启用 SmoothGrad 降噪：1=对输入 𝐱 添加随机噪声（以 sample_prob 概率保留原始碱基），"
             "进行 n_samples 次采样后对梯度取平均。这等价于对梯度场做高斯平滑，可减少显著性图中的高频噪声 [Default: %default]",
    )
    parser.add_option(
        "--samples",
        dest="n_samples",
        default=5,
        type="int",
        help="SmoothGrad 的噪声采样次数 N。每次对输入以 sample_prob 概率随机突变碱基，"
             "计算梯度后取平均。N 越大梯度越平滑但计算成本线性增加 [Default: %default]",
    )
    parser.add_option(
        "--sampleprob",
        dest="sample_prob",
        default=0.875,
        type="float",
        help="SmoothGrad 中每个位置保持原始碱基不变的概率。1-sample_prob 即每个位置被随机突变的概率。"
             "较高的值保留更多原始序列信息 [Default: %default]",
    )
    parser.add_option(
        "--clip_soft",
        dest="clip_soft",
        default=None,
        type="float",
        help="模型中使用的 soft clipping 阈值（论文中的 384）。在训练时对 >384 的值做软截断 "
             "（√(x-384)+384），预测时需逆向还原 [Default: %default]",
    )
    parser.add_option(
        "--no_transform",
        dest="no_transform",
        default=0,
        type="int",
        help="是否跳过逆变换：1=直接使用模型的原始输出梯度，不做 ^(4/3) 和 soft clip 逆变换。"
             "通常仅用于调试或比较 [Default: %default]",
    )
    parser.add_option(
        "--get_preds",
        dest="get_preds",
        default=0,
        type="int",
        help="是否额外存储标量预测值 u（除梯度 𝐬 外）：1=在 HDF5 中写入 preds 数据集 [Default: %default]",
    )
    parser.add_option(
        "--pseudo_qtl",
        dest="pseudo_qtl",
        default=None,
        type="float",
        help="使用预测值分布的哪个分位数作为伪计数 C（pseudo count）。"
             "C 用于 u = log(C + ...) 中，防止低表达基因的梯度 ∂u/∂𝐱 因分母过小而被放大 [Default: %default]",
    )
    parser.add_option(
        "--pseudo_tissue",
        dest="pseudo_tissue",
        default=None,
        type="str",
        help="计算 pseudo count 时用于筛选基因的组织类型。仅该组织的基因参与分位数计算 [Default: %default]",
    )
    parser.add_option(
        "--gene_file",
        dest="gene_file",
        default=None,
        type="str",
        help="基因元数据的 CSV 文件（制表符分隔），至少包含 gene_base 和 tissue 列，"
             "用于将基因映射到组织类型 [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="靶标（track）索引和标签的表格文件。每行对应一个输出 track t∈𝒯，包含 strand（链方向）、"
             "strand_pair（配对链）、scale（训练缩放因子）等列 [Default: %default]",
    )
    parser.add_option(
        "--gcs",
        dest="gcs",
        default=False,
        action="store_true",
        help="输入和输出文件位于 Google Cloud Storage 上。启用后自动下载输入到本地临时目录，"
             "计算完成后上传结果并清理 [Default: %default]",
    )
    (options, args) = parser.parse_args()

    # ========================================================================
    # GCS 模式初始化：创建本地临时目录用于中间计算
    # ========================================================================
    if options.gcs:
        """假设 output_dir 将位于 GCS，先在本地创建临时输出目录"""
        gcs_output_dir = options.out_dir
        temp_dir = tempfile.mkdtemp()  # 创建本地临时目录
        out_dir = temp_dir + "/output_dir"
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        options.out_dir = out_dir

    # ========================================================================
    # 解析命令行参数：支持三种调用模式
    #
    #   args 数量 | 模式           | 参数含义
    #   ----------|----------------|------------------------------------------
    #   3         | 单 worker      | <params> <model> <gene_gtf>
    #   4         | 主调度脚本     | <options_pkl> <params> <model> <gene_gtf>
    #   5         | 多 worker      | <options_pkl> <params> <model> <gene_gtf> <worker_index>
    #
    # options_pkl 包含从主脚本序列化的额外命令行选项，用于覆盖默认值。
    # ========================================================================
    if len(args) == 3:
        # 单 worker 模式：直接使用命令行传入的参数
        params_file = args[0]
        model_folder = args[1]
        genes_gtf_file = args[2]
    elif len(args) == 4:
        # 主脚本模式：从 pickle 文件加载额外选项（覆盖命令行默认值）
        options_pkl_file = args[0]
        params_file = args[1]
        model_folder = args[2]
        genes_gtf_file = args[3]

        # 保存输出目录（load_extra_options 可能会覆盖 options.out_dir）
        out_dir = options.out_dir

        # 加载额外选项
        if options.gcs:
            options_pkl_file = download_rename_inputs(options_pkl_file, temp_dir)
        options = load_extra_options(options_pkl_file, options)
        # 恢复输出目录
        options.out_dir = out_dir

    elif len(args) == 5:
        # 多 worker 模式：通过 worker_index 切分基因列表
        # 例如 4 个 worker 处理 1000 个基因：worker_0→[0,250), worker_1→[250,500), ...
        options_pkl_file = args[0]
        params_file = args[1]
        model_folder = args[2]
        genes_gtf_file = args[3]
        worker_index = int(args[4])

        # 加载选项
        if options.gcs:
            options_pkl_file = download_rename_inputs(options_pkl_file, temp_dir)
        options = load_extra_options(options_pkl_file, options)
        # 每个 worker 有独立的输出子目录 job0, job1, ...
        os.makedirs(options.out_dir, exist_ok=True)
        options.out_dir = "%s/job%d" % (options.out_dir, worker_index)
    else:
        parser.error("必须提供参数文件（params）、模型文件夹（model）和 GTF 文件（gene_gtf）")

    # 确保输出目录存在
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # 将逗号分隔的字符串解析为整数列表
    # folds: 模型副本索引，论文中训练了 4 个独立初始化的 replicate
    # shifts: 序列平移量，用于集成增强
    options.folds = [int(fold) for fold in options.folds.split(",")]
    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #################################################################
    # 阶段 2：从 GCS 下载输入文件到本地
    #
    # 下载的内容包括：
    #   - params_file: 模型参数 JSON（模型架构 + 训练超参数）
    #   - genes_gtf_file: 基因注释 GTF 文件
    #   - model_folder: 训练好的模型权重文件夹
    #   - genome_fasta: 参考基因组（如 hg38.fa）
    #   - targets_file: 靶标 track 元数据表
    #################################################################
    if options.gcs:
        print("Downloading input files from gcs to a local file")
        t0 = time.time()
        params_file = download_rename_inputs(params_file, temp_dir)
        genes_gtf_file = download_rename_inputs(genes_gtf_file, temp_dir)
        model_folder_name = model_folder.split("/")[-1]
        model_folder = download_rename_inputs(
            model_folder, f"{temp_dir}/{model_folder_name}", is_dir=True
        )
        if options.genome_fasta is not None:
            options.genome_fasta = download_rename_inputs(
                options.genome_fasta, temp_dir
            )
        if options.targets_file is not None:
            options.targets_file = download_rename_inputs(
                options.targets_file, temp_dir
            )
        print("Done in {:.1f} seconds".format(time.time() - t0))

    #################################################################
    # 阶段 3：读取模型参数和靶标定义
    #################################################################

    # 读取模型参数 JSON：
    #   params_model: 模型架构参数
    #     - seq_length: 输入序列长度（论文中的 524,288 bp = 524 kb）
    #     - 各卷积层的 kernel size、通道数、dilation rate 等
    #     - transformer/self-attention 层的头数、embedding 维度等
    #   params_train: 训练超参数
    #     - learning_rate, batch_size, optimizer 配置等
    #     - Poisson loss + Multinomial loss 的权重比（论文中 multinomial 权重为 5×）
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    seq_len = params_model["seq_length"]  # 输入序列长度 L = 524,288（论文中的 524 kb）

    # 读取靶标表格 targets_df：
    #   每行对应一个覆盖度 track t∈𝒯（论文中人类有 7,611 个 tracks）
    #   关键列：
    #     - index (第0列): track 的唯一标识符
    #     - strand: 链方向（"+" / "-"），用于按基因链方向筛选匹配的 tracks
    #     - strand_pair: 配对链 track 的标识符（双链建模时使用）
    #     - scale: 训练时对 track 值施加的缩放因子（track_scale），预测时需逆缩放
    if options.targets_file is None:
        parser.error("必须提供靶标表格以正确处理链方向（stranded tracks）。")
    else:
        targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # 构建链方向映射：
    #   orig_new_index: 原始 track ID → 连续整数索引的映射
    #   targets_strand_pair: 每个 track 对应的配对链 track 索引（用于双链建模）
    #   targets_strand_df: 预处理后的链方向 DataFrame
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = np.array(
        [orig_new_index[ti] for ti in targets_df.strand_pair]
    )
    targets_strand_df = targets_prep_strand(targets_df)
    # num_targets = 1 因为在此脚本中，多个 tracks 的预测值会被聚合（平均）为单个标量
    num_targets = 1

    # ========================================================================
    # 加载基因元数据并根据组织类型筛选基因
    #
    # gene_file 包含所有基因的组织归属信息，用于 pseudo count 计算时
    # 按组织筛选基因子集。例如 pseudo_tissue="whole_blood" 时，
    # 仅使用血液特异性基因的预测值来计算 pseudo count 分位数。
    #
    # gene_base 是基因的符号名（如 "TP53"），去除了 Ensembl ID 的版本号后缀。
    # ========================================================================
    tissue_genes = None
    if options.gene_file is not None and options.pseudo_tissue is not None:
        gene_df = pd.read_csv(options.gene_file, sep="\t")
        gene_df = (
            gene_df.query("tissue == '" + str(options.pseudo_tissue) + "'")
            .copy()
            .reset_index(drop=True)
        )
        gene_df = gene_df.drop(columns=["Unnamed: 0"])

        # 获取该组织中所有基因的 gene_base 列表
        tissue_genes = gene_df["gene_base"].values.tolist()

        print("len(tissue_genes) = " + str(len(tissue_genes)))

    #################################################################
    # 阶段 4：加载第一个模型 fold 以获取模型结构参数
    #
    # 此处加载 fold 0 的目的仅是获取模型的结构元参数（非权重），
    # 后续遍历每个 fold 时会重新加载对应 fold 的权重。
    #
    # Borzoi 模型架构概述（论文 "Model" 节）：
    #   Borzoi 基于 Enformer 架构但做了多项简化和增强：
    #
    #   第一阶段 — 卷积塔（Convolution tower）：
    #     7 层卷积 + max pooling 块，逐层将序列长度降采样。
    #     输入 524,288 bp →
    #       第 1 层: 262,144 @ 64 bp 分辨率
    #       第 2 层: 131,072 @ 128 bp 分辨率
    #       ...
    #       第 N 层: 4,096 @ 128 bp 分辨率（= 目标 target_length 的 1/4）
    #     每层使用单次卷积（kernel=5），省略了 Enformer 的残差连接。
    #     使用 max pooling 替代 Enformer 的 attention pooling。
    #
    #   第二阶段 — Self-attention（Transformer）塔：
    #     8 层 self-attention（Enformer 为 11 层，减少以省显存）。
    #     在 128 bp 分辨率（4,096 个位置）上运行，捕获长程依赖。
    #     每个位置的 query 可以关注所有位置的 key，计算复杂度 O(N²)。
    #     论文: "we chose to remain at 128 bp resolution for the
    #     self-attention blocks" 因为 32bp 分辨率会产生 16,384 个位置，
    #     self-attention 的显存需求为 O(16384²)，超出 GPU 能力。
    #
    #   第三阶段 — U-Net 上采样：
    #     从 128 bp 分辨率上采样回 32 bp 分辨率：
    #       128 bp (4,096 pos) → 64 bp (8,192 pos) → 32 bp (16,384 pos)
    #     每次上采样：复制 embedding 向量 → point-wise 卷积对齐通道 →
    #     与卷积塔对应分辨率的中间特征相加 → 可分离卷积（kernel=3）。
    #     论文: "U-net upsampling techniques from the image segmentation
    #     and object detection literature"
    #
    #   输出:
    #     𝐲 ∈ ℝ^{16384×7611}（人类）或 ℝ^{16384×...}（小鼠）
    #     每个 32 bp bin 预测所有 tracks 的覆盖度值。
    #
    # 关键结构参数：
    #   model_stride: 输出 bin 对应的输入 bp 数（降采样步长）。
    #     Borzoi 中固定为 32 bp/bin（最终输出分辨率）。
    #     输入 524,288 bp ÷ 32 bp/bin = 16,384 output bins。
    #
    #   model_crop: 输入序列两端被裁剪的输出 bin 数量。
    #     由于卷积层使用 valid padding（无填充），每层输出比输入略短。
    #     多层累积后，两端的输出 bin 对应的有效感受野不完整。
    #     模型训练时 loss 仅计算中心 196,608 bp（两端各裁剪约 163,840 bp）。
    #     输出起始坐标 = 输入起始 + model_stride × model_crop。
    #
    #   target_length: 模型输出序列的长度（bin 数）= 16,384。
    #     对应 16,384 × 32 bp = 524,288 bp 的基因组区间。
    #
    # SeqNN.restore() 和 build_slice() 说明：
    #   restore(path, head_i): 从 HDF5 文件加载指定 head 的模型权重。
    #     Borzoi 使用 HDF5 格式存储权重（而非 TensorFlow checkpoint）。
    #   build_slice(targets_index, use_ensemble): 构建目标 track 的切片索引。
    #     用于将 targets_df 的 index 映射到模型输出的 track 维度。
    #     此处 use_ensemble=False 因为我们手动遍历 fold。
    #################################################################

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_folder + "/f0c0/model0_best.h5", 0)
    seqnn_model.build_slice(targets_df.index, False)
    # 注意：此处不调用 build_ensemble，因为我们在外层循环中手动遍历 fold
    # seqnn_model.build_ensemble(options.rc, options.shifts)

    model_stride = seqnn_model.model_strides[0]   # 输出步长 S = 32 bp/bin
    model_crop = seqnn_model.target_crops[0]       # 边界裁剪量（输出 bin 数）
    target_length = seqnn_model.target_lengths[0]  # 输出序列长度 L_out = 16,384 bins

    #################################################################
    # 阶段 5：解析 GTF 文件，读取基因列表并确定每个基因的序列窗口
    #
    # GTF（Gene Transfer Format）文件包含基因结构注释：
    #   - 每个基因的染色体位置、链方向
    #   - 每个转录本的外显子-内含子边界坐标
    #   - TSS（转录起始位点）和 TES（转录终止位点）
    #
    # 序列窗口以基因中点（gene midpoint = (TSS+TES)/2）为中心，
    # 向两侧各扩展 seq_len/2 = 262,144 bp，得到 524,288 bp 的输入窗口。
    #################################################################

    # 解析 GTF 文件，构建转录组对象（包含所有基因的外显子-内含子结构）
    transcriptome = bgene.Transcriptome(genes_gtf_file)

    # 打开参考基因组（pysam.Fastafile 支持随机访问任意染色体的任意区间）
    genome_open = pysam.Fastafile(options.genome_fasta)

    # 按基因 ID 排序以确保多 worker 之间的一致性
    gene_list = sorted(transcriptome.genes.keys())
    num_genes = len(gene_list)

    # ========================================================================
    # 多进程模式：按 worker_index 切分基因列表
    #
    # 使用 np.linspace 将 num_genes 均匀划分为 processes 个区间。
    # 例如 num_genes=1000, processes=4:
    #   worker_0: genes[0:250]
    #   worker_1: genes[250:500]
    #   worker_2: genes[500:750]
    #   worker_3: genes[750:1000]
    # ========================================================================
    if options.processes is not None:
        # 确定每个 worker 负责的基因范围
        worker_bounds = np.linspace(0, num_genes, options.processes + 1, dtype="int")
        worker_start = worker_bounds[worker_index]
        worker_end = worker_bounds[worker_index + 1]
        gene_list = [gene_list[gi] for gi in range(worker_start, worker_end)]
        num_genes = len(gene_list)

    print(f"There are {num_genes} genes in the gene list.")

    #################################################################
    # 阶段 5b：为每个基因确定序列提取窗口的基因组坐标
    #
    # 窗口定位策略：
    #   设基因中点为 m = (TSS + TES) / 2，即转录起始和终止的平均位置。
    #   输入窗口起始 = max(min_start, m - L/2)
    #   输入窗口结束 = 窗口起始 + L
    #   其中 L = seq_len = 524,288 bp。
    #
    # min_start = -S × crop 确保窗口起始不会因负坐标而无法提取序列。
    # make_seq_1hot 中会用 N 填充负坐标区域。
    #
    # 同时记录每个基因的：
    #   - chrom: 染色体名称（如 chr1, chrX）
    #   - strand: 链方向（"+" 或 "-"）
    #################################################################

    min_start = -model_stride * model_crop

    # 为每个基因收集坐标和元数据
    genes_chr = []       # 染色体名称列表
    genes_start = []     # 序列窗口起始坐标列表
    genes_end = []       # 序列窗口结束坐标列表
    genes_strand = []    # 链方向列表
    for gene_id in gene_list:
        gene = transcriptome.genes[gene_id]
        genes_chr.append(gene.chrom)
        genes_strand.append(gene.strand)

        gene_midpoint = gene.midpoint()  # 基因中点坐标（TSS 和 TES 的中点）
        # 确保窗口起始不低于 min_start（负数会被 N 填充）
        gene_start = max(min_start, gene_midpoint - seq_len // 2)
        gene_end = gene_start + seq_len
        genes_start.append(gene_start)
        genes_end.append(gene_end)

    #################################################################
    # 阶段 6-7：初始化 HDF5 输出，对每个 fold 计算预测和梯度
    #################################################################

    buffer_size = 1024  # 缓冲区大小：每次 GPU 计算处理的基因数量上限

    print("clip_soft = " + str(options.clip_soft))
    print("n genes = " + str(len(genes_chr)))

    # ═══════════════════════════════════════════════════════════════
    # 遍历每个模型 fold（独立训练的模型副本/replicate）
    #
    # 论文中训练了 4 个独立初始化的模型副本，它们的梯度取平均
    # 可以降低随机初始化和训练顺序带来的方差。
    # ═══════════════════════════════════════════════════════════════
    for fold_ix in options.folds:
        print("-- Fold = " + str(fold_ix) + " --")

        # ====================================================================
        # 初始化/重新创建 HDF5 输出文件
        #
        # HDF5 数据集结构（与论文符号的对应）：
        #   /seqs    : bool   [G, L, 4]          — 原始输入 𝐱（one-hot DNA）
        #   /grads   : float16 [G, L, 4, num_targets] — 梯度显著性 𝐬 = ∂u/∂𝐱
        #   /preds   : float32 [G, num_targets]   — （可选）汇总统计量 u
        #   /gene    : string [G]                  — 基因 ID 列表
        #   /chr     : string [G]                  — 染色体名称
        #   /start   : int    [G]                  — 序列窗口起始坐标
        #   /end     : int    [G]                  — 序列窗口结束坐标
        #   /strand  : string [G]                  — 链方向
        #
        # 其中 G = num_genes, L = seq_len = 524,288。
        #
        # 注意：grads 使用 float16 以节省存储空间（G × 524288 × 4 × 1 可能非常大）。
        # ====================================================================
        scores_h5_file = "%s/scores_f%dc0.h5" % (options.out_dir, fold_ix)
        if os.path.isfile(scores_h5_file):
            os.remove(scores_h5_file)
        scores_h5 = h5py.File(scores_h5_file, "w")
        scores_h5.create_dataset("seqs", dtype="bool", shape=(num_genes, seq_len, 4))
        scores_h5.create_dataset(
            "grads", dtype="float16", shape=(num_genes, seq_len, 4, num_targets)
        )
        if options.get_preds == 1:
            scores_h5.create_dataset(
                "preds", dtype="float32", shape=(num_genes, num_targets)
            )
        scores_h5.create_dataset("gene", data=np.array(gene_list, dtype="S"))
        scores_h5.create_dataset("chr", data=np.array(genes_chr, dtype="S"))
        scores_h5.create_dataset("start", data=np.array(genes_start))
        scores_h5.create_dataset("end", data=np.array(genes_end))
        scores_h5.create_dataset("strand", data=np.array(genes_strand, dtype="S"))

        # 加载当前 fold 的模型权重
        # SeqNN(params_model): 根据 params_model 构建模型计算图（架构）
        # restore(h5_path, head_i): 从 HDF5 加载训练好的权重到模型
        #   "f{fold_ix}c0/model0_best.h5": fold 的 best checkpoint
        #   每个 fold 是独立随机初始化训练的，权重不同
        # build_slice(targets_index, False): 构建 track 索引到模型输出
        #   维度的映射。False 表示不使用集成（手动遍历 fold）。
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_folder + "/f" + str(fold_ix) + "c0/model0_best.h5", 0)
        seqnn_model.build_slice(targets_df.index, False)

        # ═══════════════════════════════════════════════════════════════════
        # 训练数据变换参数（论文 "Training data" 节 — squashed scale）
        #
        # track_scale (α):
        #   训练时对 squashed 值施加的额外缩放因子。
        #   如果训练 target 是 squashed 后再乘以 α，则预测值需要除以 α 还原。
        #
        # track_transform (β = 3/4):
        #   论文中 squashed scale 的核心变换：
        #     𝐲^{(squashed)} = {
        #         𝐲^(3/4)                    if 𝐲^(3/4) ≤ 384
        #         384 + √(𝐲^(3/4) - 384)     otherwise
        #     }
        #   逆变换在 _count_func 中执行：先做 ^(1/β) = ^(4/3)，再撤销 soft clip。
        #
        # clip_soft (θ = 384):
        #   squashed scale 的阈值。训练时，超过 θ 的值被软截断：
        #   超过部分做 √ 变换以限制极端高表达基因对 loss 的贡献。
        #   预测时需逆向操作：对 >θ 的值还原为 (pred - θ)² + θ。
        # ═══════════════════════════════════════════════════════════════════
        track_scale = targets_df.iloc[0]["scale"]
        track_transform = 3.0 / 4.0

        # ====================================================================
        # 阶段 7a：（可选）先计算并存储前向预测值 u
        #
        # 预测值 u 的作用：
        #   1. 后续用于计算 pseudo count C 的分位数
        #   2. 作为基因表达水平的参考值，与梯度一同输出到 HDF5
        #
        # 计算流程：
        #   对每个 (shift, rev_comp) 组合 → 批量前向传播 ℳ(𝐱) → 逆变换还原 →
        #   聚合为标量 u = Σ_b 𝐲_b（不取 log，不包含 pseudo count）
        #
        #   注意此处 u 是在所有 (shift, rev_comp) 组合上累加后取平均，
        #   即 u = mean_{shift, rc} [ Σ_{t∈𝒯} mean_t(Σ_{b∈ℬ} 𝐲_{b,t}) ]
        # ====================================================================
        if options.get_preds == 1:
            print(" - (prediction) - ", flush=True)

            # ---- 遍历每个序列平移增强 ----
            for shift in options.shifts:
                print("Processing shift %d" % shift, flush=True)

                # ---- 遍历正/反向互补链 ----
                # 若 options.rc==1: 遍历 [False(正向), True(反向互补)]
                # 若 options.rc==0: 仅遍历 [False(正向)]
                for rev_comp in [False, True] if options.rc == 1 else [False]:
                    if options.rc == 1:
                        print(
                            "Fwd/rev = %s" % ("fwd" if not rev_comp else "rev"),
                            flush=True,
                        )

                    # 初始化缓冲区：累积 seq_1hot, gene_slices, gene_targets
                    # 达到 buffer_size 后批量 GPU 计算
                    seq_1hots = []
                    gene_slices = []
                    gene_targets = []

                    for gi, gene_id in enumerate(gene_list):
                        if gi % 500 == 0:
                            print("Processing %d, %s" % (gi, gene_id), flush=True)

                        gene = transcriptome.genes[gene_id]

                        # --------------------------------------------------
                        # 步骤 1: 从参考基因组提取基因周围 DNA 序列并 one-hot 编码
                        #
                        # 输入 𝐱 的形状: (L, 4)，L = seq_len = 524,288
                        # 4 个通道分别对应碱基 A(0), C(1), G(2), T(3)
                        # 编码规则: A→[1,0,0,0], C→[0,1,0,0], G→[0,0,1,0], T→[0,0,0,1]
                        # N（未知碱基）→[0,0,0,0]
                        # --------------------------------------------------
                        seq_1hot = make_seq_1hot(
                            genome_open,
                            genes_chr[gi],
                            genes_start[gi],
                            genes_end[gi],
                            seq_len,
                        )
                        # --------------------------------------------------
                        # 步骤 2: 序列平移增强（shift augmentation）
                        #
                        # shift > 0: 序列向左平移（左端截断，右端补 0）
                        # shift < 0: 序列向右平移（右端截断，左端补 0）
                        # 通过在不同平移量上平均梯度，增强模型对位置微小偏移的鲁棒性。
                        # --------------------------------------------------
                        seq_1hot = dna.hot1_augment(seq_1hot, shift=shift)

                        # --------------------------------------------------
                        # 步骤 3: 确定模型输出序列在基因组上的起始位置
                        #
                        # 由于 valid padding（无填充卷积），模型输出比输入短。
                        # 输出起始 = 输入起始 + model_stride × model_crop
                        # 输出长度 = model_stride × target_length (= 32 × 16384 覆盖的 bp 数)
                        #
                        # 例如输入起始为 1,000,000，model_stride=32，model_crop=...:
                        #   输出起始 = 1,000,000 + 32 × crop
                        #   每个输出 bin 覆盖 32 bp，共 16,384 个 bin
                        # --------------------------------------------------
                        seq_out_start = genes_start[gi] + model_stride * model_crop
                        seq_out_len = model_stride * target_length

                        # --------------------------------------------------
                        # 步骤 4: 确定基因外显子在模型输出中的位置切片
                        #
                        # gene.output_slice(...) 返回一维索引数组 ℬ = {b₀,...,b_B}，
                        # 每个 b_k 是输出 bins 中属于该基因外显子的位置索引。
                        # 若 options.span==1，则覆盖整个 gene span 而非仅外显子区域。
                        #
                        # gene_slice 形状: (1, num_exon_bins)
                        # --------------------------------------------------
                        gene_slice = gene.output_slice(
                            seq_out_start, seq_out_len, model_stride, options.span == 1
                        )

                        # --------------------------------------------------
                        # 步骤 5: 反向互补链处理
                        #
                        # 若 rev_comp==True，对输入序列做反向互补变换：
                        #   - 反转序列方向（5'→3' 变为 3'→5'）
                        #   - 交换互补碱基（A↔T, C↔G）
                        # 同时输出位置索引也需要反转。
                        #
                        # 论文中："the gradient computation was repeated for
                        # all four model replicates, for both forward-complemented
                        # and reverse-complemented input sequences, and averaged."
                        # --------------------------------------------------
                        if rev_comp:
                            seq_1hot = dna.hot1_rc(seq_1hot)
                            gene_slice = target_length - gene_slice - 1

                        # --------------------------------------------------
                        # 步骤 6: 根据基因的链方向筛选匹配的靶标 tracks
                        #
                        # 背景：Borzoi 的训练数据是链特异性的 (stranded) —
                        #   RNA-seq 的正链和负链覆盖度是分开的 tracks。
                        #   因此需要根据基因的链方向选择正确的 tracks 子集。
                        #
                        # 逻辑表（4 种情况）：
                        #   Gene strand | rev_comp? | 选择 tracks      | 原因
                        #   ------------|-----------|-----------------|------------------
                        #   +           | No (fwd)  | strand != "-"   | 正链基因用正链 track
                        #   +           | Yes (rev) | strand != "+"   | rev 后变负链，用负链 track
                        #   -           | No (fwd)  | strand != "+"   | 负链基因用负链 track
                        #   -           | Yes (rev) | strand != "-"   | rev 后变正链，用正链 track
                        #
                        # 为什么 rev_comp 会导致链方向翻转？
                        #   正向序列:   5'-ATG...-3'  (原始正链)
                        #   反向互补后: 3'-TAC...-5' → 等价于 5'-CAT...-3' (互补链)
                        #   原本的正链基因在互补链上看起来像负链基因，
                        #   因此需要使用负链 tracks 来正确匹配。
                        #
                        # 注意：targets_df.strand 可能包含 "+", "-", "." (非链特异性)
                        #   使用 != 而非 == 来筛选，使得 "." (非特异性) tracks
                        #   在所有情况下都被包含，最大化可用数据。
                        # --------------------------------------------------
                        if genes_strand[gi] == "+":
                            gene_strand_mask = (
                                (targets_df.strand != "-")
                                if not rev_comp
                                else (targets_df.strand != "+")
                            )
                        else:
                            gene_strand_mask = (
                                (targets_df.strand != "+")
                                if not rev_comp
                                else (targets_df.strand != "-")
                            )

                        gene_target = np.array(
                            targets_df.index[gene_strand_mask].values
                        )

                        # --------------------------------------------------
                        # 步骤 7: 将数据累积到缓冲区
                        #
                        # [None, ...] 操作在 axis=0 添加 batch 维度:
                        #   seq_1hot:    (L, 4) → (1, L, 4)
                        #   gene_slice:  (1, B) → (1, 1, B)  其中 B=|ℬ|
                        #   gene_target: (T,)   → (1, T)      其中 T=|𝒯|
                        # --------------------------------------------------
                        seq_1hots.append(seq_1hot[None, ...])
                        gene_slices.append(gene_slice[None, ...])
                        gene_targets.append(gene_target[None, ...])

                        # --------------------------------------------------
                        # 步骤 8: 当缓冲区满（或处理完所有基因）时，批量 GPU 计算
                        # --------------------------------------------------
                        if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size:
                            # 8a. 沿 batch 维度拼接序列
                            #     结果形状: [K, L, 4]，K = min(buffer_size, 剩余基因数)
                            seq_1hots = np.concatenate(seq_1hots, axis=0)

                            # 8b. 将不同基因的位置切片填充到相同长度
                            #     不同基因的外显子数量不同 → |ℬ| 不同
                            #     gene_masks: 标记哪些 bin 是有效外显子 (1=有效, 0=填充)
                            #     gene_slices_padded: 填充后的索引对齐矩阵
                            #     为什么需要 padding？numpy/TF batch 操作要求
                            #     所有样本在非 batch 维度形状一致。
                            max_slice_len = int(
                                np.max(
                                    [gene_slice.shape[1] for gene_slice in gene_slices]
                                )
                            )

                            gene_masks = np.zeros(
                                (len(gene_slices), max_slice_len), dtype="float32"
                            )
                            gene_slices_padded = np.zeros(
                                (len(gene_slices), max_slice_len), dtype="int32"
                            )
                            for gii, gene_slice in enumerate(gene_slices):
                                for j in range(gene_slice.shape[1]):
                                    gene_masks[gii, j] = 1.0
                                    gene_slices_padded[gii, j] = gene_slice[0, j]

                            gene_slices = gene_slices_padded

                            # 8c. 拼接基因特异的目标 track 索引
                            gene_targets = np.concatenate(gene_targets, axis=0)

                            # 8d. 批量前向传播计算预测值
                            #     preds 形状: [K]（每个基因一个标量预测值）
                            #
                            #     注意与后续 gradients() 调用的区别：
                            #     - 仅前向传播，不做反向传播（无梯度计算）
                            #     - dtype="float32"（预测用全精度，梯度用 float16）
                            #     - 不使用 pseudo_count、smooth_grad、subtract_avg
                            preds = predict_counts(
                                seqnn_model,
                                seq_1hots,
                                head_i=0,
                                target_slice=gene_targets,
                                pos_slice=gene_slices,
                                pos_mask=gene_masks,
                                chunk_size=buffer_size,
                                batch_size=1,
                                track_scale=track_scale,
                                track_transform=track_transform,
                                clip_soft=options.clip_soft,
                                use_mean=False,
                                dtype="float32",
                            )

                            # 8e. 将预测值写入 HDF5（对不同 shift 累加，最后取平均）
                            for gii, gene_slice in enumerate(gene_slices):
                                # 计算全局基因索引（参见梯度循环中 h5_gi 的详细解释）
                                h5_gi = (gi // buffer_size) * buffer_size + gii

                                # 累加并除以 shift 数量 = 对 shift 取平均
                                # preds[gii] 是该基因的标量预测值 u
                                # 除以 len(shifts) 实现 shift 维度的在线平均
                                # 注意：rev_comp 维度的平均通过外层的 rev_comp 循环
                                # 中的 += 累加隐式完成（最后除以 rc 因子）。
                                scores_h5["preds"][h5_gi, :] += preds[gii] / float(
                                    len(options.shifts)
                                )

                            # 8f. 清空缓冲区，触发垃圾回收释放 GPU 显存
                            seq_1hots = []
                            gene_slices = []
                            gene_targets = []

                            gc.collect()

        # ====================================================================
        # 阶段 7b：（可选）根据预测值设置 pseudo count C
        #
        # 论文中的 pseudo count（"with optional pseudo count"）：
        #   u = log( C + (1/T) × Σ_{t∈𝒯} Σ_{b∈ℬ} 𝐲_{b,t} )
        #
        # C 的作用：
        #   当基因表达量很低时，(1/T) Σ_t Σ_b 𝐲_{b,t} 接近 0，
        #   如果没有 C，log(ε) → -∞，导致梯度 ∂u/∂𝐱 数值不稳定/爆炸。
        #   加入 C 后 u = log(C + small_value)，梯度更稳定。
        #
        # C 的计算方式：
        #   取所有（或组织特异性）基因预测值的 options.pseudo_qtl 分位数。
        #   例如 pseudo_qtl=0.25 表示取第 25 百分位数作为 C。
        #
        # 论文中："Replicating the entire analysis with pseudo counts added
        # to the predicted sum of exon coverage before applying log and
        # computing gradients resulted in nearly identical results."
        # 说明 pseudo count 对结果影响不大（但有助于数值稳定性）。
        # ====================================================================
        pseudo_count = 0.0
        if options.pseudo_qtl is not None:
            gene_preds = scores_h5["preds"][:]  # 形状: [G, 1]

            # 根据组织类型筛选基因子集
            tissue_preds = None

            if tissue_genes is not None:
                tissue_set = set(tissue_genes)

                # 仅保留属于目标组织的基因预测值
                tissue_preds = []
                for gi, gene_id in enumerate(gene_list):
                    # gene_id.split(".")[0]: 去掉 Ensembl ID 的版本号后缀
                    # 例如 "ENSG00000141510.17" → "ENSG00000141510"
                    if gene_id.split(".")[0] in tissue_set:
                        tissue_preds.append(gene_preds[gi, 0])

                tissue_preds = np.array(tissue_preds, dtype="float32")
            else:
                tissue_preds = np.array(gene_preds[:, 0], dtype="float32")

            print("tissue_preds.shape[0] = " + str(tissue_preds.shape[0]))
            print("np.min(tissue_preds) = " + str(np.min(tissue_preds)))
            print("np.max(tissue_preds) = " + str(np.max(tissue_preds)))

            # 计算预测值分布的指定分位数作为 pseudo count C
            pseudo_count = np.quantile(tissue_preds, q=options.pseudo_qtl)

            print("")
            print("pseudo_count = " + str(round(pseudo_count, 6)))

        # ====================================================================
        # 阶段 7c：核心 —— 计算梯度显著性得分
        #
        # ⚠️ 注意：以下代码结构与 7a（预测阶段）几乎完全相同。
        # 这不是代码重复，而是必然的设计选择：
        #   - 预测阶段 (7a): 调用 predict_counts → _count_func，仅做前向传播
        #   - 梯度阶段 (7c): 调用 seqnn_model.gradients()，前向+反向传播
        #   两者共享相同的序列预处理管线（提取、one-hot、增强、切片、筛选），
        #   但调用的后端函数不同。拆分是为了可选执行（--get_preds 控制 7a）。
        #
        # 论文公式（"Gradient × input" 节）：
        #   u = log( C + (1/T) × Σ_{t∈𝒯} Σ_{b∈ℬ} 𝐲_{b,t} )
        #   𝐬_{i,j} = ∂u(𝐱)/∂𝐱_{i,j} - (1/4) × Σ_{k=1}^{4} ∂u(𝐱)/∂𝐱_{i,k}
        #
        # seqnn_model.gradients() 内部实现：
        #   1. 前向传播：𝐱 → ℳ(𝐱) = 𝐲
        #   2. 选择目标 tracks 和基因位置
        #   3. 聚合为标量 u（含 pseudo count、逆变换）
        #   4. 反向传播：计算 ∂u/∂𝐱（即输入 𝐱 每个元素的梯度）
        #   5. subtract_avg: 减去每个位置四个碱基通道的均值
        #
        # 梯度 ∂u/∂𝐱_{i,j} 的含义：
        #   在位置 i 将碱基 j 的 one-hot 值微增 δ，
        #   预测表达量 u 预期变化约 δ × ∂u/∂𝐱_{i,j}。
        #   正值 = 该碱基增强表达，负值 = 抑制表达。
        #
        # 集成策略：
        #   对每个 shift（平移量）和每个 rev_comp（链方向）独立计算梯度，
        #   所有结果在 HDF5 中累加，最后除以集成大小得到平均梯度。
        # ═══════════════════════════════════════════════════════════════════
        print(" - (gradients) - ", flush=True)

        for shift in options.shifts:
            print("Processing shift %d" % shift, flush=True)

            for rev_comp in [False, True] if options.rc == 1 else [False]:
                if options.rc == 1:
                    print(
                        "Fwd/rev = %s" % ("fwd" if not rev_comp else "rev"), flush=True
                    )

                # 重置缓冲区
                seq_1hots = []
                gene_slices = []
                gene_targets = []

                for gi, gene_id in enumerate(gene_list):
                    if gi % 500 == 0:
                        print("Processing %d, %s" % (gi, gene_id), flush=True)

                    gene = transcriptome.genes[gene_id]

                    # 步骤 1-6：与预测阶段相同的序列处理流程。
                    # 以下六步的目标是将原始基因组坐标转换为模型可接受的
                    # 张量格式（𝐱, 𝒯, ℬ），并对链方向和序列增强做预处理。
                    #
                    # 管线：基因组坐标 → DNA序列 → one-hot 𝐱 → shift增强
                    #       → 输出bin索引 ℬ → rev_comp → tracks筛选 𝒯

                    # 步骤 1: 从参考基因组提取 DNA 并做 one-hot 编码 → 𝐱
                    seq_1hot = make_seq_1hot(
                        genome_open,
                        genes_chr[gi],
                        genes_start[gi],
                        genes_end[gi],
                        seq_len,
                    )
                    # 步骤 2: 序列平移增强
                    seq_1hot = dna.hot1_augment(seq_1hot, shift=shift)

                    # 步骤 3: 确定输出序列起始位置
                    seq_out_start = genes_start[gi] + model_stride * model_crop
                    seq_out_len = model_stride * target_length

                    # 步骤 4: 确定基因外显子的输出 bin 索引 ℬ
                    gene_slice = gene.output_slice(
                        seq_out_start, seq_out_len, model_stride, options.span == 1
                    )

                    # 步骤 5: 反向互补处理
                    if rev_comp:
                        seq_1hot = dna.hot1_rc(seq_1hot)
                        gene_slice = target_length - gene_slice - 1

                    # 步骤 6: 链方向筛选匹配的 tracks 𝒯
                    if genes_strand[gi] == "+":
                        gene_strand_mask = (
                            (targets_df.strand != "-")
                            if not rev_comp
                            else (targets_df.strand != "+")
                        )
                    else:
                        gene_strand_mask = (
                            (targets_df.strand != "+")
                            if not rev_comp
                            else (targets_df.strand != "-")
                        )

                    gene_target = np.array(targets_df.index[gene_strand_mask].values)

                    # 累积数据到缓冲区
                    seq_1hots.append(seq_1hot[None, ...])
                    gene_slices.append(gene_slice[None, ...])
                    gene_targets.append(gene_target[None, ...])

                    # 缓冲区满或处理完所有基因 → 批量计算梯度
                    if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size:
                        # --- 缓冲区刷新：将累积的 K 个基因的数据打包送入 GPU ---
                        #
                        # 拼接序列 → [K, L, 4]（K = 当前缓冲区中的基因数）
                        seq_1hots = np.concatenate(seq_1hots, axis=0)

                        # 将不同长度的基因外显子切片 ℬ 填充对齐到相同长度 B_max。
                        # 为什么需要 padding？不同基因的外显子数量不同，
                        # 它们对应的输出 bin 索引 ℬ = {b₀,...,b_B} 的长度 B 不同。
                        # 为了在 batch 维度堆叠，需要填充到统一的最大长度 B_max。
                        # gene_masks (1=有效, 0=填充) 确保填充位置不参与后续聚合。
                        max_slice_len = int(
                            np.max([gene_slice.shape[1] for gene_slice in gene_slices])
                        )

                        gene_masks = np.zeros(
                            (len(gene_slices), max_slice_len), dtype="float32"
                        )
                        gene_slices_padded = np.zeros(
                            (len(gene_slices), max_slice_len), dtype="int32"
                        )
                        for gii, gene_slice in enumerate(gene_slices):
                            for j in range(gene_slice.shape[1]):
                                gene_masks[gii, j] = 1.0
                                gene_slices_padded[gii, j] = gene_slice[0, j]

                        gene_slices = gene_slices_padded

                        # 拼接基因特异的 track 索引 𝒯 → [K, T_max]
                        gene_targets = np.concatenate(gene_targets, axis=0)

                        # ════════════════════════════════════════════════════
                        # 【核心调用】seqnn_model.gradients()
                        #
                        # 这是整个脚本的核心：通过 TensorFlow 的自动微分机制
                        # （tf.GradientTape）计算汇总统计量 u 对输入 𝐱 的梯度。
                        #
                        # ════════════════════════════════════════════════════
                        # Forward pass（前向传播）—— 计算标量 u：
                        # ════════════════════════════════════════════════════
                        #
                        #   1. 𝐲 = ℳ(𝐱)                              [K, L_out, N_tracks]
                        #      模型前向传播。Borzoi 架构：
                        #        - 7 层卷积 + 池化塔（逐层降采样至 128bp 分辨率）
                        #        - 8 层 self-attention（transformer）处理长程依赖
                        #        - 2 层 U-Net 上采样（从 128bp → 64bp → 32bp 分辨率）
                        #      输出: 覆盖度预测值（squashed scale 空间）
                        #
                        #   2. 𝐲_selected = 𝐲[:, :, target_slice]     [K, L_out, T]
                        #      按 batch 中每个基因单独选择匹配的 tracks 𝒯
                        #      （batch_dims=1 确保每个样本独立切片）
                        #
                        #   3. 逆变换还原到 count 空间（undo squashed scale）：
                        #      a) 逆缩放:   𝐲_1 = 𝐲_selected / α
                        #      b) 逆 soft clip: 𝐲_2 = { (𝐲_1-θ)²+θ  if 𝐲_1>θ
                        #                                    else 𝐲_1        }
                        #      c) 逆幂变换: 𝐲' = 𝐲_2 ^ (1/β) = 𝐲_2 ^ (4/3)
                        #      注意：这些逆变换都是可微分的，因此梯度可以
                        #      通过链式法则传播回输入 𝐱。
                        #
                        #   4. 𝐲_mean = mean(𝐲', axis=-1)            [K, L_out]
                        #      (1/T) × Σ_{t∈𝒯} —— tracks 维度平均
                        #
                        #   5. 𝐲_pos = 𝐲_mean[:, pos_slice]          [K, B]
                        #      选择基因外显子 ℬ 对应的输出 bin 位置
                        #
                        #   6. 𝐲_masked = 𝐲_pos ⊙ pos_mask           [K, B]
                        #      掩码无效位置（填充补齐的多余位置置零）
                        #
                        #   7. u = log( C + Σ_b 𝐲_masked[:, b] )     [K]
                        #      聚合为标量：∑_{b∈ℬ} 后加伪计数 C，取 log
                        #      论文: u = log( C + (1/T) Σ_t Σ_b 𝐲_{b,t} )
                        #
                        # ════════════════════════════════════════════════════
                        # Backward pass（反向传播）—— 通过自动微分计算 ∂u/∂𝐱：
                        # ════════════════════════════════════════════════════
                        #
                        #   8. TensorFlow 的 tf.GradientTape 自动追踪上述
                        #      所有操作的计算图，然后通过链式法则反向传播：
                        #
                        #      ∂u/∂𝐱 = ∂u/∂𝐲_masked · ∂𝐲_masked/∂𝐲_pos ·
                        #               ∂𝐲_pos/∂𝐲_mean · ∂𝐲_mean/∂𝐲' ·
                        #               ∂𝐲'/∂𝐲_selected · ∂𝐲_selected/∂𝐲 ·
                        #               ∂𝐲/∂𝐱
                        #
                        #      每一项由 TF 自动求导（autograd）计算。
                        #      𝐬_raw = ∂u/∂𝐱                        [K, L, 4]
                        #
                        #   9. 𝐬_{i,j} = ∂u/∂𝐱_{i,j} - (1/4) Σ_k ∂u/∂𝐱_{i,k}
                        #      （subtract_avg=True: 减去每个位置四个碱基通道的均值）
                        #      论文: 𝐬_{i,j} = ∂u/∂𝐱_{i,j} - (1/4) Σ_k ∂u/∂𝐱_{i,k}
                        #
                        #      为什么减去均值？因为 one-hot 编码的四个通道
                        #      之和恒为 1（或 0），存在一个自由度。减去均值
                        #      可以消除与碱基无关的基线偏移，使归因更清晰。
                        #
                        # ════════════════════════════════════════════════════
                        # 参数详解（按论文符号体系）：
                        # ════════════════════════════════════════════════════
                        #
                        # 【模型选择】
                        #   head_i=0: Borzoi 的物种特异性输出头索引。
                        #     Borzoi 有多个输出头（head），每个头对应不同物种
                        #     或任务。head_i=0 通常是人类 (hg38) 的预测头。
                        #     论文: "a species-specific head attached to the
                        #     shared model trunk"
                        #
                        # 【数据流控制】
                        #   batch_size=1: 每次反向传播处理的样本数。设为 1 是因
                        #     为每个基因的 target_slice (𝒯 大小) 和 pos_slice
                        #     (ℬ 大小) 可能不同，无法简单堆叠为统一形状的 batch。
                        #
                        #   chunk_size: 每次加载到 GPU 的最大样本数。
                        #     SmoothGrad 时需要除以 n_samples，因为每个原始
                        #     样本会被复制 N 次（N 个噪声版本），显存消耗放大
                        #     N 倍。设置 chunk_size = buffer_size // N 可以
                        #     将 GPU 显存峰值控制在同等水平。
                        #
                        #   dtype="float16": 梯度以半精度 (float16) 存储。
                        #     每个梯度的形状是 [L, 4] = [524288, 4]，一个基因
                        #     就需要 524288×4×2 = 4MB (fp16) vs 8MB (fp32)。
                        #     对于数万个基因，使用 fp16 可节省约一半存储空间。
                        #     梯度值本身范围较小（>1e-4），fp16 精度足够。
                        #
                        # 【squashed scale 逆变换参数】（见 _count_func 详解）
                        #   track_scale (α): 训练缩放因子
                        #   track_transform (β): 幂变换指数，论文中为 3/4
                        #   clip_soft (θ): soft clipping 阈值，论文中为 384
                        #   no_transform=False: 设为 True 则跳过所有逆变换，
                        #     直接在 squashed scale 空间计算梯度。此时 u 不
                        #     等于论文中的 count 空间表达量，但梯度图可能更平滑。
                        #
                        # 【汇总统计量 u 的变体选择】
                        #   此脚本固定使用 Expression attribution：
                        #     u = log( C + (1/T) Σ_t Σ_b 𝐲_{b,t} )
                        #
                        #   以下参数切换到论文中其他两种归因模式（本脚本未用）：
                        #
                        #   use_ratio=False: 若为 True，使用 splicing attribution
                        #     论文: u = log( (C+Σ_exon)/(C+Σ_intron) )
                        #     即外显子与内含子覆盖度的 log ratio。
                        #
                        #   use_logodds=False: 若为 True，使用 polyadenylation
                        #     attribution
                        #     论文: u = log( (C+Σ_proximal)/(C+Σ_distal) )
                        #     即近端 PAS 与远端 PAS 覆盖度的 log odds。
                        #
                        #   use_mean=False: 位置聚合方式。
                        #     False → Σ_b（求和，用于总表达量）
                        #     True  → (1/|ℬ|) Σ_b（均值，用于平均覆盖度）
                        #
                        #   pseudo_count (C): 伪计数，u = log(C + Σ)。
                        #     防止低表达基因的 ∂log(ε)/∂𝐱 → 1/ε → ∞ 导致梯度爆炸。
                        #
                        # 【梯度后处理】
                        #   subtract_avg=True: 减去每个位置四个碱基通道的均值。
                        #     论文标准做法（见公式 s_{i,j} 定义）。
                        #
                        #   input_gate=False: 若为 True，则额外乘以输入 𝐱：
                        #     𝐬_{i,j} = 𝐬_{i,j} × 𝐱_{i,j}
                        #     仅保留参考碱基对应通道的梯度（其他通道归零）。
                        #     论文中此操作在可视化阶段做（s_i^{(vis)}）。
                        #     此处设为 False 以保留全部四个通道的信息。
                        #
                        # 【SmoothGrad 降噪】（可选）
                        #   smooth_grad=True: 对输入 𝐱 添加随机噪声（以
                        #     sample_prob 概率保留原始碱基，否则随机突变为
                        #     其他三种碱基之一），生成 N=n_samples 个噪声版本，
                        #     分别计算梯度后取平均。
                        #     数学上近似于对梯度场做高斯平滑，可有效抑制
                        #     one-hot 编码不连续性导致的高频噪声。
                        #
                        #   n_samples (N): 噪声采样次数，N 越大越平滑。
                        #   sample_prob: 每个位置保留原始碱基的概率。
                        #     例如 0.875 表示每个位置有 12.5% 概率被突变。
                        # ════════════════════════════════════════════════════
                        grads = seqnn_model.gradients(
                            seq_1hots,
                            head_i=0,
                            target_slice=gene_targets,
                            pos_slice=gene_slices,
                            pos_mask=gene_masks,
                            chunk_size=buffer_size
                            if options.smooth_grad != 1
                            else buffer_size // options.n_samples,
                            batch_size=1,
                            track_scale=track_scale,
                            track_transform=track_transform,
                            clip_soft=options.clip_soft,
                            pseudo_count=pseudo_count,
                            no_transform=options.no_transform == 1,
                            use_mean=False,
                            use_ratio=False,
                            use_logodds=False,
                            subtract_avg=True,
                            input_gate=False,
                            smooth_grad=options.smooth_grad == 1,
                            n_samples=options.n_samples,
                            sample_prob=options.sample_prob,
                            dtype="float16",
                        )

                        # ════════════════════════════════════════════════════
                        # 撤销序列增强操作（unaugment_grads）
                        #
                        # 梯度 𝐬 = ∂u/∂𝐱 是在增强后的序列上计算的
                        # （可能做了 rev_comp 和 shift），需要还原到
                        # 原始基因组坐标 𝐱_original 以保持空间对齐：
                        #
                        #   若 rev_comp=True: 𝐱 = rev_comp(𝐱_original)
                        #     → grad = ∂u/∂(rev_comp(𝐱_original))
                        #     → 需要映射回 ∂u/∂𝐱_original 的空间：
                        #       (A↔T, C↔G, 反转序列方向)
                        #
                        #   若 shift≠0: 𝐱 = shift(𝐱_original)
                        #     → grad = ∂u/∂(shift(𝐱_original))
                        #     → 需要平移回 𝐱_original 的空间：
                        #       (shift>0: 右移; shift<0: 左移; 空位补0)
                        #
                        # grads[gii, :, :, None] 的形状：
                        #   输入: [L, 4] → 加 None → [L, 4, 1]
                        #   匹配 unaugment_grads 的期望 [seq_len, 4, num_targets]
                        # ════════════════════════════════════════════════════
                        for gii, gene_slice in enumerate(gene_slices):
                            grad = unaugment_grads(
                                grads[gii, :, :, None],  # [L, 4] → [L, 4, 1]
                                fwdrc=(not rev_comp),     # True=正向链，不需还原
                                shift=shift,
                            )

                            # 计算该基因在 HDF5 中的全局索引：
                            #   gi 是当前处理的最后一个基因在 gene_list 中的索引
                            #   gi // buffer_size 是已完成的完整 buffer 批次号
                            #   h5_gi 定位到当前 buffer 批次中的第 gii 个基因
                            # 例如 gi=1200, buffer_size=1024, gii=200：
                            #   h5_gi = (1200//1024)*1024 + 200 = 1*1024+200 = 1224
                            h5_gi = (gi // buffer_size) * buffer_size + gii

                            # 累加到 HDF5（不同 (shift, rev_comp) 组件的梯度累加）
                            # 此操作是 += 而非 =，因为每个基因可能被处理多次
                            # （每个 shift×rev_comp 组合一次），需要累加后取平均。
                            scores_h5["grads"][h5_gi] += grad

                        # 清空缓冲区
                        seq_1hots = []
                        gene_slices = []
                        gene_targets = []

                        # 手动触发垃圾回收以释放 GPU 显存
                        gc.collect()

        # ====================================================================
        # 阶段 8：保存原始序列并归一化梯度
        #
        # ⚠️ 归一化时机说明：
        #   在阶段 7c 的循环中，不同 (shift, rev_comp) 组件的梯度通过 +=
        #   累加到 HDF5。此处除以集成组件数 N_ensemble 得到平均梯度。
        #
        #   N_ensemble = len(shifts) × (2 if rc==1 else 1)
        #
        #   例如 len(shifts)=3 (0,1,-1) 且 rc=1 (双方向):
        #     N_ensemble = 3 × 2 = 6
        #     每个基因的最终梯度 𝐬 = (1/6) × Σ_{s∈{0,1,-1}} Σ_{r∈{fwd,rev}} grad_{s,r}
        #
        #   注意：此处仅归一化当前 fold 内的集成组件。
        #   跨 fold 的集成需要在后续分析中将多个 HDF5 文件的梯度取平均。
        #   论文："the gradient computation was repeated for all four model
        #   replicates, for both forward-complemented and reverse-complemented
        #   input sequences, and averaged."
        #
        #   本脚本中每个 fold 独立输出一个 HDF5 文件 (scores_f{fold}c0.h5)，
        #   跨 fold 的平均由下游分析脚本完成。
        #
        # 最终 HDF5 中的 grads[gi] = 𝐬_{i,j}（按论文符号），
        # 形状为 [seq_len, 4, 1] = [524288, 4, 1]。
        # ====================================================================
        for gi, gene_id in enumerate(gene_list):
            # 重新提取原始序列 𝐱（不含 shift 或 rev_comp 增强变换）
            seq_1hot = make_seq_1hot(
                genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len
            )

            # 写入 HDF5
            scores_h5["seqs"][gi] = seq_1hot

            # 梯度归一化：除以当前 fold 的集成组件总数
            #   每个 grads[gi] 在双重循环 (shift × rev_comp) 中被累加了
            #   N_ensemble 次，此处除以 N_ensemble 得到算术平均。
            #
            #   数学上：
            #     grads[gi] ← (1/N_ensemble) × Σ_{shift} Σ_{rev_comp} grad
            #
            #   这等价于对增强变换的期望取平均，降低了单次计算的方差。
            scores_h5["grads"][gi] /= float(
                (len(options.shifts) * (2 if options.rc == 1 else 1))
            )

        # 手动触发垃圾回收
        gc.collect()

    # ========================================================================
    # 阶段 9：关闭文件，上传结果到 GCS
    # ========================================================================
    genome_open.close()
    scores_h5.close()

    # 如果输出目录在 GCS，同步上传并清理临时目录
    if options.gcs:
        upload_folder_gcs(options.out_dir, gcs_output_dir)
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)  # 清理本地临时目录


def unaugment_grads(grads, fwdrc=False, shift=0):
    """
    【撤销序列增强操作】—— 将增强后序列上的梯度还原到原始基因组坐标空间。

    论文中的梯度计算在增强后的序列上进行（正向/反向互补链、不同平移量），
    但最终需要将梯度还原到原始序列 𝐱 的坐标，以确保空间对齐。

    参数:
        grads: 梯度张量 𝐬，形状 (L, 4, num_targets)
               - 轴 0 (L): 序列长度方向 (seq_len = 524,288)
               - 轴 1 (4): 碱基通道 (A=0, C=1, G=2, T=3)
               - 轴 2 (num_targets): 靶标 track 维度（此处为 1）
        fwdrc: 是否为正向链方向。
               True (正向, 未做反向互补) → 不需要还原反向互补变换
               False (做了反向互补) → 需要还原
        shift: 序列平移量。
               shift > 0: 原始序列向左平移
               shift < 0: 原始序列向右平移

    返回:
        还原后的梯度张量，形状同输入 (L, 4, num_targets)

    变换还原的数学逻辑：
    --------------------
    1. 反向互补（reverse complement）还原：
       若 fwdrc=False（即做了 rev_comp），则：
         - 反转序列方向：grads[i] → grads[L-1-i]
         - 交换互补碱基通道：A(0)↔T(3), C(1)↔G(2)

       原因：rev_comp 操作将 𝐱 变为 rev_comp(𝐱)，使得
       梯度 ∂u(rev_comp(𝐱))/∂𝐱 的空间位置和碱基通道与
       原始 𝐱 不对齐。通过上述逆操作恢复对齐。

    2. 平移（shift）还原：
       若 shift > 0（原始序列左移了 shift 个位置）：
         梯度需向右平移：grads[0:L-shift] ← grads[shift:L]
         右侧空出的 shift 个位置填 0

       若 shift < 0（原始序列右移了 |shift| 个位置）：
         梯度需向左平移：grads[|shift|:L] ← grads[0:L-|shift|]
         左侧空出的 |shift| 个位置填 0

       原因：shift 改变了序列在基因组上的对齐位置，
       计算出的梯度在 shifted 坐标空间中，需要还原。
    """
    # ---- 步骤 1: 撤销反向互补变换 ----
    if not fwdrc:
        # 1a: 反转序列方向
        #     grads[i, :, :] ← grads[L-1-i, :, :] 对所有 i
        grads = grads[::-1, :, :]

        # 1b: 交换互补碱基通道
        #     A(0) ↔ T(3)
        grads[:, [0, 3], :] = grads[:, [3, 0], :]
        #     C(1) ↔ G(2)
        grads[:, [1, 2], :] = grads[:, [2, 1], :]

    # ---- 步骤 2: 撤销序列平移变换 ----
    if shift < 0:
        # shift < 0: 原始序列向右平移 |shift| 个位置
        #   还原：梯度向左平移
        #   尾部 |shift| 个位置的梯度来自头部（被截断的原始左侧）
        grads[-shift:, :, :] = grads[:shift, :, :]
        #   头部剩余位置填 0（原始序列右移后左侧的新位置，无信息）
        grads[:-shift, :, :] = 0

    elif shift > 0:
        # shift > 0: 原始序列向左平移 shift 个位置
        #   还原：梯度向右平移
        #   头部 L-shift 个位置的梯度来自尾部（被截断的原始右侧）
        grads[:-shift, :, :] = grads[shift:, :, :]
        #   尾部剩余位置填 0（原始序列左移后右侧的新位置，无信息）
        grads[-shift:, :, :] = 0

    return grads


def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    """
    【生成 one-hot 编码的 DNA 序列】—— 从参考基因组提取 DNA 并编码为 𝐱 ∈ {0,1}^{L×4}。

    对应论文中的输入序列 𝐱 ∈ {0,1}^{524288×4}，4 个通道分别代表 A, C, G, T。

    参数:
        genome_open: pysam.Fastafile 句柄，已打开的参考基因组文件
        chrm: 染色体名称（如 'chr1', 'chrX', 'chr17'）
        start: 序列起始坐标（0-based，包含）
        end: 序列结束坐标（0-based，不包含），end - start = seq_len
        seq_len: 目标序列长度 L = 524,288

    返回:
        seq_1hot: 形状 (L, 4)，dtype=bool/float
                  编码规则（由 dna.dna_1hot 定义）：
                    A → [1,0,0,0]
                    C → [0,1,0,0]
                    G → [0,0,1,0]
                    T → [0,0,0,1]
                    N（未知/模糊碱基）→ [0,0,0,0]

    边界处理：
    ---------
    - 若 start < 0（窗口超出染色体左边界），左侧用 N 填充
    - 若提取的序列短于 seq_len（超出染色体右边界），右侧用 N 填充
    - N 编码为全零向量，表示"不确定"，模型在该位置的信息为零
    """
    # 处理左边界溢出
    if start < 0:
        # start < 0 意味着窗口超出染色体左边界：用 N 填充 -start 个位置
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # 处理右边界不足：用 N 填充至目标长度 L
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    # 碱基字符串 → one-hot 编码矩阵 𝐱 ∈ {0,1}^{L×4}
    seq_1hot = dna.dna_1hot(seq_dna)
    return seq_1hot


# ============================================================================
# _count_func: TensorFlow 图函数 —— 计算原始表达量的前向预测值
#
# @tf.function 装饰器：
#   将 Python 函数编译为静态 TensorFlow 计算图（graph）。
#   第一次调用时触发"图追踪"（tracing），后续调用直接执行编译后的图，
#   避免 Python 解释器开销，在 GPU 上高效运行。
#
#   ⚠️ 关键限制：由于使用了 @tf.function，此函数内部不能有
#   Python 副作用（如 print）。所有操作必须是 TF ops。
#
# 该函数实现论文中 Expression attribution 的 u 值计算
# （不含 log 和 pseudo count，这些在 gradients() 内部添加）：
#
#   原始 u_before_log = (1/T) × Σ_{t∈𝒯} Σ_{b∈ℬ} 𝐲_{b,t}
#
# 其中 𝐲 是经过逆变换还原到 count 空间的预测值。
#
# 与 seqnn_model.gradients() 的关系：
#   _count_func 仅做前向传播，返回标量预测值（用于 preds 存储和
#   pseudo count 计算）。seqnn_model.gradients() 内部执行相同的
#   前向传播逻辑 + 自动微分反向传播。两者共享相同的逆变换步骤。
# ============================================================================
@tf.function
def _count_func(
    model,
    seq_1hot,
    target_slice,
    pos_slice,
    pos_mask=None,
    track_scale=1.0,
    track_transform=1.0,
    clip_soft=None,
    use_mean=False,
):
    """
    【TensorFlow 图函数】—— 计算标量预测值 u（不含 log 和 pseudo count）。

    张量维度流转（按论文符号标注）：
    --------------------------------
    输入:
      seq_1hot      : tf.float32 [K, L, 4]
                      K=batch_size, L=seq_len=524288
                      即论文中的输入 𝐱（mini-batch）
      target_slice  : tf.int32   [K, T_indices]
                      每个样本选择哪些 tracks t∈𝒯 参与聚合
                      论文中的 𝒯 = {t₀,...,t_T}
      pos_slice     : tf.int32   [K, B_padded]
                      每个样本选择输出序列的哪些 bins b∈ℬ
                      论文中的 ℬ = {b₀,...,b_B}
      pos_mask      : tf.float32 [K, B_padded] 或 None
                      位置有效性掩码（1=有效外显子bin, 0=填充）
      track_scale   : tf.float32 scalar — 训练缩放因子 α
      track_transform: tf.float32 scalar — 幂变换指数 β (论文中为 3/4)
      clip_soft     : tf.float32 scalar 或 None — squashed scale 阈值 θ (论文中为 384)
      use_mean      : bool — 位置聚合时是否取平均（默认求和）

    中间步骤及维度变化:
      1. model(seq_1hot, training=False)
         → 形状 [K, L_out, N_all_tracks]
         论文：ℳ(𝐱) = 𝐲，L_out = 16,384，N_all_tracks = 7,611 (human)

      2. tf.gather(..., target_slice, axis=-1, batch_dims=1)
         → 形状 [K, L_out, T]
         选择 𝒯 中的 tracks，T = |𝒯| = gene_target 的大小

      3. preds / track_scale
         → 形状 [K, L_out, T]
         逆缩放：撤销训练时的 α 倍缩放

      4. tf.where(preds > clip_soft, (preds-clip_soft)²+clip_soft, preds)
         → 形状 [K, L_out, T]
         逆 soft clipping：撤销训练时的 squashed scale 变换
         训练时: 若 y^(3/4) > 384 → 384 + √(y^(3/4)-384)
         预测逆: 若 pred > 384 → (pred-384)² + 384

      5. preds ** (1.0 / track_transform)
         → 形状 [K, L_out, T]
         逆幂变换：撤销 ^(3/4)，即做 ^(4/3)

      6. tf.reduce_mean(preds, axis=-1)
         → 形状 [K, L_out]
         在 tracks 维度上做平均：(1/T) × Σ_{t∈𝒯}

      7. tf.gather(preds, pos_slice, axis=-1, batch_dims=1)
         → 形状 [K, B_padded]
         选择外显子位置 bins ℬ

      8. preds_slice * pos_mask
         → 形状 [K, B_padded]
         掩码填充位置（将其值置零）

      9. tf.reduce_sum(preds_slice, axis=-1) (或 reduce_mean)
         → 形状 [K]
         聚合为标量：Σ_{b∈ℬ}（每个基因一个值）

    输出:
      preds_agg: tf.float32 [K] — 每个基因的标量预测值
                 (1/T) × Σ_{t∈𝒯} Σ_{b∈ℬ} 𝐲_{b,t}
    """
    # 步骤 1: 模型前向传播 ℳ(𝐱) → 𝐲
    #   training=False: 使用推理模式（BatchNorm 用移动平均统计量，Dropout 关闭）
    #   输出形状: [K, L_out, N_all_tracks]
    #   论文中 L_out=16384, N_all_tracks=7611 (human)
    preds = tf.gather(
        model(seq_1hot, training=False), target_slice, axis=-1, batch_dims=1
    )
    # tf.gather 在最后一维（tracks 维）按 target_slice 索引选择
    # batch_dims=1: 第一个维度是 batch，对每个样本独立沿 axis=-1 切片
    # 输出形状: [K, L_out, T]（T = target_slice 中每行不等长，但此处已广播对齐）

    # 步骤 2: 逆缩放 —— 撤销训练时的 track_scale (α) 因子
    #   训练: y_train = α × y_squashed
    #   预测: y_unsquashed ≈ preds / α
    preds = preds / track_scale

    # 步骤 3: 逆 soft clipping —— 撤销训练时的 squashed scale 软截断
    #   训练时论文公式（对每个 bin j, track t）:
    #     y^{(squashed)}_{j,t} = {
    #         y_{j,t}^(3/4)                      if y_{j,t}^(3/4) ≤ 384
    #         384 + √(y_{j,t}^(3/4) - 384)       otherwise
    #     }
    #   预测时逆向:
    #     对 >θ 的值，做 (pred - θ)² + θ 还原（近似，非精确逆）
    #     因为 √ 的逆是平方，但训练时的 (x-θ)+θ 结构使逆变换略有偏差
    if clip_soft is not None:
        preds = tf.where(preds > clip_soft, (preds - clip_soft) ** 2 + clip_soft, preds)

    # 步骤 4: 逆幂变换 —— 撤销 ^(3/4)
    #   训练: y_squashed = y^(3/4)
    #   预测: y_restored = preds^(4/3)
    preds = preds ** (1.0 / track_transform)

    # 步骤 5: 在 tracks 维度上平均聚合
    #   (1/T) × Σ_{t∈𝒯} → 将多个 tracks 的预测合并为一个信号
    #   形状变化: [K, L_out, T] → [K, L_out]
    preds = tf.reduce_mean(preds, axis=-1)

    # 步骤 6: 根据 pos_slice 选择基因外显子相关的输出位置
    #   pos_slice[b] 是输出 bin 索引，指示哪些 bins 属于该基因的外显子 ℬ
    #   形状变化: [K, L_out] → [K, B_padded]
    preds_slice = tf.gather(preds, pos_slice, axis=-1, batch_dims=1)

    # 步骤 7: 用掩码将填充位置的值置零
    #   pos_mask = 1（有效外显子位置），0（填充补齐位置）
    #   逐元素乘法：填充位置的贡献 = 0 × preds_slice = 0
    if pos_mask is not None:
        preds_slice = preds_slice * pos_mask

    # 步骤 8: 在位置维度上聚合得到最终标量预测值
    #   use_mean=False（默认）：Σ_{b∈ℬ}（总表达量）
    #   use_mean=True：均值（每个 bin 的平均表达量）
    if not use_mean:
        # 求和聚合：将基因所有外显子位置的表达量相加
        # [K, B_padded] → [K]
        preds_agg = tf.reduce_sum(preds_slice, axis=-1)
    else:
        if pos_mask is not None:
            # 加权平均：总和 / 有效位置数（mask 的求和）
            preds_agg = tf.reduce_sum(preds_slice, axis=-1) / tf.reduce_sum(
                pos_mask, axis=-1
            )
        else:
            # 简单算术平均
            preds_agg = tf.reduce_mean(preds_slice, axis=-1)

    return preds_agg


# ============================================================================
# predict_counts: CPU/GPU 桥接函数 —— 从一批输入序列获取模型预测值
#
# 该函数是 _count_func 的包装器，负责：
#   1. 数据类型转换和形状验证（numpy → TensorFlow tensor）
#   2. 超参数转换为 TF 常量（避免每次调用时的重复转换开销）
#   3. 分 chunk 和分 batch 处理（以控制 GPU 显存使用）
#   4. 调用 _count_func 获取预测值（TF 张量），再转回 numpy 数组
#   5. 拼接所有 chunk/batch 的结果
#
# 分块策略（chunk + batch 两层循环）的设计原因：
#   - 外层 (chunk): 将大数据集拆分为 GPU 显存可容纳的块。
#     每个 chunk 内的样本共享 GPU 显存，但不同 chunk 之间可以
#     释放显存（配合 gc.collect()）。
#   - 内层 (batch): 每个基因的 target_slice (𝒯) 和 pos_slice (ℬ)
#     长度不同，无法在 batch 维度直接堆叠。因此 batch_size=1，
#     每次仅处理一个基因，逐个调用 _count_func。
#
#   这种双层循环虽然降低了吞吐量，但保证了正确性（每个基因
#   有不同的输出切片形状），并且通过 chunk 层的显存管理避免 OOM。
#
# 该函数本身不做梯度计算，仅做前向传播获取标量预测值 u。
# ============================================================================
def predict_counts(
    seqnn_model,
    seq_1hot,
    head_i=None,
    target_slice=None,
    pos_slice=None,
    pos_mask=None,
    chunk_size=None,
    batch_size=1,
    track_scale=1.0,
    track_transform=1.0,
    clip_soft=None,
    use_mean=False,
    dtype="float32",
):
    """
    【CPU/GPU 桥接函数】—— 从一批输入序列获取标量前向预测值。

    该函数仅做前向传播（不计算梯度），用于：
      - 在梯度计算前获取基因表达量的估计值
      - 计算 pseudo count C 的分位数

    参数:
        seqnn_model: Borzoi SeqNN 模型对象
        seq_1hot: numpy [batch_size, L, 4] — 输入 𝐱
        head_i: 模型输出头索引（多头模型时指定）
        target_slice: numpy [batch_size, T] — tracks 索引 𝒯
        pos_slice: numpy [batch_size, B_padded] — 位置 bin 索引 ℬ
        pos_mask: numpy [batch_size, B_padded] 或 None — 位置掩码
        chunk_size: 每次 GPU 计算的最大样本数（控制显存）
        batch_size: 每次 _count_func 调用的样本数（通常为 1，因每个基因的 T 和 B 不同）
        track_scale: float — 训练缩放因子 α
        track_transform: float — 幂变换指数 β（论文中为 3/4）
        clip_soft: float 或 None — soft clipping 阈值 θ（论文中为 384）
        use_mean: bool — 位置聚合是否取平均
        dtype: str — 输出数据类型

    返回:
        preds: numpy [batch_size] 或 [batch_size, 1]
               每个基因的标量预测值 u = (1/T) × Σ_{t∈𝒯} Σ_{b∈ℬ} 𝐲_{b,t}
    """
    # 计时开始
    t0 = time.time()

    # ---- 选择模型 ----
    # 优先级：ensemble > 指定 head > 单模型
    # ensemble 是多个 fold 的集成模型
    if seqnn_model.ensemble is not None:
        model = seqnn_model.ensemble
    elif head_i is not None:
        model = seqnn_model.models[head_i]
    else:
        model = seqnn_model.model

    # ---- 数据类型转换 ----
    # 确保输入 numpy 数组的类型正确
    seq_1hot = seq_1hot.astype("float32")
    target_slice = np.array(target_slice).astype("int32")
    pos_slice = np.array(pos_slice).astype("int32")

    # 将 Python 标量转换为 TensorFlow 常量张量
    # 好处：TF 常量在 GPU 上只需传输一次，后续调用可直接复用
    track_scale = tf.constant(track_scale, dtype=tf.float32)
    track_transform = tf.constant(track_transform, dtype=tf.float32)
    if clip_soft is not None:
        clip_soft = tf.constant(clip_soft, dtype=tf.float32)

    if pos_mask is not None:
        pos_mask = np.array(pos_mask).astype("float32")

    # ---- 确保输入至少是 2D/3D（添加 batch 维度如果缺失） ----
    # seq_1hot:   期望 [batch, L, 4] → 若 [L, 4] 则扩展为 [1, L, 4]
    # target_slice: 期望 [batch, T] → 若 [T] 则扩展为 [1, T]
    # pos_slice:    期望 [batch, B] → 若 [B] 则扩展为 [1, B]
    if len(seq_1hot.shape) < 3:
        seq_1hot = seq_1hot[None, ...]

    if len(target_slice.shape) < 2:
        target_slice = target_slice[None, ...]

    if len(pos_slice.shape) < 2:
        pos_slice = pos_slice[None, ...]

    if pos_mask is not None and len(pos_mask.shape) < 2:
        pos_mask = pos_mask[None, ...]

    # ---- 分块参数 ----
    # chunk_size 控制一次加载到 GPU 的样本数上限
    num_chunks = 1
    if chunk_size is None:
        chunk_size = seq_1hot.shape[0]
    else:
        num_chunks = int(np.ceil(seq_1hot.shape[0] / chunk_size))

    # ---- 遍历每个 chunk ----
    pred_chunks = []
    for ci in range(num_chunks):
        # 从大数组中切出当前 chunk
        seq_1hot_chunk = seq_1hot[ci * chunk_size : (ci + 1) * chunk_size, ...]
        target_slice_chunk = target_slice[ci * chunk_size : (ci + 1) * chunk_size, ...]
        pos_slice_chunk = pos_slice[ci * chunk_size : (ci + 1) * chunk_size, ...]

        pos_mask_chunk = None
        if pos_mask is not None:
            pos_mask_chunk = pos_mask[ci * chunk_size : (ci + 1) * chunk_size, ...]

        actual_chunk_size = seq_1hot_chunk.shape[0]

        # numpy → TensorFlow tensor（触发 CPU→GPU 数据传输）
        seq_1hot_chunk = tf.convert_to_tensor(seq_1hot_chunk, dtype=tf.float32)
        target_slice_chunk = tf.convert_to_tensor(target_slice_chunk, dtype=tf.int32)
        pos_slice_chunk = tf.convert_to_tensor(pos_slice_chunk, dtype=tf.int32)

        if pos_mask is not None:
            pos_mask_chunk = tf.convert_to_tensor(pos_mask_chunk, dtype=tf.float32)

        # ---- 分批参数 ----
        # batch_size 通常为 1，因为每个基因可能有不同的 T=|𝒯| 和 B=|ℬ|
        num_batches = int(np.ceil(actual_chunk_size / batch_size))

        # ---- 遍历每个 batch ----
        pred_batches = []
        for bi in range(num_batches):
            # 切出当前 batch
            seq_1hot_batch = seq_1hot_chunk[
                bi * batch_size : (bi + 1) * batch_size, ...
            ]
            target_slice_batch = target_slice_chunk[
                bi * batch_size : (bi + 1) * batch_size, ...
            ]
            pos_slice_batch = pos_slice_chunk[
                bi * batch_size : (bi + 1) * batch_size, ...
            ]

            pos_mask_batch = None
            if pos_mask is not None:
                pos_mask_batch = pos_mask_chunk[
                    bi * batch_size : (bi + 1) * batch_size, ...
                ]

            # 调用 _count_func（TF 图函数）做前向传播
            # .numpy(): 将 GPU TF 张量拷贝回 CPU 内存为 numpy 数组
            pred_batch = (
                _count_func(
                    model,
                    seq_1hot_batch,
                    target_slice_batch,
                    pos_slice_batch,
                    pos_mask_batch,
                    track_scale,
                    track_transform,
                    clip_soft,
                    use_mean,
                )
                .numpy()
                .astype(dtype)
            )

            pred_batches.append(pred_batch)

        # 拼接当前 chunk 的所有 batch → [chunk_size]
        preds = np.concatenate(pred_batches, axis=0)

        pred_chunks.append(preds)

        # 手动触发垃圾回收
        gc.collect()

    # 拼接所有 chunk → [total_batch_size]
    preds = np.concatenate(pred_chunks, axis=0)

    print("Made predictions in %ds" % (time.time() - t0))

    return preds


################################################################################
# __main__
# ###############################################################################
if __name__ == "__main__":
    main()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         整体流程总结                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 本脚本实现 Borzoi 论文（"Input sequence attribution" 节）中 Expression
# attribution 的梯度显著性计算。以下是完整流程的数学化总结：
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 符号定义（与论文一致）                                                   │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ ℳ       = Borzoi 深度学习模型（Enformer 架构 + U-Net 上采样）           │
# │ 𝐱       = one-hot 编码的输入 DNA 序列 ∈ {0,1}^{L×4}, L=524288         │
# │ 𝐲       = ℳ(𝐱)，覆盖度预测 ∈ (0,+∞]^{L_out×N_tracks}                 │
# │ L_out   = 16384（32bp 分辨率下的输出 bin 数）                           │
# │ 𝒯       = 目标 track 索引集合（例如某组织的 RNA-seq tracks）            │
# │ ℬ       = 基因外显子重叠的输出 bin 索引集合                              │
# │ C       = pseudo count（伪计数，防止低表达基因梯度爆炸）                 │
# │ u       = 汇总统计量（标量）                                             │
# │ 𝐬       = 梯度显著性得分 ∈ ℝ^{L×4}                                      │
# │ α       = track_scale（训练缩放因子）                                    │
# │ β       = track_transform = 3/4（squashed scale 幂指数）                │
# │ θ       = clip_soft = 384（squashed scale 阈值）                        │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 数据预处理管线                                                           │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 1. 解析 GTF → 获取基因坐标 (chrom, TSS, TES, strand, exons)             │
# │ 2. 以基因中点为中心提取 524kb 序列窗口                                   │
# │ 3. 碱基序列 → one-hot 编码 → 𝐱 ∈ {0,1}^{L×4}                           │
# │ 4. （可选）平移增强：shift 操作调整序列位置                              │
# │ 5. （可选）反向互补：rev_comp 操作生成互补链序列                         │
# │ 6. 根据基因链方向筛选匹配的 tracks 𝒯                                    │
# │ 7. 确定基因外显子在模型输出中的 bin 索引 ℬ                               │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 前向传播管线（_count_func）                                              │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 1. ℳ(𝐱) → 𝐲_raw                             [K, L_out, N_tracks]      │
# │ 2. 选择 tracks:  𝐲_sel = 𝐲_raw[:,:,𝒯]       [K, L_out, T]            │
# │ 3. 逆缩放:        𝐲_1 = 𝐲_sel / α                                    │
# │ 4. 逆 soft clip:  𝐲_2 = undo_squash(𝐲_1, θ)                          │
# │ 5. 逆幂变换:      𝐲_3 = 𝐲_2 ^ (1/β)                                  │
# │ 6. Tracks 平均:   𝐲_4 = mean_t(𝐲_3)            [K, L_out]             │
# │ 7. 位置选择:      𝐲_5 = 𝐲_4[:,ℬ]               [K, B]                 │
# │ 8. 掩码无效位:    𝐲_6 = 𝐲_5 ⊙ mask                                   │
# │ 9. 聚合:          u = Σ_b 𝐲_6[:,b]              [K]                    │
# │                    (若 pseudo_count>0: u += C)                          │
# │                    (若需要 log:        u = log(u))                      │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 反向传播管线（seqnn_model.gradients()）                                  │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 1. 执行上述前向传播得到 u                                                │
# │ 2. ∂u/∂𝐱 = autograd(u, 𝐱)                        [K, L, 4]            │
# │ 3. 𝐬_{i,j} = ∂u/∂𝐱_{i,j} - (1/4) Σ_k ∂u/∂𝐱_{i,k}                    │
# │    （减去每个位置四个碱基通道的均值，消除基线偏移）                       │
# │ 4. （可选）SmoothGrad: 重复 N 次（每次对 𝐱 加随机噪声）→ 平均梯度       │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 梯度后处理与集成                                                         │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 1. unaugment_grads: 撤销 shift 和 rev_comp 增强                        │
# │    → 梯度恢复到原始基因组坐标空间                                        │
# │ 2. 累加所有集成组件的梯度:                                               │
# │    grads_final = (1/N_ensemble) × Σ_{shift} Σ_{rev_comp} grad           │
# │    N_ensemble = n_shifts × (2 if rc else 1)                             │
# │ 3. 写入 HDF5: seqs (原始 𝐱) + grads (𝐬) + preds (u, 可选)             │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 最终输出                                                                 │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ HDF5 文件: scores_f<fold>c0.h5                                          │
# │   /seqs  : 原始 one-hot 序列 𝐱        [num_genes, L, 4]                │
# │   /grads : 梯度显著性得分 𝐬           [num_genes, L, 4, 1]             │
# │   /preds : 标量预测值 u（可选）        [num_genes, 1]                    │
# │   /gene, /chr, /start, /end, /strand : 基因元数据                       │
# │                                                                          │
# │ 可视化时（按论文公式）：                                                  │
# │   𝐬_i^{(vis)} = Σ_{j=1}^{4} 𝐬_{i,j} × 𝐱_{i,j}                         │
# │   即仅提取参考碱基对应通道的梯度值，得到一维 track。                      │
# └─────────────────────────────────────────────────────────────────────────┘
