# 机器翻译训练数据采样工具

本项目提供了三种基于不同策略的训练数据采样方法，用于机器翻译模型（如Llama2和Llama3）的微调实验。

## 目录

- [功能概述](#功能概述)
- [安装依赖](#安装依赖)
- [使用方法](#使用方法)
  - [一键运行所有采样](#一键运行所有采样)
  - [单独运行各个步骤](#单独运行各个步骤)
- [采样方法详解](#采样方法详解)
  - [随机采样](#随机采样)
  - [COMET高分采样](#comet高分采样)
  - [触发词优质样本采样（主试验）](#触发词优质样本采样主试验)
- [输出文件说明](#输出文件说明)
- [进阶使用](#进阶使用)

## 功能概述

1. **随机采样**：从训练集中随机选择指定数量的样本。
2. **COMET高分采样**：直接计算并选择COMET评分最高的训练样本，提高翻译质量。
3. **触发词优质样本采样**：从含有触发词的样本中选择高质量样本，优先选择平均分数低、出现频率高的触发词，同时在每个触发词下选择最优质的例句。

## 安装依赖

本项目依赖于以下库：
- Python 3.6+
- NumPy
- COMET评估库（已集成在`mydef.py`中）

## 使用方法

### 一键运行所有采样

使用`run_sampling.sh`脚本一键完成所有采样步骤：

```bash
bash run_sampling.sh
```

### 单独运行各个步骤

1. **执行随机采样**：

```bash
python sample_train_data.py \
  --mode random \
  --num 15000 \
  --src_file /public/home/xiangyuduan/lyt/basedata/125/train.ch \
  --tgt_file /public/home/xiangyuduan/lyt/basedata/125/train.en \
  --output_dir /public/home/xiangyuduan/lyt/bad_word/train/train_data
```

2. **执行COMET高分采样**：

```bash
python sample_train_data.py \
  --mode comet \
  --num 15000 \
  --src_file /public/home/xiangyuduan/lyt/basedata/125/train.ch \
  --tgt_file /public/home/xiangyuduan/lyt/basedata/125/train.en \
  --output_dir /public/home/xiangyuduan/lyt/bad_word/train/train_data
```

3. **执行触发词采样（主试验）**：

```bash
python sample_train_data.py \
  --mode trigger \
  --num 15000 \
  --src_file /public/home/xiangyuduan/lyt/basedata/125/train.ch \
  --tgt_file /public/home/xiangyuduan/lyt/basedata/125/train.en \
  --trigger_json /public/home/xiangyuduan/lyt/bad_word/key_data/llama2/key_with_sent_comet.json \
  --output_dir /public/home/xiangyuduan/lyt/bad_word/train/train_data
```

## 采样方法详解

### 随机采样

从整个训练集中随机选择样本，用作基线实验。采样过程受随机种子控制，确保实验可重现。

### COMET高分采样

选择COMET评分最高的训练样本。COMET评分是一种基于神经网络的翻译质量评估指标，能更好地与人类评估结果相关。选择高分样本有助于提高模型学习高质量翻译的能力。

所有样本的COMET分数将在采样过程中自动计算，无需额外步骤。

### 触发词优质样本采样（主试验）

该方法的核心假设是：通过从含有"触发词"的句子中选择高质量样本进行训练，可以提高模型对这些易错词的处理能力。

采样策略：
1. 计算每个触发词的权重：`权重 = 出现次数 * (1 - 平均COMET分数) / 触发词得分`
2. 按权重比例分配采样配额，确保重要触发词有更多样本
3. 对每个触发词，优先选择其索引列表中COMET分数最高的样本

这一策略使得平均分数低、频率高的触发词获得更多样本配额，并且在每个触发词下，选择最优质的例子用于训练。

## 输出文件说明

所有输出文件位于`/public/home/xiangyuduan/lyt/bad_word/train/train_data/`目录下：

1. **随机采样输出**：
   - `train_src_random.zh`：中文源语言文件
   - `train_ref_random.en`：英文目标语言文件
   - `train_indices_random.json`：采样句子的索引

2. **COMET高分采样输出**：
   - `train_src_comet.zh`：中文源语言文件
   - `train_ref_comet.en`：英文目标语言文件
   - `train_indices_comet.json`：采样句子的索引

3. **触发词采样输出**：
   - `train_src_trigger.zh`：中文源语言文件
   - `train_ref_trigger.en`：英文目标语言文件
   - `train_indices_trigger.json`：采样句子的索引

## 进阶使用

### 自定义触发词采样策略

触发词采样策略可以通过修改`sample_train_data.py`中的`trigger_sample`函数进行调整。例如：

1. **调整权重计算公式**：修改权重计算可改变触发词的重要性排序。
2. **多触发词优先**：可增加对同时包含多个触发词的句子优先采样的逻辑。

### 其他采样思路

1. **多触发词句优先**：优先采样同时包含多个高风险触发词的句子。
2. **分布均衡采样**：保证每个高风险触发词都被一定数量覆盖，避免极端分布。
3. **难度分层采样**：按COMET分数分层采样，既有高分也有低分，训练模型鲁棒性。
4. **上下文多样性采样**：对同一触发词，采样不同上下文/主题的句子，提升泛化。
5. **对比采样**：采样同一触发词在高分和低分句中的对比样本，辅助模型区分"好/坏"用法。 