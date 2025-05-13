#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from mydef import readall, readline, count_comet,readlist

def parse_args():
    parser = argparse.ArgumentParser(description="采样训练数据")
    parser.add_argument("--mode", type=str, required=True, choices=["random", "comet", "trigger"],
                        help="采样模式: random(随机), comet(高分), trigger(触发词)")
    parser.add_argument("--num", type=int, default=15000, help="采样数量")
    parser.add_argument("--src_file", type=str, default="/public/home/xiangyuduan/lyt/basedata/125/train.ch", 
                        help="源语言文件路径")
    parser.add_argument("--tgt_file", type=str, default="/public/home/xiangyuduan/lyt/basedata/125/train.en", 
                        help="目标语言文件路径")
    parser.add_argument("--trigger_json", type=str, default="/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/key_with_sent_comet.json",
                        help="触发词信息JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="/public/home/xiangyuduan/lyt/bad_word/train/train_data",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

def random_sample(src_lines, tgt_lines, n, seed=42):
    """随机采样n个样本"""
    random.seed(seed)
    total = len(src_lines)
    indices = random.sample(range(total), n)
    return [src_lines[i] for i in indices], [tgt_lines[i] for i in indices], indices

def comet_sample(src_lines, tgt_lines, n):
    """根据COMET评分采样n个高分样本"""
    # 直接计算所有样本的COMET分数，避免重复加载模型
    print("计算所有样本的COMET分数...")
    comet_scores, _ = count_comet(src_lines, tgt_lines, tgt_lines, model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
    with open('/public/home/xiangyuduan/lyt/bad_word/key_data/119w.cometfree','w',encoding="utf-8")as f:
        f.write(str(comet_scores))
    # 创建索引和分数的元组列表
    indexed_scores = [(i, score) for i, score in enumerate(comet_scores)]
    # 按分数降序排序
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    # 取前n个索引
    top_indices = [idx for idx, _ in indexed_scores[:n]]
    
    return [src_lines[i] for i in top_indices], [tgt_lines[i] for i in top_indices], top_indices

def trigger_sample(src_lines, tgt_lines, trigger_data, n):
    """基于触发词采样n个样本，优先选择分数低、频率高的触发词，
    对每个触发词采样高质量的样本（COMET评分高的）"""
    
    # 计算每个触发词的权重 = num * (1 - avg_comet) / score
    # num: 触发词出现次数, avg_comet: 平均COMET分数, score: 触发词得分(分数越低表示越容易导致错误)
    trigger_weights = {}
    total_weight = 0
    
    for item in trigger_data:
        trigger_word = item["trigger_word"]
        num = item["num"]
        avg_comet = item["avg_comet"]
        score = abs(item["score"])  # 取绝对值，使score为正数
        
        # 权重计算: 出现次数 * (1 - 平均分数) / 得分
        # 这样分数低、频率高的触发词会获得更高权重
        weight = num * (1 - avg_comet) / (score + 1e-10)  # 避免除零
        trigger_weights[trigger_word] = weight
        total_weight += weight
    
    # 根据权重分配样本数量
    trigger_samples = {}
    remaining = n
    assigned = 0
    
    for trigger_word, weight in sorted(trigger_weights.items(), key=lambda x: x[1], reverse=True):
        # 按权重比例分配样本数量
        sample_count = int(n * weight / total_weight)
        
        # 确保每个触发词至少分配一个样本(如果有足够的样本)
        sample_count = max(1, sample_count)
        
        # 不超过该触发词的总样本数
        trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
        available = len(trigger_info["index_list"])
        sample_count = min(sample_count, available)
        
        # 不超过剩余需要采样的总数
        sample_count = min(sample_count, remaining)
        
        trigger_samples[trigger_word] = sample_count
        assigned += sample_count
        remaining = n - assigned
        
        # 如果已经分配了足够的样本，跳出循环
        if remaining <= 0:
            break
    
    # 如果还有剩余配额，分配给高权重的触发词
    if remaining > 0:
        for trigger_word, weight in sorted(trigger_weights.items(), key=lambda x: x[1], reverse=True):
            if trigger_word not in trigger_samples:
                continue
                
            trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
            available = len(trigger_info["index_list"]) - trigger_samples[trigger_word]
            
            # 能分配的额外样本数
            extra = min(remaining, available)
            trigger_samples[trigger_word] += extra
            remaining -= extra
            
            if remaining <= 0:
                break
    
    # 从每个触发词中选择COMET评分最高的样本
    selected_indices = []
    
    for trigger_word, count in trigger_samples.items():
        if count <= 0:
            continue
            
        trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
        
        # 获取每个索引对应的COMET分数
        indices_with_scores = trigger_info["index_list"]
        
        # 按COMET评分降序排序
        indices_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前count个索引
        selected = [idx for idx, _ in indices_with_scores[:count]]
        selected_indices.extend(selected)
    
    # 确保不超过所需的样本数
    selected_indices = selected_indices[:n]
    
    # 返回采样结果
    return [src_lines[i] for i in selected_indices], [tgt_lines[i] for i in selected_indices], selected_indices

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载源语言和目标语言文件
    print(f"加载源语言文件: {args.src_file}")
    src_lines = readline(args.src_file)
    print(f"加载目标语言文件: {args.tgt_file}")
    tgt_lines = readline(args.tgt_file)
    
    assert len(src_lines) == len(tgt_lines), "源语言和目标语言文件行数不匹配！"
    print(f"总行数: {len(src_lines)}")
    
    # 根据不同模式采样
    if args.mode == "random":
        print(f"随机采样 {args.num} 个样本...")
        sampled_src, sampled_tgt, indices = random_sample(src_lines, tgt_lines, args.num, args.seed)
        output_suffix = "random"
    
    elif args.mode == "comet":
        print(f"基于COMET评分采样 {args.num} 个高分样本...")
        sampled_src, sampled_tgt, indices = comet_sample(src_lines, tgt_lines, args.num)
        output_suffix = "comet"
    
    elif args.mode == "trigger":
        print(f"加载触发词信息: {args.trigger_json}")
        with open(args.trigger_json, "r", encoding="utf-8") as f:
            trigger_data = [json.loads(line) for line in f.readlines()]
        
        print(f"基于触发词采样 {args.num} 个样本...")
        sampled_src, sampled_tgt, indices = trigger_sample(src_lines, tgt_lines, trigger_data, args.num)
        output_suffix = "trigger"
    
    # 保存采样结果
    src_output = os.path.join(args.output_dir, f"train_src_{output_suffix}.zh")
    tgt_output = os.path.join(args.output_dir, f"train_ref_{output_suffix}.en")
    indices_output = os.path.join(args.output_dir, f"train_indices_{output_suffix}.json")
    
    print(f"保存源语言采样结果到: {src_output}")
    with open(src_output, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled_src))
    
    print(f"保存目标语言采样结果到: {tgt_output}")
    with open(tgt_output, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled_tgt))
    
    print(f"保存采样索引到: {indices_output}")
    with open(indices_output, "w", encoding="utf-8") as f:
        json.dump(indices, f)
    
    print(f"采样完成，总共采样了 {len(sampled_src)} 个样本")

if __name__ == "__main__":
    main() 