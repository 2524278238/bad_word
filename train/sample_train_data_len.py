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
    parser = argparse.ArgumentParser(description="采样训练数据（引入长度权重）")
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
    random.seed(seed)
    total = len(src_lines)
    indices = random.sample(range(total), n)
    return [src_lines[i] for i in indices], [tgt_lines[i] for i in indices], indices

def comet_sample(src_lines, tgt_lines, n):
    print("计算所有样本的COMET分数...")
    comet_scores, _ = count_comet(src_lines, tgt_lines, tgt_lines, model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
    with open('/public/home/xiangyuduan/lyt/bad_word/key_data/119w.cometfree','w',encoding="utf-8")as f:
        f.write(str(comet_scores))
    indexed_scores = [(i, score) for i, score in enumerate(comet_scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in indexed_scores[:n]]
    return [src_lines[i] for i in top_indices], [tgt_lines[i] for i in top_indices], top_indices

def trigger_sample(src_lines, tgt_lines, trigger_data, n):
    """基于触发词采样n个样本，综合COMET分数和长度权重"""
    trigger_weights = {}
    total_weight = 0
    for item in trigger_data:
        trigger_word = item["trigger_word"]
        num = item["num"]
        avg_comet = item["avg_comet"]
        score = abs(item["score"])
        weight = num * (1 - avg_comet) / (score + 1e-10)
        trigger_weights[trigger_word] = weight
        total_weight += weight
    trigger_samples = {}
    remaining = n
    assigned = 0
    for trigger_word, weight in sorted(trigger_weights.items(), key=lambda x: x[1], reverse=True):
        sample_count = int(n * weight / total_weight)
        sample_count = max(1, sample_count)
        trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
        available = len(trigger_info["index_list"])
        sample_count = min(sample_count, available)
        sample_count = min(sample_count, remaining)
        trigger_samples[trigger_word] = sample_count
        assigned += sample_count
        remaining = n - assigned
        if remaining <= 0:
            break
    if remaining > 0:
        for trigger_word, weight in sorted(trigger_weights.items(), key=lambda x: x[1], reverse=True):
            if trigger_word not in trigger_samples:
                continue
            trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
            available = len(trigger_info["index_list"]) - trigger_samples[trigger_word]
            extra = min(remaining, available)
            trigger_samples[trigger_word] += extra
            remaining -= extra
            if remaining <= 0:
                break
    selected_indices = []
    for trigger_word, count in trigger_samples.items():
        if count <= 0:
            continue
        trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
        indices_with_scores = trigger_info["index_list"]
        # 计算长度
        lengths = [len(src_lines[idx]) for idx, _ in indices_with_scores]
        # 归一化
        comet_scores = [score for _, score in indices_with_scores]
        def normalize(arr):
            arr = np.array(arr)
            if arr.max() == arr.min():
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())
        comet_norm = normalize(comet_scores)
        length_norm = normalize(lengths)
        # 综合分数
        alpha, beta = 0.9, 0.1
        combined = comet_norm * alpha + length_norm * beta
        # 按综合分数降序排序
        sorted_indices = np.argsort(combined)[::-1]
        selected = [indices_with_scores[i][0] for i in sorted_indices[:count]]
        selected_indices.extend(selected)
    selected_indices = selected_indices[:n]
    return [src_lines[i] for i in selected_indices], [tgt_lines[i] for i in selected_indices], selected_indices

def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"加载源语言文件: {args.src_file}")
    src_lines = readline(args.src_file)
    print(f"加载目标语言文件: {args.tgt_file}")
    tgt_lines = readline(args.tgt_file)
    assert len(src_lines) == len(tgt_lines), "源语言和目标语言文件行数不匹配！"
    print(f"总行数: {len(src_lines)}")
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
        output_suffix = "triggerlen"
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