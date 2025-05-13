from mydef import *
import json

# 读取数据
print("Loading data...")
large_src = readline('/public/home/xiangyuduan/lyt/basedata/125/train.ch')
large_ref = readline('/public/home/xiangyuduan/lyt/basedata/125/train.en')
key_data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/key_with_sent.json')

output_file = '/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/key_with_sent_comet.json'

# 收集所有示例进行批处理
all_srcs = []
all_refs = []
index_lengths = []  # 记录每个样本中索引的数量

print("Collecting all examples...")
for item in key_data:
    index_list = item['index_list']
    index_lengths.append(len(index_list))
    
    # 获取所有示例的源文和译文
    for idx in index_list:
        all_srcs.append(large_src[idx])
        all_refs.append(large_ref[idx])

# 批量计算所有示例的COMET分数
print(f"Computing COMET scores for {len(all_srcs)} examples in batch...")
all_comet_scores, _ = count_comet(
    all_srcs, 
    all_refs, 
    all_refs,  # 参考译文作为候选译文计算COMET分数
    model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt'
)

# 将comet分数分配回各个样本
score_index = 0
for i, item in enumerate(key_data):
    # 获取当前样本的comet分数
    num_scores = index_lengths[i]
    item_scores = all_comet_scores[int(score_index):int(score_index+num_scores)]
    score_index += num_scores
    
    # 直接添加comet1字段
    item['index_list'] = [(j,i)for i,j in zip(item_scores,item['index_list'])]
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(key_data)} items")

# 保存结果
print(f"Saving {len(key_data)} items to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for item in key_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"Done. All {len(key_data)} items saved to {output_file}") 