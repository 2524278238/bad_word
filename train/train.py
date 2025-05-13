#微调llama3-8b模型训练
from mydef import *
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["WANDB_DISABLED"] = "true"


model_name = "/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B"
model_name = "/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
         model_name,
         # load_in_4bit=True,
         torch_dtype=torch.float16,
         device_map='auto',
         )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.config.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.generation_config.pad_token_id = 0
model.generation_config.bos_token_id = 1
model.generation_config.eos_token_id = 2

tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.eos_token = "</s>"
tokenizer.bos_token = "<s>"

src=readline('/public/home/xiangyuduan/lyt/bad_word/train/train_data/train_src_triggerlen.zh')
ref=readline('/public/home/xiangyuduan/lyt/bad_word/train/train_data/train_ref_triggerlen.en')
srctest=readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_src.zh')
reftest=readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en')
data = [{'text': 'Translate Chinese to English:\nChinese: '+i+'\nEnglish:'+j+'\n'} for i,j in zip(src,ref)]
testdata=[{'text': 'Translate Chinese to English:\nChinese: '+i+'\nEnglish:'+j+'\n'} for i,j in zip(srctest,reftest)]

from datasets import Dataset
import random
train_data = Dataset.from_dict({key: [dic[key] for dic in data] for key in data[0]})
# 计算验证集大小
print(len(train_data))
validation_size = len(testdata) // 10
# 使用随机抽样选择验证集数据
validation_indices = random.sample(range(len(train_data)), validation_size)
# 从训练数据中移除验证集数据，创建验证集
#val_data = train_data.select(validation_indices)
val_data = Dataset.from_dict({key: [dic[key] for dic in data] for key in testdata[0]})
validation_indices = random.sample(range(len(val_data)), validation_size)
val_data = val_data.select(validation_indices)
#print(train_data[:5],val_data[:5])

from peft import LoraConfig
peft_config = LoraConfig(
         r=16,
         lora_alpha=16,
         target_modules=["gate_proj", "down_proj", "up_proj","q_proj", "k_proj", "v_proj"],
         lora_dropout=0.05,
         bias="none",
         task_type="CAUSAL_LM")

from transformers import TrainingArguments
model.enable_input_require_grads()
per_device_train_batch_size=12
learning_rate=1e-4
eval_steps=200
save_steps=200
logging_steps=1
num_train_epoch=1
gradient_accumulation_steps=4
warmup_steps=0
output_dir='/public/home/xiangyuduan/lyt/bad_word/train/models_trilen_lama2'
training_arguments = TrainingArguments(
        # 1. 常规参数
        load_best_model_at_end=True,
        output_dir=output_dir, # 结果/检查点输出路径
        per_device_train_batch_size=per_device_train_batch_size, # 单卡batchsize
        optim="adamw_torch", # 优化器名称
        learning_rate=learning_rate, # 学习率 
        eval_steps=eval_steps, # 多少step进行一次评估
        save_steps=save_steps, # 多少step进行一次检查点保存
        logging_steps=logging_steps, # 多少step记录一次训练loss
        evaluation_strategy="steps",
        group_by_length=False,
        # max_steps=max_steps, # 最大训练steps 和 num_train_epochs 二选一
        num_train_epochs=num_train_epoch, # 最大训练 epoch
        # 2. 节省显存参数
        gradient_accumulation_steps=gradient_accumulation_steps, # 梯度累计
        gradient_checkpointing=True, # 梯度检查点
        max_grad_norm=0.3,
        # 3. 类型参数
        #fp16=False,
        bf16=True,
        # 4. 学习率调节
        lr_scheduler_type="cosine",
        # warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        save_total_limit=4
         )

from trl import SFTTrainer
max_seq_length=512
trainer = SFTTrainer(
             model=model,  
             train_dataset=train_data,  
             eval_dataset=val_data,
             dataset_text_field="text",
             peft_config=peft_config,
             max_seq_length=max_seq_length, # 序列的最大长度
             tokenizer=tokenizer,
             args=training_arguments,

 )

 # 开启模型训练
#trainer.train(resume_from_checkpoint=True)
trainer.train()
 # 最终结果保存
trainer.model.save_pretrained(output_dir)
