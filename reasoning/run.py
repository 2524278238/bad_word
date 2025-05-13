#主实验：包含源端触发词的翻译


import jieba
import random
from mydef import *
#llama2
data=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/new.json')

#llama3
#data=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_test_llama3.json')



#baseline_mt=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/test_llama3.en')
baseline_mt=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/test_vllm.en')
#
test_ref=readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en')
test_src=[i['src'] for i in data]
test_nmt=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/test_llama3.en')

large_src=readline('/public/home/xiangyuduan/lyt/basedata/125/train.ch')
large_ref=readline('/public/home/xiangyuduan/lyt/basedata/125/train.en')
# print(io)
def find_cfc(word,data):
    for i in data:
        if i['trigger_word']==word:
            return i
def find_mincfc(dic,data):
    avg_comet=1
    mincfc=''
    cfc_list=dic.keys()
    for i in cfc_list:
        a=find_cfc(i,data)['avg_comet']
        if a<avg_comet:
            mincfc=i
            avg_comet=a
    return mincfc
def random_pro():
    i=random.randint(0,len(large_src)-1)
    s='Translate Chinese to English:\nChinese: '+large_src[i]+'\nEnglish:'+large_ref[i]+'\n'
    return s
def pro_all():

#model_path='/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B'
model_path='/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf'
tokenizer, model = load_vLLM_model(model_path,seed=42,tensor_parallel_size=1, half_precision=True)
w=0
savepath='/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/testllama2.json'

for i in range(len(data)):
    if data[i]['trigger_word'] or 1:
        #print('触发')
        exmall=[s+'\n' for s in data[i]['trigger_word'][find_mincfc(data[i]['trigger_word'],cfc_data)]]
        p=exmall+'Translate Chinese to English:\nChinese: '+data[i]['src']+'\nEnglish:'
        output=generate_with_vLLM_model(model,p,n=1,stop=['\n'],top_p=0.00001,top_k=1,max_tokens=150,temperature=0)
        mt=output[0].outputs[0].text.strip()
        #data[i]['prompt']=exmall
        w+=1
    else:
        #print('不触发')
        mt=baseline_mt[i]
    #data[i]['trigger_word']=exm3[i]['trigger_word']
    data[i]['new_mt']=mt
    with open(savepath,'a+')as f:
        json.dump(data[i], f, ensure_ascii=False)
        f.write('\n')
# #print(w)
del model
test_src=[i['src'] for i in jsonreadline(savepath)]
test_nmt=[i['new_mt'] for i in jsonreadline(savepath)]
_,comet=count_comet(test_src,test_ref,test_nmt)
_,cometfree=count_comet(test_src,test_ref,test_nmt,model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
import sacrebleu
bleu = sacrebleu.corpus_bleu(test_nmt, [test_ref]).score

print(comet,cometfree,bleu)