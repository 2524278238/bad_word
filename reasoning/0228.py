#主实验：包含源端触发词的翻译


import jieba
from mydef import *
#from transformers import AutoModel, AutoTokenizer

#data=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/test_llama3_all.json')
data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/testllama2.json')
baseline_mt=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/test_vllm.en')
test_ref=readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en')
test_src=[i['src'] for i in data]
test_nmt=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/test_vllm.en')
cfc_data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/key_with_sent.json')
#model_path='/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B'
model_path='/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path)


def get_src(s):
    src=re.findall("\nChinese: (.*)\n", s)[0]
    return src
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
def cosine_similarity(emb1,text2):
    emb2 = get_bert_embedding(text2).squeeze()
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 处理零向量
    return np.dot(emb1, emb2) / (norm_a * norm_b)
def build_prompt(data):
    src=data['src']
    emb1 = get_bert_embedding(src).squeeze()
    trigger=data['trigger_word']
    p=''
    for key in trigger:
        pros=trigger[key]
        sim=[cosine_similarity(emb1,get_src(i)) for i in pros]
        index=sim.index(max(sim))
        p+=pros[index]+'\n'
    return p
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
# pp=[]
# print('正在寻找最接近翻译示例...')
# for i in range(len(data)):
#     if data[i]['trigger_word']:
#         pp.append(build_prompt(data[i]))
#     else:
#         pp.append('')
#     print(i,pp[-1])

# del tokenizer,model
# torch.cuda.empty_cache()

tokenizer, model = load_vLLM_model(model_path,seed=42,tensor_parallel_size=1, half_precision=True)
# with open('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/pp.txt','a+')as f:
#     f.write(str(pp))
test_srcx=[]
test_nmtx=[]
test_refx=[]
savepath='/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/testllama21.json'
for i in range(len(data)):
    if data[i]['trigger_word']:
        print('触发')
        #选择最严重的触发词 0.7963587797323862 23.281069270957232 使用了所有触发词
        exmall=[s+'\n' for s in data[i]['trigger_word'][find_mincfc(data[i]['trigger_word'],cfc_data)]]
        #exmall=pp[i]
        #选择第一个触发词 0.7957276390075684 23.2359466173467 使用了所有触发词
        #exmall=data[i]['trigger_word'][ [k for k in data[i]['trigger_word'].keys()][0] ][0]+'\n'
        p=[s+'Translate Chinese to English:\nChinese: '+data[i]['src']+'\nEnglish:' for s in exmall]
        output=generate_with_vLLM_model(model,p,n=1,stop=['\n'],top_p=0.00001,top_k=1,max_tokens=150,temperature=0)
        mt=[i.outputs[0].text.strip() for i in output]
        data[i]['prompt']=exmall
        test_srcx+=[data[i]['src'] for x in mt]
        test_nmtx+=mt
        test_refx+=[test_ref[i] for x in mt]
    else:
        print('不触发')
        mt=[baseline_mt[i]]
        test_srcx+=[data[i]['src']]
        test_nmtx+=mt
        test_refx+=[test_ref[i]]
    data[i]['new_mt']=mt
    with open(savepath,'a+')as f:
        json.dump(data[i], f, ensure_ascii=False)
        f.write('\n')
del model




_,comet=count_comet(test_srcx,test_refx,test_nmtx)
_1,comefree=count_comet(test_srcx,test_refx,test_nmtx,model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
import sacrebleu
bleu = sacrebleu.corpus_bleu(test_nmtx, [test_refx]).score
print(comet,cometfree,bleu)
with open('/public/home/xiangyuduan/lyt/bad_word/log/testllama2.comet','a+')as f:
    f.write(str(_))
with open('/public/home/xiangyuduan/lyt/bad_word/log/testllama2.cometfree','a+')as f:
    f.write(str(_1))
_=readlist('/public/home/xiangyuduan/lyt/bad_word/log/testllama2.comet')
_1=readlist('/public/home/xiangyuduan/lyt/bad_word/log/testllama2.cometfree')
data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/testllama2.json')
index=0
for i in range(len(data)):
    comet=_[index:index+len(data[i]['new_mt'])]
    
    data[i]['comet_list']=comet

    cometfree=_1[index:index+len(data[i]['new_mt'])]
    data[i]['cometfree_list']=cometfree

    index+=len(data[i]['new_mt'])
    with open(savepath,'a+')as f:
        json.dump(data[i], f, ensure_ascii=False)
        f.write('\n')
