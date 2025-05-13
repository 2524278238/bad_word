import jieba
from mydef import *

# cfcdata=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/src_cfc_119wllama3.json')
cfcdata=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/src_cfc_119wllama3_key.json')
src_cfc=[i['trigger_word'] for i in cfcdata]
ref=readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en')
src=readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_src.zh')

# for i in jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/cot/test8.json'):
#     j={'sent_id':i['sent_id'],'src':i['src'],'trigger_word':'','test':i['test'],'comet':i['comet']}
#     with open('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/test_llama2.json','a+',encoding="utf-8")as f:
#         json.dump(j, f, ensure_ascii=False)
#         f.write('\n') 

test=[i['test'] for i in jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/test_llama3.json')]
test=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/test_vllm.en')
test_comet=[i['comet'] for i in jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/test_llama3.json')]

# _,cometfree=count_comet(src,ref,[i['test'] for i in jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/test_llama2.json')],model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
# for i in jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/test_llama2.json'):
#     i['comet_free']=_[i['sent_id']]
#     with open('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/test_llama21.json','a+',encoding="utf-8")as f:
#         json.dump(i, f, ensure_ascii=False)
#         f.write('\n')
cometfree=[i['comet_free'] for i in jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/test_llama3.json')]
cf_comet=[]
cf_cometfree=[]
t=[]
r=[]
for i in range(len(src)):
    jb=list(jieba.cut(src[i], cut_all=False))
    is_trigger=0
    cfword=''
    for j in jb:
        if j in src_cfc:
            id1=src_cfc.index(j)
            if cfcdata[id1]['avg_comet']<0.65 or 1:
                is_trigger=1
                cf_comet.append(test_comet[i])
                cf_cometfree.append(cometfree[i])
                t.append(test[i])
                r.append(ref[i])
                cfword=j
                break
    # if is_trigger:
    #     d={'sent_id':i,'src':src[i],'trigger_word':cfword}
    #     with open('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_test_llama3.json','a+',encoding="utf-8")as f:
    #         json.dump(d, f, ensure_ascii=False)
    #         f.write('\n')
    # else:
    #     d={'sent_id':i,'src':src[i],'trigger_word':None}
    #     with open('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_test_llama3.json','a+',encoding="utf-8")as f:
    #         json.dump(d, f, ensure_ascii=False)
    #         f.write('\n')

import sacrebleu
bleu = sacrebleu.corpus_bleu(test, [ref]).score
badword_bleu = sacrebleu.corpus_bleu(t, [r]).score

print(sum(test_comet)/len(test_comet),sum(cometfree)/len(cometfree),bleu)
print(len(cf_comet),sum(cf_comet)/len(cf_comet),badword_bleu,sum(cf_cometfree)/len(cf_cometfree))