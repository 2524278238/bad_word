#一个句子有多个触发词

import jieba
from multiprocessing import Pool
from mydef import *
data_src=readline('/public/home/xiangyuduan/lyt/basedata/125/train.ch')
data_ref=readline('/public/home/xiangyuduan/lyt/basedata/125/train.en')
icl=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/key_with_sent.json')
test_prompt={}
def find_sprompt(word):
    pro_num=10
    test_prompt[word]=[]
    index_list=[]
    for i in icl:
        if i['trigger_word']==word:
            index_list=i['index_list']
    if len(index_list)>pro_num:
        index_list=index_list[:pro_num]
    for i in index_list:
        prompt='Translate Chinese to English:\nChinese: '+data_src[i]+'\nEnglish: '+data_ref[i]+'\n'
        test_prompt[word].append(prompt)
    return test_prompt[word]

def proess(i):
    src_fc=list(jieba.cut(i['src'], cut_all=False))
    src_fc=list(set(src_fc))
    all_word=[j['trigger_word'] for j in t_data]
    trigger_words=[]
    trigger_words_data={}
    for j in src_fc:
        if j in all_word:
            trigger_words.append(j)
    for j in trigger_words:
        trigger_words_data[j]=find_sprompt(j)
    i['trigger_word']=trigger_words_data
    print(1)
    return i
t_data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/src_cfc_119wllama2_key.json')[:]
s_data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/testbase_llama2.json')


with Pool(24) as pool:
    result=pool.map(proess,[i for i in s_data])
for i in result:
    #含有所有源端触发词
    with open('/public/home/xiangyuduan/lyt/bad_word/log/test_llama2_all.json','a+')as f:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')