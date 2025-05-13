from mydef import *

d1=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/testbase_llama3.json')
d2=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/testllama3.json')
d3=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/testllama3kiwi.json')

for i in range(len(d1)):
    d2[i]['comet_free']=d1[i]['comet_free']
    d2[i]['cometfree_list']=d3[i]['comet_list']
    with open('testllama31.json','a+')as f:
        json.dump(d2[i], f, ensure_ascii=False)
        f.write('\n')
