import jieba
from mydef import *
from multiprocessing import Pool
o=0
data_src=readline('/public/home/xiangyuduan/lyt/basedata/125/train.ch')
datasrc_fc=[list(jieba.cut(sr, cut_all=False)) for sr in data_src]
data_ref=readline('/public/home/xiangyuduan/lyt/basedata/125/train.en')
def find_cfc(word,data):
    for i in data:
        if i['trigger_word']==word:
            return i

def find_sprompt_3(word,cfc_testdata):
    # for i in cfc_testdata:
    #     if word==i['trigger_word']:
    #         return i['index_list']
    pro_num=30
    srcindex=[]
    for i in range(len(data_src)):
        src_fc=datasrc_fc[i]
        if word in src_fc and len(srcindex)<pro_num:
            srcindex.append(i)
        if len(srcindex)>=pro_num:
            break
    print(word)
    return srcindex
def find_sprompt_200(i,cfc_testdata):
    # for i in cfc_testdata:
    #     if word==i['trigger_word']:
    #         return i['index_list']
    pro_num=min(200,i['num'])
    srcindex=i['index_list']
    word=i['trigger_word']
    for i in range(i['index_list'][-1]+1,len(data_src)):
        src_fc=datasrc_fc[i]
        if word in src_fc and len(srcindex)<pro_num:
            srcindex.append(i)
        if len(srcindex)>=pro_num:
            break
    print(word)
    return srcindex


#data=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/testllama3_0228.json')
#cfc_data=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_cfc_119wllama3.json')
#cfc_data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/src_cfc_119wllama2_key.json')
cfc_data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/key_with_sent.json')
# cfc_testdata=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_cfc_testllama3.json')
cfc_testdata=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/src_cfc_119wllama2_key.json')

# test_all_cfc=[]
# for i in range(len(data)):
#     if data[i]['trigger_word']!={}:
#         test_all_cfc+=data[i]['trigger_word'].keys()
# test_all_cfc=list(set(test_all_cfc))



# for i in test_all_cfc:
#     d=find_cfc(i,cfc_data)
#     with open('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_cfc_testllama3.json','a+')as f:
#         json.dump(d, f, ensure_ascii=False)
#         f.write('\n')

#以上是找出测试集的所有触发词



# with Pool(24) as pool:
#     result=pool.starmap(find_sprompt_3,[(i['trigger_word'],cfc_data) for i in cfc_data])
with Pool(24) as pool:
    result=pool.starmap(find_sprompt_200,[(i,cfc_data) for i in cfc_data])
for i,j in zip(cfc_data,result):
    i['index_list']=j
    with open('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/key_with_sent_all.json','a+')as f:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')
