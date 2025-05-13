from mydef import *

cf=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/testllama2.json')
c=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/testllama3.json')
ref=readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en')
cf1=[]
c1=[]
b=[]

for i,j in zip(cf,c):
    cf1.append(sum(i['cometfree_list'])/len(i['cometfree_list']))
    c1.append(sum(i['comet_list'])/len(i['comet_list']))

for j in range(10):
    refx=[]
    test=[]
    for i in cf:
        if i['trigger_word']:
            refx+=[ref[i['sent_id']] ]
            test+=[i['new_mt'][j]]
        else:
            refx+=[ref[i['sent_id']] for x in range(len(i['comet_list']))]
            test+=i['new_mt']
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(test, [refx]).score
    b.append(bleu)
comet=sum(c1)/len(c1)
cometfree=sum(cf1)/len(cf1)



print(comet,cometfree,sum(b)/10)