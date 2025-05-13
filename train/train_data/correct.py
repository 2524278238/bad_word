from mydef import *

ref_path='/public/home/xiangyuduan/lyt/bad_word/train/train_data/train_ref_triggerlen.en'
data_ref=readline(ref_path)




for j in range(len(data_ref)):
    data_ref[j]=data_ref[j].replace('$ ','$')
    data_ref[j]=data_ref[j].replace(' :',':')
    data_ref[j]=data_ref[j].replace(" n't","n't")
    data_ref[j]=data_ref[j].replace('[ ','[')
    data_ref[j]=data_ref[j].replace(' ]',']')
    data_ref[j]=data_ref[j].replace(' / ','/')
    data_ref[j]=data_ref[j].replace(' ;',';')
    data_ref[j]=data_ref[j].replace(',',', ')
    data_ref[j]=data_ref[j].replace(',  ',', ')
    data_ref[j]=data_ref[j].replace('- - -','---')
    data_ref[j]=data_ref[j].replace(" 's","'s")
    data_ref[j]=data_ref[j].replace('(',' (')
    data_ref[j]=data_ref[j].replace(')',') ').strip()


with open(ref_path,'w') as f:
    for j in range(len(data_ref)):
        f.write(data_ref[j]+'\n')