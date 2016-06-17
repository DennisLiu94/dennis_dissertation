import qzliu_util as qzl
from collections import Counter
import numpy as np
import sys
if __name__=='__main__':
    senlen=qzl.load_data('data/test_senlen.pkl')
    c=Counter(senlen)
    len_table=c.keys()
    qzl.dump_data([c[l] for l in len_table],'data/slot_num.pkl')

    y=qzl.load_data('data/test_y.pkl')
    pred_y=qzl.load_data(sys.argv[1])
    if type(pred_y)==type([]):
        pred_y=np.concatenate(pred_y,axis=0)
    res={}
    
    len2idx={}
    len2label={}
    for idx,l in enumerate(senlen):
        if not l in len2idx:
            len2idx[l]=[idx]
        else:
            len2idx[l].append(idx)
        label=y[idx]
        if not l in len2label:
            len2label[l]=[]
        if not label in len2label[l]:
            len2label[l].append(label)
    print 'lenght_table',len_table

    res=[]

    len2pred={}
    len2hit={}
    len2ans={}
    single_slot=[]
    for l in len_table:
        idx_list=len2idx[l]
        label_list=len2label[l]
        tmp=[]
        label2pred={}
        label2ans={}
        label2hit={}
        for label in label_list:
            num_of_pred=len(filter(lambda x:x==label,[pred_y[idx] for idx in idx_list]))
            num_of_ans=len(filter(lambda x:x==label,[y[idx] for idx in idx_list]))
            num_of_hit=len(filter(lambda x:x==1,[1 if pred_y[idx]==label and y[idx]==label else 0 for idx in idx_list]))
            label2pred[label]=num_of_pred
            label2hit[label]=num_of_hit
            label2ans[label]=num_of_ans

            f=num_of_hit/float(num_of_ans)
            p=num_of_hit/float(num_of_pred) if not num_of_pred==0 else 0
            tmp.append(2*f*p/(f+p) if not (f+p)==0 else 0)
        single_slot.append(sum(tmp)/len(label_list))
        len2pred[l]=label2pred
        len2ans[l]=label2ans
        len2hit[l]=label2hit
    qzl.dump_data((len_table,single_slot),sys.argv[3])
    for l in len_table:
        f1=[]
        for label in range(0,19):
            num_of_hit=sum([len2hit[_][label] if label in len2hit[_] else 0 for _ in len_table[len_table.index(l):]])
            num_of_pred=sum([len2pred[_][label] if label in len2pred[_] else 0 for _ in len_table[len_table.index(l):]])
            num_of_ans=sum([len2ans[_][label] if label in len2ans[_] else 0 for _ in len_table[len_table.index(l):]])
            f=num_of_hit/float(num_of_ans) if num_of_ans !=0 else 0
            p=num_of_hit/float(num_of_pred)if num_of_pred !=0 else 0
            f1.append(2*f*p/(f+p) if (f+p)!= 0 else 0)
        res.append((l,sum(f1)/len(filter(lambda x:x!=0,f1))))
    
    res_r=[[d[0] for d in res],[d[1] for d in res]]
    qzl.dump_data(res_r,sys.argv[2])
    
