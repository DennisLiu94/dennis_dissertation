import qzliu_util as qzl
import numpy as np

def read_pkl(f):
    att_list=qzl.load_data(f)

    res=np.concatenate(att_list,axis=0)
    return res

def get_best(matrix,n):
    res=[]
    for row in matrix:
        tmp=[tuple((i,row[i])) for i in range(95)]
        tmp=sorted(tmp,reverse=True,key=lambda x:x[1] )
        res.append([ele[0] for ele in tmp[:n]])
    res=np.asarray(res,dtype='int32')
    return res

if __name__=='__main__':
    attention_weight_path='data/attention_weight.pkl'
    attention_weight_array=read_pkl(attention_weight_path)
    best_ten=get_best(attention_weight_array,10)
    qzl.dump_data(best_ten,'data/best_ten.pkl')
