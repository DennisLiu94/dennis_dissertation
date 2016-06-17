import qzliu_util as qzl
import sys

if __name__=='__main__':
    base=qzl.load_data('data/accu_base.pkl')
    pred=qzl.load_data('data/accu_pred.pkl')
    res=[float(p)/b for b,p in zip(base[1],pred[1])]
    print len(res)
    qzl.dump_data((base[0],res),'data/accu_res.pkl')
