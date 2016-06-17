import qzliu_util as qzl
import numpy as np
import sys

def get_var(filename):
    data_list=qzl.load_data(filename)
    data=np.asarray(data_list)
    return np.var(data)

if __name__=='__main__':
    res=map(get_var,sys.argv[1:])
    print res
