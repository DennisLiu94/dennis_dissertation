import matplotlib
matplotlib.use('Agg')

import sys
from matplotlib import pyplot as plt
import qzliu_util as qzl
style=['ro','bo','co']
if __name__=='__main__':
    for i in range(1,len(sys.argv)-1):
        data_list=qzl.load_data(sys.argv[i])
        print data_list
        if type(data_list[0])==type([]):
            plt.plot(data_list[0],data_list[1],style[i-1])
        else:
            plt.plot(data_list,style[i-1])
    plt.savefig(sys.argv[-1])
