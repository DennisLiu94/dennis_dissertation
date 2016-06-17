import sys
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import qzliu_util as qzl
def read_file(filename):
    with open(filename) as f:
        line=f.readline()
    line=line.replace('\n','')
    line=line[1:-1]
    arr=line.split('), (')
    for i in range(len(arr)):
        arr[i]=arr[i].replace('(','').replace(')','')
    return arr
def get_data_list(line):
    epochlist=[]
    valuelist=[]
    for i in line:
        nums=i.split(', ')
        epochlist.append(int(nums[0]))
        valuelist.append(float(nums[1]))
    return epochlist,valuelist

xlabel='epoch'
ylabel='F1 score'
title='performance of attentional pooling'







if __name__=='__main__':
    line_list=[read_file(filename) for filename in sys.argv[1:-1]]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
   
    sat6=line_list[0]
    #sat8=line_list[1]
    #sat10=line_list[2][:350]
    #sat12=line_list[3]
    
    def plot_fig(line,label,color):
        x,y=get_data_list(line)
        plt.plot(x,y,color,label=label)
    
    
    plot_fig(sat6,'length of chain:10','ro')
    #plot_fig(sat8,'aggrgation method 2','ro')
    #plot_fig(sat10,'aggrgation method 3','bo')
    #plot_fig(sat12,'three layers with dep feature','yo')
    #plot_fig(sat12,'extended sdp with dep tag','go')

    legend=plt.legend(loc='lower center',shadow=True,fontsize='x-large')


    legend.get_frame().set_facecolor('#00FFCC')




    plt.savefig(sys.argv[-1])
