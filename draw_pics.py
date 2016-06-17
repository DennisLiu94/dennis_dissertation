import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
if __name__=='__main__':
    line_list=[read_file(filename) for filename in sys.argv[1:-1]]
    def plot_fig(line,color):
        
        epochlist,valuelist=get_data_list(line)
        for i in range(len(valuelist)):
            valuelist[i]=valuelist[i]-70
            if valuelist[i]<0:
                valuelist[i]=0
        plt.plot(epochlist,valuelist,color)
    color = ['ro','go','bo','co']
    for num,line in enumerate(line_list):
        plot_fig(line,color[num])
    plt.savefig(sys.argv[-1])
