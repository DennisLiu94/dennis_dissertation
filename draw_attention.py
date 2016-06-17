import matplotlib
matplotlib.use('Agg')
import sys
from matplotlib import pyplot as plt
def parse_list(line):
    line=line.replace(']','').replace('[','')
    arr=line.split(', ')
    return [float(ele) for ele in arr]

def parse_matrix(line):
    line=line.strip()
    line=line[1:-2]
    arr=line.split('], [')
    matrix=[]
    for line in arr:
        matrix.append(parse_list(line))
    return matrix
def plot_weight(weights):
    plt.plot(weights,'ro')
if __name__=='__main__':
    target_count=int(sys.argv[2])
    attention_file=open(sys.argv[1])
    while target_count>0:
        line=attention_file.readline()
        target_count-=1
    matrix=parse_matrix(attention_file.readline())
    plot_weight(matrix[3][:20])
    plt.savefig(sys.argv[3])
