import draw_pics as dp
import sys

if __name__=='__main__':
    lines=dp.read_file(sys.argv[1])
    _,value_list=dp.get_data_list(lines)
    a=int(sys.argv[2])
    b=int(sys.argv[3])
    res=0
    for i in range(a,b):
        res+=value_list[i]
    res/=(b-a)
    print res

