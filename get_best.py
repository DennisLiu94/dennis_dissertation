import draw_pics as dp
import sys

if __name__=='__main__':
    lines=dp.read_file(sys.argv[1])
    _,value_list=dp.get_data_list(lines)
    res=0
    n=0
    for _id,i in enumerate(value_list):
        res=i if i > res else res
        n=_id if i == res else n
    print res,n

