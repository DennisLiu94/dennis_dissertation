import qzliu_util as qzl
import sys

if __name__=='__main__':
    data_set=qzl.load_data(sys.argv[1])
    print type(data_set['train']['pos_x'])
