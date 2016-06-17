import qzliu_util as qzl

if __name__=='__main__':
    log1=qzl.load_data('log1')
    log2=qzl.load_data('log2')

    res=[]
    res_1=[]
    for k in log1.keys():
        res.append([k,log1[k],log2[k],0 if log1[k]>log2[k] else 1])
        res_1.append(log1[k]-log2[k])
    for line in res:
        print line
    qzl.dump_data(res_1,'data/pred_diff.pkl')
    qzl.dump_data([t[1] for t in res],'data/atten_p_list.pkl')
    qzl.dump_data([t[2] for t in res],'data/base_p_list.pkl')
    
