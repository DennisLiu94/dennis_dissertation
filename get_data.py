import sys

def read_file(filename):
    with open(filename) as f:
        lines=f.readlines()
    return lines

def get_score_list(lines,name):
    name2line={}
    name2line['test']=-2
    name2line['train_loss']=-6
    return lines[name2line[name]]
def save_data(line,savepath):
    with open(savepath,'w') as f:
        f.write(line)
    return 0

if __name__=='__main__':
    filename=sys.argv[1]
    scorename=sys.argv[2]
    savepath=sys.argv[3]

    lines=read_file(filename)
    line=get_score_list(lines,scorename)
    save_data(line,savepath)
