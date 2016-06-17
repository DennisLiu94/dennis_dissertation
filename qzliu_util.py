import cPickle
def readlines(filename):
    with open(filename) as f:
        lines=f.readlines()
    return lines

def writelines(lines,filename):
    with open(filename,'w')as f:
	try:
	    for line in lines:
		f.write(line)
		f.write('\n')
	except:
	    print line
def load_data(filename):
    with open(filename) as f:
        obj=cPickle.load(f)
    return obj
def dump_data(obj,filename):
    with open(filename,'w') as f:
        cPickle.dump(obj,f)
def writematrix(obj,filename):
    with open(filename,'w')as f:
        for a in obj:
            for b in a:
                f.write(str(b))
                f.write(' ')
            f.write('\n')
