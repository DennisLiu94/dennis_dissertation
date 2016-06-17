import sys
import os

run_file=sys.argv[1]
gpu=sys.argv[2]
name=run_file[:-3]

os.system('thpy%s %s > qz_res/%s.log 2>qz_res/%s.err'%(gpu,run_file,name,name))
