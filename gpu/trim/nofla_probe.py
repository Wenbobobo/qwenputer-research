import sys
sys.modules['fla']=None
import runpy
from chipformer.q35.fast import exact_matmuls
exact_matmuls()
sys.argv=['chipformer.rv.gpucheck','--decode','--progs','signext']
runpy.run_module('chipformer.rv.gpucheck',run_name='__main__')
