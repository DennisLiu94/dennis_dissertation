Using gpu device 0: Tesla K40c
Traceback (most recent call last):
  File "./pure_attention_before_lstm.py", line 4, in <module>
    from model import *
  File "/home/zhangshu/Qiuzhi/FINAL/model.py", line 10, in <module>
    import theano
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/__init__.py", line 98, in <module>
    theano.sandbox.cuda.tests.test_driver.test_nvidia_driver1()
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/sandbox/cuda/tests/test_driver.py", line 32, in test_nvidia_driver1
    profile=False)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/compile/function.py", line 266, in function
    profile=profile)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/compile/pfunc.py", line 511, in pfunc
    on_unused_input=on_unused_input)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/compile/function_module.py", line 1465, in orig_function
    on_unused_input=on_unused_input).create(
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/compile/function_module.py", line 1103, in __init__
    theano.gof.cc.get_module_cache().refresh()
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/cc.py", line 85, in get_module_cache
    return cmodule.get_module_cache(config.compiledir, init_args=init_args)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/cmodule.py", line 1390, in get_module_cache
    _module_cache = ModuleCache(dirname, **init_args)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/cmodule.py", line 617, in __init__
    self.refresh()
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/cmodule.py", line 724, in refresh
    key_data = cPickle.load(f)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/scalar/basic.py", line 3317, in __setstate__
    self.init_fgraph()
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/scalar/basic.py", line 3136, in init_fgraph
    gof.MergeOptimizer().optimize(fgraph)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/opt.py", line 77, in optimize
    self.add_requirements(fgraph)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/opt.py", line 582, in add_requirements
    fgraph.attach_feature(MergeFeature())
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/fg.py", line 531, in attach_feature
    attach(self)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/opt.py", line 468, in on_attach
    for node in fgraph.toposort():
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/fg.py", line 619, in toposort
    order = graph.io_toposort(fg.inputs, fg.outputs, ords)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/graph.py", line 804, in io_toposort
    topo = general_toposort(outputs, deps)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/graph.py", line 745, in general_toposort
    reachable, clients = stack_search(deque(r_out), _deps, 'dfs', True)
  File "/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/gof/graph.py", line 525, in stack_search
    expand_inv.setdefault(r, []).append(l)
KeyboardInterrupt
