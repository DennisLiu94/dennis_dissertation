import theano
from theano import tensor as T
import model
import numpy as np
if __name__=='__main__':
    x=T.fvector()
    y=x.mean()
    x=x
    z=model.rectify(x)
    #z=T.ceil(z)
    f=theano.function(inputs=[x],outputs=[z])
    print f(np.asarray([0.1,0.2,0.3],dtype='float32'))
