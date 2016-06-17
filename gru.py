import os
import sys
import time
import copy
import cPickle
import pdb
import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from collections import OrderedDict
from util import *

#utils-------------------

def rectify(X):
    return T.maximum(X,0.)

    # Jobman channel remplacement
def p_rectify(X,a):
    return T.maximum(X,0.)+T.minimum(X,0.)*a

class Channel(object):
    def __init__(self, state):
        self.state = state
        f = open(self.state.savepath + '/orig_state.pkl', 'w')
        cPickle.dump(self.state, f, -1)
        f.close()
        self.COMPLETE = 1

    def save(self):
        f = open(self.state.savepath + '/current_state.pkl', 'w')
        cPickle.dump(self.state, f, -1)
        f.close()


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def sgd_updates_adadelta(params,cost,rho=0.7,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    debug_dict={}
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        learning_rate=(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon))
        step =-learning_rate * gp
        debug_dict[param]=learning_rate
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        updates[param] = stepped_param    
        #norm_lim=1
        #normalize=False
        #if normalize and (param.get_value(borrow=True).ndim == 2) and (param.name.count('emb')==0):
        #    col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
        #    desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
        #    scale = desired_norms / (1e-7 + col_norms)
        #    updates[param] = stepped_param * scale
       
    return (updates,debug_dict)


class GradClip(theano.compile.ViewOp):
    """
    Here we clip the gradients as Alex Graves does in his
    recurrent neural networks. In particular this prevents
    explosion of gradients during backpropagation.

    The original poster of this code was Alex Lamb,
    [here](https://groups.google.com/forum/#!topic/theano-dev/GaJwGw6emK0).

    """

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]


def clip_gradient(x, bound):
    grad_clip = GradClip(-bound, bound)
    try:
        T.opt.register_canonicalize(theano.gof.OpRemove(grad_clip), name='grad_clip_%.1f' % (bound))
    except ValueError:
        pass
    return grad_clip(x)


def create_shared(out_size, in_size = None):
    """
    Creates a shared matrix or vector
    using the given in_size and out_size.

    Inputs
    ------

    out_size int            : outer dimension of the
                              vector or matrix
    in_size  int (optional) : for a matrix, the inner
                              dimension.

    Outputs
    -------

    theano shared : the shared matrix, with random numbers in it

    """

    if in_size is None:
        W_value=np.asarray(
                    np.random.uniform(
                        low=-np.sqrt(6.0/out_size),
                        high=np.sqrt(6.0/out_size),
                        size=(out_size,)
                        ),
                    dtype=theano.config.floatX
        )
        return theano.shared(value=W_value)
       # return theano.shared((np.random.standard_normal([out_size])* 1./out_size).astype(theano.config.floatX))
    else:
        W_value=np.asarray(
                    np.random.uniform(
                        low=-np.sqrt(6.0/(in_size+out_size)),
                        high=np.sqrt(6.0/(in_size+out_size)),
                        size=(out_size,in_size)
                        ),
                    dtype=theano.config.floatX
        )
        return theano.shared(value=W_value)


#---------------------------------------
# Embeddings class -----------------------------------------------------------
class Embeddings(object):
    """Class for the embeddings matrix."""

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings.
        :param tag: name of the embeddings for parameter declaration.
        every row is one embedding
        """
        self.N = N
        self.D = D
        self.fan_in=self.D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(N, D))
        W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=1)).reshape(N,1)
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # embedding vectors.
        self.mode1_updates = {self.E:self.E/T.sqrt((self.E**2).sum(axis=1)).dimshuffle(0,'x')}
        self.mode1_normalize = theano.function([], [], updates=self.mode1_updates)
        # std version normalization
        self.std_updates={self.E:0.1*self.E/T.std(self.E)}
        self.std_normalize=theano.function([],[],updates=self.std_updates)
        self.params=[self.E]    
    def load_init(self,idx2emb):
        '''
        load init embedding by a dict idx2embedding
        '''
        W_values=self.E.get_value()
        #for idx in idx2emb.keys():
        for idx in range(self.N):
            if idx2emb.has_key(idx):
                assert self.D==len(idx2emb[idx])
                W_values[idx]=idx2emb[idx]
            else:
                W_values[idx]=idx2emb[0]#use "UNKNOW* to init
        self.E.set_value(W_values)
        print 'set embedding for %d word in all embedding row:%d'%(len(idx2emb),self.N)
# ----------------------------------------------------------------------------

# Layers ----------------------------------------------------------------------
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out,W=None,b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.fan_in=n_in
        if W is None:
            self.W = theano.shared(
                value=np.asarray(
                        np.random.uniform(
                            low=-np.sqrt(6./(n_in+n_out)),
                            high=np.sqrt(6./(n_in+n_out)),
                            size=(n_in,n_out)
                        )*4,
                        dtype=theano.config.floatX
                ),
                #value=np.zeros((n_in,n_out),dtype=theano.config.floatX),
                name='W',
                borrow=True
            )
        else:
            self.W=W
        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                 value=np.zeros(
                     (n_out,),
                  dtype=theano.config.floatX
                 ),
                name='b',
                borrow=True
             )
        else:
            self.b=b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activation=T.tanh):
        '''
            Typical hidden layer of a MLP:units are fully-connected and hava sigmod
            activation functionj.Weight matrix w is of shape(n_in,n_out) and the 
            bias vactor b is of shape(n_out)
            
            NOTE:the nonlinearity used here is tanh

            Hidden unit activation is given by:tanh(dot(input,W)+b)

            :type rng:numpy.random.RandomState
            :param rng:a random number generator used to initialize weights
    
            :type input:theano.tensor.dmatrix
            :param input:a symbolic tensor of shape(n_examples,n_in)
    
            :type n_in:int 
            :type n_out:int
    
            :type activation:theano.Op or function
            :param activation:Non linearity to be applied in the hidden layer
        '''
        self.input=input
        self.fan_in=n_in
            #'W' si initialized with 'W_values' which is uniformely sampled form sqrt(-6./(n_in+n_hidden))
            #and sqrt(6./(n_in+n_hidden)) for tanh activation function  
            #the outpu of uniform if converted using asarray to dtype theano.config.floatX so that the code is runable on GPU
            #Note:optimal initialization of weights is dependents on the activation function used (among other things).
                #for example,results presented in [Xavier10] suggest that you should use 4 times larger initial weights for 
                #sigmoid compared to tanh. we have no info for other function,so we use the same as tanh.
        if W is None:
            W_values=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6./(n_in+n_out)),
                        high=np.sqrt(6./(n_in+n_out)),
                        size=(n_in,n_out)
                        ),
                    dtype=theano.config.floatX
            )
            if activation==theano.tensor.nnet.sigmoid:
                W_values*=4
            W=theano.shared(value=W_values,name='W',borrow=True)
        if b is None:
            b_values=np.zeros((n_out,),dtype=theano.config.floatX)
            b=theano.shared(value=b_values,name='b',borrow=True)

        self.W=W
        self.b=b
        self.params=[self.W,self.b]
        
        self.line_output=T.dot(input,self.W)+self.b
        self.output=(self.line_output if activation is None else activation(self.line_output))
        #parameters of the model

        ###################
        # p_rectify begins
        #a_values=np.zeros((n_out,),dtype=theano.config.floatX)+0.25
        #self.a=theano.shared(a_values,name='a',borrow=True)
        #self.output=p_rectify(line_output,self.a)
        #self.params=[self.W,self.b,self.a]
        # p_rectify end

def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
"""
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 dropout_rate, activation,W=None,b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

class DropConnectHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 dropout_rate, activation,W=None,b=None):
        super(DropConnectHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)
        W=_dropout_from_layer(rng,self.W, p=dropout_rate)
        b=_dropout_from_layer(rng,self.b, p=dropout_rate)
        self.line_output =T.dot(input,W)+b
        self.output=(self.line_output if activation is None else activation(self.line_output))

class MaxoutLayer(object):
    def __init__(self,rng,input,n_in,n_out,k=2,W=None,b=None):
        '''
            :param activation:Non linearity to be applied in the hidden layer
        '''
        self.input=input
        self.fan_in=n_in
        self.k=k
        if W is None:
            W_values=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6./(n_in+n_out)),
                        high=np.sqrt(6./(n_in+n_out)),
                        size=(self.k,n_in,n_out)
                        ),
                    dtype=theano.config.floatX
            )
            W=theano.shared(value=W_values,name='W',borrow=True)
        if b is None:
            b_values=np.zeros((self.k,n_out),dtype=theano.config.floatX)
            b=theano.shared(value=b_values,name='b',borrow=True)

        self.W=W
        self.b=b
        self.params=[self.W,self.b]
        
        self.line_output=T.dot(input,self.W)+self.b
        self.output=self.line_output.max(axis=1)

class DropoutMaxoutLayer(MaxoutLayer):
    def __init__(self, rng, input, n_in, n_out,k,dropout_rate,W=None,b=None):
        super(DropoutMaxoutLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out,k=k, W=W, b=b)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class LSTM(object):
    """
    The structure of the LSTM allows it to learn on problems with
    long term dependencies relatively easily. The "long term"
    memory is stored in a vector of memory cells c.
    Although many LSTM architectures differ in their connectivity
    structure and activation functions, all LSTM architectures have
    memory cells that are suitable for storing information for long
    periods of time. Here we implement the LSTM from Graves et al.
    (2013).
    """
    
    @staticmethod
    def lstm_share_params(lstm):
        result=LSTM(
                input_size=lstm.input_size,
                hidden_size=lstm.hidden_size,
                activation=lstm.activation,
                clip_gradients=lstm.clip_gradients,
                in_p=[lstm.in_W,lstm.in_b],
                in_gate_p=[lstm.in_gate_W,lstm.in_gate_b],
                f_gate_p=[lstm.f_gate_W,lstm.f_gate_b],
                out_gate_p=[lstm.out_gate_W,lstm.out_gate_b],
                initial_state=lstm.initial_hidden_state
              )
        #result.result.set_value(result.result.get_value()+i)
        return result

    def __init__(self,input_size,hidden_size,activation,clip_gradients=10,in_p=None,in_gate_p=None,f_gate_p=None,out_gate_p=None,initial_state=None):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.activation=activation
        self.clip_gradients=clip_gradients
        self.is_recursive=True
        # input weight 
        if in_p is None:
            self.in_W=create_shared(self.hidden_size,self.input_size+self.hidden_size)
            self.in_b= create_shared(self.hidden_size)
        else:
            self.in_W=in_p[0]
            self.in_b=in_p[1]
        #input gate
        if in_gate_p is None:
            self.in_gate_W=create_shared(self.hidden_size,self.input_size+self.hidden_size)
            self.in_gate_b= create_shared(self.hidden_size)
        else:
            self.in_gate_W=in_gate_p[0]
            self.in_gate_b=in_gate_p[1]
        # forgate gate
        if f_gate_p is None:
            self.f_gate_W=create_shared(self.hidden_size,self.input_size+self.hidden_size)
            self.f_gate_b= create_shared(self.hidden_size)
        else:
            self.f_gate_W=f_gate_p[0]
            self.f_gate_b=f_gate_p[1]
        # output gate
        if out_gate_p is None:
            self.out_gate_W=create_shared(self.hidden_size,self.input_size+self.hidden_size)
            self.out_gate_b= create_shared(self.hidden_size)
        else:
            self.out_gate_W=out_gate_p[0]
            self.out_gate_b=out_gate_p[1]
        if initial_state is None:
            self.initial_hidden_state = create_shared(self.hidden_size * 2)
        else:
            self.initial_hidden_state = initial_state
        #self.result=T.zeros_like(self.initial_hidden_state)
        #self.result=theano.shared(value=np.zeros((self.hidden_size*2,)))
        self.params=[self.initial_hidden_state,self.in_W,self.in_b,self.in_gate_W,self.in_gate_b,self.f_gate_W,self.f_gate_b,self.out_gate_W,self.out_gate_b]
        self.L1=(
                abs(self.in_W).sum()+abs(self.in_gate_W).sum()+abs(self.f_gate_W).sum()+abs(self.out_gate_W).sum()
                )
        self.L2=(
                abs(self.in_W**2).sum()+abs(self.in_gate_W**2).sum()+abs(self.f_gate_W**2).sum()+abs(self.out_gate_W**2).sum()
                )

    #@params.setter
    #def params(self, param_list):
    #    pass
        #self.initial_hidden_state.set_value(param_list[0].get_value())
        #start = 1
        #for layer in self.internal_layers:
            #end = start + len(layer.params)
            #layer.params = param_list[start:end]
            #start = end

    def postprocess_activation(self, x, *args):
        if x.ndim > 1:
            return x[:, self.hidden_size:]
        else:
            return x[self.hidden_size:]
   
    def layer_activate(self,x,linear_matrix,bias_matrix,activation):
        if self.clip_gradients is not False:
            print 'clip gradents!'
            print self.clip_gradients
            x = clip_gradient(x, self.clip_gradients)
        if x.ndim > 1:
            return activation((
                T.dot(linear_matrix, x.T) + bias_matrix[:,None] ).T)
        else:  
            return activation(
                T.dot(linear_matrix, x) + bias_matrix )


    def activate(self, x, h):
        """
        The hidden activation, h, of the network, along
        with the new values for the memory cells, c,
        Both are concatenated as follows:

        >      y = f( x, past )

        Or more visibly, with past = [prev_c, prev_h]

        > [c, h] = f( x, [prev_c, prev_h] )

        """
        if h.ndim>1:
            
            prev_c = h[:, :self.hidden_size]
            prev_h = h[:, self.hidden_size:]
        else:
            prev_c=h[:self.hidden_size]
            prev_h=h[self.hidden_size:]
        
        if h.ndim>1:
            obs=T.concatenate([x,prev_h],axis=1)
        else:
            obs = T.concatenate([x, prev_h])
        
        
        # how much to add to the memory cells
        #in_gate = self.in_gate.activate(obs)
        self.in_x = self.layer_activate(obs,self.in_W,self.in_b,self.activation)
        
        # how much to forget the current contents of the memory
        self.forget_gate = self.layer_activate(obs,self.f_gate_W,self.f_gate_b,T.nnet.sigmoid)
        
        # modulate the input for the memory cells
        self.in_gate = self.layer_activate(obs,self.in_gate_W,self.in_gate_b,T.nnet.sigmoid)
        
        # new memory cells
        #shape of child_c and forget_gate should be (children_num,hidden_size)
        next_c = self.forget_gate *prev_c+  self.in_x*self.in_gate
        
        # modulate the memory cells to create the new output
        self.out_gate = self.layer_activate(obs,self.out_gate_W,self.out_gate_b,T.nnet.sigmoid)
        
        # new hidden output
        next_h = self.out_gate * T.tanh(next_c)
        if h.ndim>1:
            self.gate=T.concatenate([self.in_x,self.in_gate,self.forget_gate,self.out_gate],axis=1)
            #return T.concatenate([next_c,next_h],axis=1)
            return T.concatenate([T.concatenate([next_c,next_h],axis=1)],axis=1)
        else:
            self.gate=T.concatenate([self.in_x,self.in_gate,self.forget_gate,self.out_gate])
            return T.concatenate((T.concatenate([next_c, next_h]),self.gate))


# Gated Recurrent Unit
# Recently proposed: 
# K. Cho, B. van Merrienboer,
# D. Bahdanau, and Y. Bengio.
# On the properties of neural 
# machine translation: Encoder-decoder approaches. 
class GRU(object):
    def __init__(self,input_size, hidden_size,activation,clip_gradients=10):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.n_u = int(input_size)
        self.n_h = int(hidden_size)
        self.activation=activation
        self.clip_gradients=clip_gradients
        # Here we define 6 weight matrices and 3 bias vector.
        # To determine the dimention of weight matrices and bias vectors,
        # we just need to have the following numbers: n_u, n_h
        # (You can check it using the equations).
        #
        # Thus:
        # <var>: <dimention>
        #
        # W_xz : n_h  x  n_u
        # W_hz : n_h  x  n_h
        # W_xr : n_h  x  n_u
        # W_hr : n_h  x  n_h
        # W_xh : n_h  x  n_h
        # W_hh : n_h  x  n_h
        #
        # b_z : n_h  x  1
        # b_r : n_h  x  1
        # b_h : n_h  x  1

        # Update gate weights
        self.W_xz=create_shared(self.n_h,self.n_u)
        self.W_hz=create_shared(self.n_h,self.n_h)
        self.b_z=create_shared(self.n_h)
        # Reset gate weights
        self.W_xr=create_shared(self.n_h,self.n_u)
        self.W_hr=create_shared(self.n_h,self.n_h)
        self.b_r=create_shared(self.n_h)
        # Other weights :-)
        self.W_xh=create_shared(self.n_h,self.n_u)
        self.W_hh=create_shared(self.n_h,self.n_h)
        self.b_h=create_shared(self.n_h)
       
        #initial hidden state
        self.initial_hidden_state=create_shared(self.n_h)
        
        self.params = [self.W_xz, self.W_hz, self.W_xr, self.W_hr, 
                          self.W_xh, self.W_hh, self.b_z, self.b_r, 
                          self.b_h,self.initial_hidden_state]


    def gru_activate(self, x_t, h_tm1):
        # update gate
        if self.clip_gradients is not False:
            print 'clip gradients %d'%self.clip_gradients
            x_t=clip_gradient(x_t,self.clip_gradients)
            h_tm1=clip_gradient(h_tm1,self.clip_gradients)
        if x_t.ndim>1:
            print x_t.ndim
            z_t = T.nnet.sigmoid(T.dot(self.W_xz, x_t.T) + \
                             T.dot(self.W_hz, h_tm1.T) + \
                             self.b_z[:,None]).T
        # reset gate
            r_t = T.nnet.sigmoid(T.dot(self.W_xr, x_t.T) + \
                             T.dot(self.W_hr, h_tm1.T) + \
                             self.b_r[:,None]).T
        # candidate h_t
            can_h_t = self.activation(T.dot(self.W_xh, x_t.T) + \
                         (r_t.T) * T.dot(self.W_hh, h_tm1.T) + \
                         self.b_h[:,None]).T
        else:
            print x_t.ndim
            z_t = T.nnet.sigmoid(T.dot(self.W_xz, x_t) + \
                             T.dot(self.W_hz, h_tm1) + \
                             self.b_z)
        # reset gate
            r_t = T.nnet.sigmoid(T.dot(self.W_xr, x_t) + \
                             T.dot(self.W_hr, h_tm1) + \
                             self.b_r)
        # candidate h_t
            can_h_t = self.activation(T.dot(self.W_xh, x_t) + \
                         r_t * T.dot(self.W_hh, h_tm1) + \
                         self.b_h)
        # h_t
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        return h_t

class ColingConvLayer(object):
    def __init__(self,rng,input,n0,n1,W=None):
        '''
            simple modification of MLP HiddenLayer class. corresponding to section3.4.3-Convolution
            all symbols(n0,n1,W) are consitent with papper

            :type rng:numpy.random.RandomState
            :param rng:a random number generator used to initialize weights
    
            :type input:theano.tensor.dmatrix(3D)
            :param input:a symbolic tensor of shape(n_examples,n0,t)(n0=w*n)
    
            :type n0,t,n1:int 
            :type n0:w*n
    
            :type activation:theano.Op or function
            :param activation:Non linearity to be applied in the ConvLayer layer
        '''
        self.input=input
        self.fan_in=n0
        assert self.input.ndim==3,'input must be 3 dim matrix(n_examples,n0,t)' 
        if W is None:
            W_values=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6./(n1+n0)),
                        high=np.sqrt(6./(n1+n0)),
                        size=(n1,n0)
                        ),
                    dtype=theano.config.floatX
            )
            W=theano.shared(value=W_values,name='W',borrow=True)
        self.W=W
        self.params=[self.W]

        assert self.W is not None
        #input is a list of sample(A)
        #we need list of WA,but we cant not use T.dot(self.W,input) get what we need 
        #because the matrix with higher dim must come first ,meaning T.dot(input,self.W)
        # so we use the trans rule of matix WA=(A'W')'
        
        #TODO: try T.dot(self.W.dimshuffle(0,1,'x'),input)

        #get list of A'
        input_T=input.dimshuffle(0,2,1)
        #get W'
        W_T=self.W.T
        #get list of (A'W')'
        self.con_output=T.dot(input_T,W_T).dimshuffle(0,2,1)
        
        #######################
        # p_rectify begin
        #a_values=np.zeros((n1,60),dtype=theano.config.floatX)+0.25
        #self.a=theano.shared(a_values,name='a',borrow=True)
        #self.con_output=p_rectify(self.con_output,self.a)
        #self.params=[self.W,self.a]
        # p_rectify end
        
        #######################
        # p_max begin
       # p_values=np.zeros((n1,),dtype=theano.config.floatX)-0.1
       # self.p=theano.shared(p_values,name='conv_p',borrow=True)
       # self.params=[self.W,self.p]
       # max_out=self.con_output.max(axis=2)
       # min_out=self.con_output.min(axis=2)
       # self.output=max_out+min_out*self.p
       # self.output=T.concatenate([max_out,min_out],axis=1)
        
        # p max end
        pooling_output=self.con_output.max(axis=2)
        self.output=pooling_output




# start-snippet-1
class dA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]


# start-snippet-1
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /numpy.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape), 
                                                dtype=theano.config.floatX),borrow=True,name="W_conv")
        else:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),borrow=True,name="W_conv")   
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]
        
    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        if self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output
        

# Theano functions creation --------------------------------------------------
