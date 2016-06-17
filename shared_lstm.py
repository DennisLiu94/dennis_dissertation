"""
Small Theano LSTM recurrent network module.

@author: Jonathan Raiman
@date: December 10th 2014

Implements most of the great things that came out
in 2014 concerning recurrent neural networks, and
some good optimizers for these types of networks.

Note (from 5 January 2015): Dropout api is a bit sophisticated due to the way
random number generators are dealt with in Theano's scan.

"""

import theano, theano.tensor as T
import numpy as np
from collections import OrderedDict

srng = theano.tensor.shared_randomstreams.RandomStreams(1234)

from masked_loss import masked_loss, masked_loss_dx
from shared_memory import wrap_params, borrow_memory, borrow_all_memories

import pdb
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
      #  return theano.shared((np.random.standard_normal([out_size, in_size])* 1./out_size).astype(theano.config.floatX))


def Dropout(shape, prob):
    """
    Return a dropout mask on x.

    The probability of a value in x going to zero is prob.

    Inputs
    ------

    x    theano variable : the variable to add noise to
    prob float, variable : probability of dropping an element.
    size tuple(int, int) : size of the dropout mask.


    Outputs
    -------

    y    theano variable : x with the noise multiplied.

    """
    
    mask = srng.binomial(n=1, p=1-prob, size=shape)
    return T.cast(mask, theano.config.floatX)


def MultiDropout(shapes, dropout = 0.):
    """
    Return all the masks needed for dropout outside of a scan loop.
    """
    return [Dropout(shape, dropout) for shape in shapes]


class Layer(object):
    """
    Base object for neural network layers.

    A layer has an input set of neurons, and
    a hidden activation. The activation, f, is a
    function applied to the affine transformation
    of x by the connection matrix W, and the bias
    vector b.

    > y = f ( W * x + b )

    """
        
    def __init__(self, input_size, hidden_size, activation, clip_gradients=False):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.activation  = activation
        self.clip_gradients = clip_gradients
        self.is_recursive = False
        self.create_variables()
        
    def create_variables(self):
        """
        Create the connection matrix and the bias vector
        """
        self.linear_matrix        = create_shared(self.hidden_size, self.input_size)
        self.bias_matrix          = create_shared(self.hidden_size)

        

    def activate(self, x):
        """
        The hidden activation of the network
        """
        if self.clip_gradients is not False:
            x = clip_gradient(x, self.clip_gradients)

        if x.ndim > 1:
            return self.activation(
                T.dot(self.linear_matrix, x.T) + self.bias_matrix[:,None] ).T
        else:  
            return self.activation(
                T.dot(self.linear_matrix, x) + self.bias_matrix )

    @property
    def params(self):
        return [self.linear_matrix, self.bias_matrix]

    @params.setter
    def params(self, param_list):
        self.linear_matrix.set_value(param_list[0].get_value())
        self.bias_matrix.set_value(param_list[1].get_value())


class Embedding(Layer):
    def __init__(self, vocabulary_size, hidden_size):
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.create_variables()
        self.is_recursive = False
        
    def create_variables(self):
        self.embedding_matrix = create_shared(self.vocabulary_size, self.hidden_size)

    def activate(self, x):
        return self.embedding_matrix[x]

    @property
    def params(self):
        return [self.embedding_matrix]

    @params.setter
    def params(self, param_list):
        self.embedding_matrix.set_value(param_list[0].get_value())


class RNN(Layer):
    """
    Special recurrent layer than takes as input
    a hidden activation, h, from the past and
    an observation x.

    > y = f ( W * [x, h] + b )

    Note: x and h are concatenated in the activation.

    """
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.is_recursive = True
        
    def create_variables(self):
        """
        Create the connection matrix and the bias vector,
        and the base hidden activation.

        """
        self.linear_matrix        = create_shared(self.hidden_size, self.input_size+ self.hidden_size)
        self.bias_matrix          = create_shared(self.hidden_size)
        self.initial_hidden_state = create_shared(self.hidden_size)

    def activate(self, x, h):
        """
        The hidden activation of the network
        """
        if self.clip_gradients is not False:
            x = clip_gradient(x, self.clip_gradients)
            h = clip_gradient(h, self.clip_gradients)
        if x.ndim > 1:
            return self.activation(
                T.dot(
                    self.linear_matrix,
                    T.concatenate([x, h], axis=1).T
                ) + self.bias_matrix[:,None] ).T
        else:
            return self.activation(
                T.dot(
                    self.linear_matrix,
                    T.concatenate([x, h])
                ) + self.bias_matrix )

    @property
    def params(self):
        return [self.linear_matrix, self.bias_matrix,
                self.initial_hidden_state]

    @params.setter
    def params(self, param_list):
        self.linear_matrix.set_value(param_list[0].get_value())
        self.bias_matrix.set_value(param_list[1].get_value())
        self.initial_hidden_state.set_value(param_list[2].get_value())


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

    def __init__(self,input_size,hidden_size,activation,clip_gradients=False,in_p=None,in_gate_p=None,f_gate_p=None,out_gate_p=None,initial_state=None):
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
        in_x = self.layer_activate(obs,self.in_W,self.in_b,self.activation)
        
        # how much to forget the current contents of the memory
        forget_gate = self.layer_activate(obs,self.f_gate_W,self.f_gate_b,T.nnet.sigmoid)
        
        # modulate the input for the memory cells
        in_gate = self.layer_activate(obs,self.in_gate_W,self.in_gate_b,T.nnet.sigmoid)
        
        # new memory cells
        #shape of child_c and forget_gate should be (children_num,hidden_size)
        next_c = forget_gate *prev_c+  in_x*in_gate
        
        # modulate the memory cells to create the new output
        out_gate = self.layer_activate(obs,self.out_gate_W,self.out_gate_b,T.nnet.sigmoid)
        
        # new hidden output
        next_h = out_gate * T.tanh(next_c)
        if h.ndim>1:
            return T.concatenate([next_c,next_h],axis=1)
        else:
            return T.concatenate([next_c, next_h])

class GatedInput(RNN):
    def create_variables(self):
        # input gate for cells
        self.in_gate     = Layer(self.input_size + self.hidden_size, 1, T.nnet.sigmoid, self.clip_gradients)
        self.internal_layers = [self.in_gate]

    @property
    def params(self):
        """
        Parameters given by the 4 gates and the
        initial hidden activation of this LSTM cell
        layer.

        """
        return [param for layer in self.internal_layers
                for param in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

    def activate(self, x, h):
        # input and previous hidden constitute the actual
        # input to the LSTM:
        if h.ndim > 1:
            obs = T.concatenate([x, h], axis=1)
        else:
            obs = T.concatenate([x, h])

        gate = self.in_gate.activate(obs)
        if h.ndim > 1:
            gate = gate[:,0][:,None]
        else:
            gate = gate[0]

        return gate

    def postprocess_activation(self, gate, x, h):
        return gate * x


def apply_dropout(x, mask):
    if mask is not None:
        return mask * x
    else:
        return x


class StackedCells(object):
    """
    Sequentially connect several recurrent layers.

    celltypes can be RNN or LSTM.

    """
    def __init__(self, input_size, celltype=RNN, layers=None,
                 activation=lambda x:x, clip_gradients=False):
        if layers is None:
            layers = []
        self.input_size = input_size
        self.clip_gradients = clip_gradients
        self.create_layers(layers, activation, celltype)
        
    def create_layers(self, layer_sizes, activation_type, celltype):
        self.layers = []
        prev_size   = self.input_size
        for k, layer_size in enumerate(layer_sizes):
            layer = celltype(prev_size, layer_size, activation_type,
                             clip_gradients=self.clip_gradients)
            self.layers.append(layer)
            prev_size = layer_size
    
    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
            
    def forward(self, x, prev_hiddens=None, dropout=None):
        """
        Return new hidden activations for all stacked RNNs
        """
        #pdb.set_trace()
        if dropout is None:
            dropout = []
        if prev_hiddens is None:
            prev_hiddens = [(T.repeat(T.shape_padleft(layer.initial_hidden_state),
                                      x.shape[0], axis=0)
                             if x.ndim > 1 else layer.initial_hidden_state)
                            if hasattr(layer, 'initial_hidden_state') else None
                            for layer in self.layers]
       # pdb.set_trace()
       # if prev_hiddens[0].ndim<x.ndim:
       #     print 'ndim:',prev_hiddens[0].ndim,x.ndim
       #     prev_hiddens=[ T.repeat(T.shape_padleft(layer_h),50,axis=0)   for layer_h in prev_hiddens]

        out = []
        layer_input = x
        for k, layer in enumerate(self.layers):
            level_out = layer_input
            if len(dropout) > 0:
              #  pdb.set_trace()
               # mask=Dropout(level_out.shape,dropout[k])
               # level_out = apply_dropout(level_out, mask)
                level_out = apply_dropout(level_out, dropout[k])
            if layer.is_recursive:
                level_out = layer.activate(level_out, prev_hiddens[k])
            else:
                level_out = layer.activate(level_out)
            out.append(level_out)
            # deliberate choice to change the upward structure here
            # in an RNN, there is only one kind of hidden values
            if hasattr(layer, 'postprocess_activation'):
                # in this case the hidden activation has memory cells
                # that are not shared upwards
                # along with hidden activations that can be sent
                # updwards
                if layer.is_recursive:
                    level_out = layer.postprocess_activation(level_out, layer_input, prev_hiddens[k])
                else:
                    level_out = layer.postprocess_activation(level_out, layer_input)

            layer_input = level_out

        return out


def create_optimization_updates(cost, params, updates=None, max_norm=5.0,
                                lr=0.01, eps=1e-6, rho=0.95,
                                method = "adadelta"):
    """
    Get the updates for a gradient descent optimizer using
    SGD, AdaDelta, or AdaGrad.

    Returns the shared variables for the gradient caches,
    and the updates dictionary for compilation by a
    theano function.

    Inputs
    ------

    cost     theano variable : what to minimize
    params   list            : list of theano variables
                               with respect to which
                               the gradient is taken.
    max_norm float           : cap on excess gradients
    lr       float           : base learning rate for
                               adagrad and SGD
    eps      float           : numerical stability value
                               to not divide by zero
                               sometimes
    rho      float           : adadelta hyperparameter.
    method   str             : 'adagrad', 'adadelta', or 'sgd'.


    Outputs:
    --------

    updates  OrderedDict   : the updates to pass to a
                             theano function
    gsums    list          : gradient caches for Adagrad
                             and Adadelta
    xsums    list          : gradient caches for AdaDelta only
    lr       theano shared : learning rate
    max_norm theano_shared : normalizing clipping value for
                             excessive gradients (exploding).

    """
    lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
    eps = np.float64(eps).astype(theano.config.floatX)
    rho = theano.shared(np.float64(rho).astype(theano.config.floatX))
    if max_norm is not None and max_norm is not False:
        max_norm = theano.shared(np.float64(max_norm).astype(theano.config.floatX))

    gsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if (method == 'adadelta' or method == 'adagrad') else None for param in params]
    xsums   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) if method == 'adadelta' else None for param in params]

    gparams = T.grad(cost, params)

    if updates is None:
        updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
        # clip gradients if they get too big
        if max_norm is not None and max_norm is not False:
            grad_norm = gparam.norm(L=2)
            gparam = (T.minimum(max_norm, grad_norm)/ grad_norm) * gparam
        
        if method == 'adadelta':
            updates[gsum] = T.cast(rho * gsum + (1. - rho) * (gparam **2), theano.config.floatX)
            dparam = -T.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] = T.cast(rho * xsum + (1. - rho) * (dparam **2), theano.config.floatX)
            updates[param] = T.cast(param + dparam, theano.config.floatX)
        elif method == 'adagrad':
            updates[gsum] =  T.cast(gsum + (gparam ** 2), theano.config.floatX)
            updates[param] =  T.cast(param - lr * (gparam / (T.sqrt(updates[gsum] + eps))), theano.config.floatX)
        else:
            updates[param] = param - gparam * lr

    if method == 'adadelta':
        lr = rho

    return updates, gsums, xsums, lr, max_norm


__all__ = [
    "create_optimization_updates",
    "masked_loss",
    "masked_loss_dx",
    "clip_gradient",
    "create_shared",
    "Dropout",
    "apply_dropout",
    "StackedCells",
    "Layer",
    "LSTM",
    "RNN",
    "GatedInput",
    "Embedding",
    "MultiDropout",
    "wrap_params",
    "borrow_memory",
    "borrow_all_memories"
    ]
