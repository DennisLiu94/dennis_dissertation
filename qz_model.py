import sys
import numpy as np
import theano
import theano.tensor as T
from model import *


def simple_max(input,size):	
    res_size=size
    res_output=input.max(axis=1)
    return res_output,res_size
class qz_blstm(object):
    def __init__(self,rng,input_size,activation,hidden_size):
        self.hidden_size=hidden_size
        self.forward_model=LSTM(input_size,hidden_size=hidden_size,activation=activation)
        self.backward_model=LSTM(input_size,hidden_size=hidden_size,activation=activation)
        self.params=self.forward_model.params+self.backward_model.params

        

    def forward_step(self,x,prev_hiddens):
        new_state=self.forward_model.activate(x,prev_hiddens)
        return new_state
    def backward_step(self,x,prev_hiddens):
        new_state=self.backward_model.activate(x,prev_hiddens)
        return new_state

    def get_output(self,input):
        self.batch_size=input.shape[1]
        
        self.forward_output_info=[]
        shared_h=self.forward_model.initial_hidden_state
        shared_h=T.repeat(T.shape_padleft(shared_h),self.batch_size,axis=0)
        self.forward_output_info.append(shared_h)

        self.backward_output_info=[]
        shared_h=self.backward_model.initial_hidden_state
        shared_h=T.repeat(T.shape_padleft(shared_h),self.batch_size,axis=0)
        self.backward_output_info.append(shared_h)

        forward_input=input
        backward_input=input[::-1]
        forward_result,updates=theano.scan(
            self.forward_step,
            sequences=[forward_input],
            outputs_info=self.forward_output_info,
        )
        backward_result,updates=theano.scan(
            self.backward_step,
            sequences=[backward_input],
            outputs_info=self.backward_output_info,
        )
        self.all_output=T.concatenate([forward_result,backward_result[::-1]],axis=2)
        self.output=self.all_output.max(axis=0)
        self.output_size=self.hidden_size*4
        return self.output,self.output_size

class attention_chain(object):
    def __init__(self,rng,n,n_out,act):
        n_in=n
        activation=act
        W_values=np.asarray(
            rng.uniform(
                low=-np.sqrt(6./(n_in+n_out)),
                high=np.sqrt(6.0/(n_in+n_out)),
                size=(n_in,n_out)
            ),
            dtype=theano.config.floatX
        )

        if activation==theano.tensor.nnet.sigmoid:
            W_values*=4
        W=theano.shared(value=W_values,name='attention_W',borrow=True)

        b_values=np.zeros((n_out,),dtype=theano.config.floatX)
        b=theano.shared(value=b_values,name='attention_b',borrow=True)

        self.W=W
        self.b=b
        self.params=[self.W,self.b]
        self.activation=activation
    def get_output(self,input):
        line_output=T.dot(input,self.W)+self.b
        return line_output if self.activation is None else self.activation(line_output)
        

class glimpse_network(object):
    
    def __init__(self,sen_input,sen_size,e_pos=None,sen_len=None,rng=None,hidden_size=None):
        self.compress_layer=HiddenLayer(
                    rng=rng,
                    input=sen_input,
                    n_in=sen_size,
                    n_out=20,
                    activation=rectify
                )
        not_clear=self.compress_layer.output
        not_clear_size=20
        glimpse_size=20
        glimpse_pad_value=T.zeros_like(not_clear[:,:,:glimpse_size])
        
        self.attention_layer=attention_chain(rng,(not_clear_size+glimpse_size),1,T.tanh)
        attention_output_1=self.attention_layer.get_output(T.concatenate([not_clear,glimpse_pad_value],axis=2))
        pos_value_1=attention_output_1.argmax(axis=1)

        selected_sen_1=self.attention_pooling(sen_input,pos_value_1)

        self.blstm_layer=qz_blstm(rng=rng,input_size=sen_size,activation=rectify,hidden_size=hidden_size)
        self.lstm_output_1,self.lstm_output_size_1=self.blstm_layer.get_output(selected_sen_1.dimshuffle(1,0,2))
        self.params=self.blstm_layer.params
    

        
    def get_output(self):
        return self.lstm_output_1,self.lstm_output_size_1
def attention_pooling(sen_input,pos_value):
    

    def step(sen,idx):
        return sen[idx,:]
    output,updates=theano.scan(step,sequences=[sen_input,pos_value])
    return output
'''
def attention_pooling_n_best(sen_input,pos_value,n):
    pos_value_pad=T.zeros_like(pos_value[:,0,:])
    res=[]
    tmp_pos_value=T.shared(value=pos_value)
    def step(sen,idx):
        
        return T.concatenate([sen[idx,:]])
    for i in range(n):
        tmp_max=tmp_pos_value.argmax(axis=1)
        tmp_res,updates=theano.scan(step,sequences=[sen_input,tmp_max])
        res.append(tmp_res)
        tmp_pos_value=
'''

class attention_chain_shallow(object):
    def __init__(self,lstm_output,lstm_size,e_pos,sen_len,rng):
        self.attention_layer_list=[]
        self.glimpse_list=[]
        attention_layer_1=HiddenLayer(
                    rng=rng,
                    input=lstm_output,
                    n_in=lstm_size,
                    n_out=lstm_size,
                    activation=T.tanh
                )

        self.attention_layer_list.append(attention_layer_1)
        glimpse_1=self.attention_pooling(lstm_output,self.attention_layer_list[-1].output,e_pos,sen_len)
        self.glimpse_list.append(glimpse_1)
        '''
        attention_layer_2=HiddenLayer(
                rng=rng,    
                input=T.concatenate([lstm_output,self.glimpse_list[-1].dimshuffle('x',0,1),self.attention_layer_list[-1].output],axis=2),
                    n_in=lstm_size*3,
                    n_out=lstm_size,
                    activation=T.tanh
                )

        glimpse_2=self.attention_pooling(lstm_output,self.attention_layer_list[-1].output,e_pos,sen_len)
        self.glimpse_list.append(glimpse_2)

        attention_layer_3=HiddenLayer(
                    rng=rng,
                    input=T.concatenate([lstm_output,self.glimpse_list[-1].dimshuffle('x',0,1),self.attention_layer_list[-1].output],axis=2),
                    n_in=lstm_size*3,
                    n_out=lstm_size,
                    activation=T.tanh
                )

        glimpse_3=self.attention_pooling(lstm_output,self.attention_layer_list[-1].output,e_pos,sen_len)
        self.glimpse_list.append(glimpse_3)
        '''
        self.output=T.concatenate(self.glimpse_list,axis=1)
	self.output_size=lstm_size*len(self.glimpse_list)
    def attention_pooling(self,lstm_output,attention_value,e_pos,sen_len):
        def step(h,e,s_len,atten):
            e1_index=e[0]
            e2_index=e[1]
            # 3 segment, [0-e1);(e1-e2);(e2,end_of_sen)
            pad=T.zeros_like(h[:,0])
            pad_min=h.min(axis=1).dimshuffle(0,'x') 
            area1=T.concatenate([pad_min,atten[:,:e2_index],pad_min],axis=1)
            h_area1=T.concatenate([pad_min,h[:,:e2_index],pad_min],axis=1)
            #area2=h[:,e2_index:s_len+1]
            area2=T.concatenate([atten[:,e2_index:s_len],pad_min],axis=1) 
            h_area2=T.concatenate([h[:,e2_index:s_len],pad_min],axis=1) 
            area=T.concatenate([area1,area2],axis=1)
            h_area=T.concatenate([h_area1,h_area2],axis=1)
            new_e1_index=e1_index+1
            new_e2_index=e2_index+2
            new_s_len=s_len+2+1 #the last 1 is natrual pad from the sentence,but make sure sen_len < maxlen
            
            #concatenate [0,e1_index) (e1_index,e2_index). skip e1
            max1_area=T.concatenate([area[:,:new_e1_index],area[:,new_e1_index+1:new_e2_index],area[:,new_e2_index+1:new_s_len]],axis=1)
            h_max1_area=T.concatenate([h_area[:,:new_e1_index],h_area[:,new_e1_index+1:new_e2_index],h_area[:,new_e2_index+1:new_s_len]],axis=1)
            #max1_area.shape[1]==2 mean no word before e2 other than e1,only 2 pad left if remove e1 from the area, condition can also be new_e2_index==3
            
            max1=max1_area.argmax(axis=1)

            def s(h,m):
                print m.shape
                return h[m]
            res,updates=theano.scan(s,sequences=[h_max1_area,max1])
            
            #concateante (e1_index,e2_index) (e2_index,s_len) skip e2
            #max2_area=T.concatenate([area[:,new_e1_index+1:new_e2_index],area[:,new_e2_index+1:new_s_len]],axis=1)
            #same as max1, if max2_area.shape[1]==2 mean no word after e1 other than e2, only 2 pad left if remove e2 from thea area (one pad is padmin,one is from sentence pad),condition can also e1_index+2=s_len
            #max2=T.switch(T.gt(max2_area.shape[1],2),max2_area.max(axis=1),pad)
           
            #total_max=T.concatenate([max1_area,max2_area],axis=1).max(axis=1)
            '''
            if add_total:
                max_result=T.concatenate([max1,max2,total_max])
            else:
                max_result=T.concatenate([max1,max2])
            '''
            return res
        result,update=theano.scan(
                                step,
                                sequences=[lstm_output.dimshuffle(1,2,0),e_pos,sen_len,attention_value.dimshuffle(1,2,0)],
                                outputs_info=None
                               )
	return result	

'''
class attention_chain(object):
    def __init__(self,lstm_output,lstm_size,e_pos,sen_len,rng):
        self.attention_layer_list=[]
        self.glimpse_list=[]
        attention_layer_1=HiddenLayer(
                    rng=rng,
                    input=lstm_output,
                    n_in=lstm_size,
                    n_out=lstm_size,
                    activation=T.tanh
                )
        self.attention_layer_list.append(attention_layer_1)
        glimpse_1=self.attention_pooling(lstm_output,self.attention_layer_list[-1].output,e_pos,sen_len)
        self.glimpse_list.append(glimpse_1)
        attention_layer_2=HiddenLayer(
                rng=rng,    
                input=T.concatenate([lstm_output,self.glimpse_list[-1].dimshuffle('x',0,1),self.attention_layer_list[-1].output],axis=2),
                    n_in=lstm_size*3,
                    n_out=lstm_size,
                    activation=T.tanh
                )

        glimpse_2=self.attention_pooling(lstm_output,self.attention_layer_list[-1].output,e_pos,sen_len)
        self.glimpse_list.append(glimpse_2)

        attention_layer_3=HiddenLayer(
                    rng=rng,
                    input=T.concatenate([lstm_output,self.glimpse_list[-1].dimshuffle('x',0,1),self.attention_layer_list[-1].output],axis=2),
                    n_in=lstm_size*3,
                    n_out=lstm_size,
                    activation=T.tanh
                )

        glimpse_3=self.attention_pooling(lstm_output,self.attention_layer_list[-1].output,e_pos,sen_len)
        self.glimpse_list.append(glimpse_3)
        self.output=T.concatenate(self.glimpse_list,axis=1)
	self.output_size=lstm_size*len(self.glimpse_list)
    def attention_pooling(self,lstm_output,attention_value,e_pos,sen_len):
        def step(h,e,s_len,atten):
            e1_index=e[0]
            e2_index=e[1]
            # 3 segment, [0-e1);(e1-e2);(e2,end_of_sen)
            pad=T.zeros_like(h[:,0])
            pad_min=h.min(axis=1).dimshuffle(0,'x') 
            area1=T.concatenate([pad_min,atten[:,:e2_index],pad_min],axis=1)
            h_area1=T.concatenate([pad_min,h[:,:e2_index],pad_min],axis=1)
            #area2=h[:,e2_index:s_len+1]
            area2=T.concatenate([atten[:,e2_index:s_len],pad_min],axis=1) 
            h_area2=T.concatenate([h[:,e2_index:s_len],pad_min],axis=1) 
            area=T.concatenate([area1,area2],axis=1)
            h_area=T.concatenate([h_area1,h_area2],axis=1)
            new_e1_index=e1_index+1
            new_e2_index=e2_index+2
            new_s_len=s_len+2+1 #the last 1 is natrual pad from the sentence,but make sure sen_len < maxlen
            
            #concatenate [0,e1_index) (e1_index,e2_index). skip e1
            max1_area=T.concatenate([area[:,:new_e1_index],area[:,new_e1_index+1:new_e2_index],area[:,new_e2_index+1:new_s_len]],axis=1)
            h_max1_area=T.concatenate([h_area[:,:new_e1_index],h_area[:,new_e1_index+1:new_e2_index],h_area[:,new_e2_index+1:new_s_len]],axis=1)
            #max1_area.shape[1]==2 mean no word before e2 other than e1,only 2 pad left if remove e1 from the area, condition can also be new_e2_index==3
            
            max1=max1_area.argmax(axis=1)

            def s(h,m):
                print m.shape
                return h[m]
            res,updates=theano.scan(s,sequences=[h_max1_area,max1])
            
            #concateante (e1_index,e2_index) (e2_index,s_len) skip e2
            #max2_area=T.concatenate([area[:,new_e1_index+1:new_e2_index],area[:,new_e2_index+1:new_s_len]],axis=1)
            #same as max1, if max2_area.shape[1]==2 mean no word after e1 other than e2, only 2 pad left if remove e2 from thea area (one pad is padmin,one is from sentence pad),condition can also e1_index+2=s_len
            #max2=T.switch(T.gt(max2_area.shape[1],2),max2_area.max(axis=1),pad)
           
            #total_max=T.concatenate([max1_area,max2_area],axis=1).max(axis=1)
            '''
'''
            if add_total:
                max_result=T.concatenate([max1,max2,total_max])
            else:
                max_result=T.concatenate([max1,max2])
            '''
'''
            return res
        result,update=theano.scan(
                                step,
                                sequences=[lstm_output.dimshuffle(1,2,0),e_pos,sen_len,attention_value.dimshuffle(1,2,0)],
                                outputs_info=None
                               )
	return result	
'''

class ClassificationByRanking(object):
    def __init__(self,input,n_in,n_out,W=None,b=None,other_model=None,regulization=0,r=2,b1=2.5,b2=0.5):
        self.fan_in=n_in
	self.r=r
        self.b1=b1
        self.b2=b2
        self.other_model=other_model
        self.input=input
        if W is None:
            self.W=theano.shared(
                        value=np.asarray(
                            np.random.uniform(
                                low=-np.sqrt(6./(n_in+n_out)),
                                high=np.sqrt(6./(n_in+n_out)),
                                size=(n_in,n_out)
                                
                                ),
                            dtype=theano.config.floatX
                            ),
                        name='W',
                        borrow=True
                    )
        else:
            self.W=W
        self.p_y_given_x=T.dot(input,self.W)
        self.y_p=T.argmax(self.p_y_given_x,axis=1)
        def step(prob,y):
            return T.switch(T.gt(prob[y],0.),y,T.constant(18))
        self.y_pred,updates=theano.scan(step,sequences=[self.p_y_given_x,self.y_p],outputs_info=None)
        self.params=[self.W]
        self.regulization=regulization
        

    def loss_function(self,y):
        def findc(prob,yy):
            t=prob

            T.set_subtensor(t[T.switch(T.lt(yy,18),yy,0)],T.switch(T.lt(yy,18),0,t[0]))
            return t.argmax()
        c,updates=theano.scan(findc,sequences=[self.p_y_given_x,y])
        def gett(y,w,other_model):
            return T.switch(T.lt(y,18),w[:,T.switch(T.lt(y,18),y,0)],other_model)
        target,updates=theano.scan(gett,sequences=[y],non_sequences=[self.W,self.other_model])
        target2=self.W[:,c].dimshuffle(1,0)
        def step(x,w):
            
            return T.dot(x,w)
        m1,updates=theano.scan(step,sequences=[self.input,target])
        m2,updates=theano.scan(step,sequences=[self.input,target2])
        m=T.log(1+T.exp(self.r*(self.b1-m1)))+T.log(1+T.exp(self.r*(self.b2+m2)))
        if self.regulization > 0:
            res=m.sum()+self.regulization*(self.W**2).sum()
        else:
            res=m.sum()
        print res.type
        return res
    def errors(self,y):
        return T.mean(T.neq(self.y_pred,y))



class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out,W=None,b=None,d=None):
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
        if d is None:
            self.d = theano.shared(
                 value=np.asarray(
                     [0.35],
                  dtype=theano.config.floatX
                 ),
                name='d',
                borrow=True
             )
        else:
            self.d=d

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
        def get_y(prob):
            res=T.switch(T.lt(prob.max(axis=0),self.d),T.constant(18,dtype='int32'),T.argmax(prob,axis=0))
            return res
	# symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred,updates=theano.scan(get_y,sequences=[self.p_y_given_x])
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
        def get_loss(prob,yy):
            return T.switch(T.lt(yy,18),prob[y],1.0-prob.max())
        m,updates=theano.scan(get_loss,sequences=[self.p_y_given_x,y])
        return -T.mean(m)
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

