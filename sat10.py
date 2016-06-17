#! /usr/bin/python
import sys
import os
from model import *
import model
import re
import random
from util import *
import pdb
import numpy as np
import time
import qz_model
#from theano import config
#from theano.ifelse import ifelse
#config.optimizer='None'
#config.exception_verbosity='high'
# Utils ----------------------------------------------------------------------
try:
    import PIL.Image as Image
except ImportError:
    import Image

###########################
#                       description
#      same as exp_428_len95_doublemax_nototal_allinmlp_removecomdep.py
#      just for tuning the hyperparams
########################

class BLSTMLayer(object):
    def __init__(self,rng,input,input_size,layers,activation):
        '''
            lstm layer.
            :type input:theano.tensor (3D,2D)
            :params input: ndim shoud be timesteps, batch_num(if any), wordembedding (if any)

            :type input_size:int
            :param input_size: len of one step input (like embedding if any)

            :type layers: list of int
            :params layers: hidden size of lstm layers
        '''
        self.forward_input=input
        self.backward_input=input[::-1]
        self.fan_in=input_size
        self.batch_size=input.shape[1]

        self.forward_model=LSTM(input_size,hidden_size=layers[-1],activation=activation)#,clip_gradients=True)
        self.backward_model=LSTM(input_size,hidden_size=layers[-1],activation=activation)#,clip_gradients=True)
        self.params=self.forward_model.params+self.backward_model.params
        #self.params=self.backward_model.params
        def forward_step(x,prev_hiddens):
            new_state=self.forward_model.activate(x,prev_hiddens)
            return new_state
        def backward_step(x,prev_hiddens):
            new_state=self.backward_model.activate(x,prev_hiddens)
            return new_state

        forward_output_info=[]
        shared_h=self.forward_model.initial_hidden_state
        if input.ndim>2:
            shared_h=T.repeat(T.shape_padleft(shared_h),self.batch_size,axis=0)
        forward_output_info.append(shared_h)
        
        backward_output_info=[]
        shared_h=self.backward_model.initial_hidden_state
        if input.ndim>2:
            shared_h=T.repeat(T.shape_padleft(shared_h),self.batch_size,axis=0)
        backward_output_info.append(shared_h)

        forward_result,updates=theano.scan(
                                forward_step,
                                sequences=[self.forward_input],
                                outputs_info=forward_output_info,
                                )

        backward_result,updates=theano.scan(
                                backward_step,
                                sequences=[self.backward_input],
                                outputs_info=backward_output_info,
                                )


        if isinstance(forward_result,list):#more than one layer
            #result[-1] indicate the top layer,result[-1] is list of hiddenstate,result[-1][-1] indicate last step of top layer
            self.forward_output=forward_result[-1]
            self.backward_output=backward_result[-1][::-1]
           # self.lstm_output=T.concatenate([forward_result[-1],backward_result[-1]],axis=2)
        else:
            self.forward_output=forward_result
            self.backward_output=backward_result[::-1]
            #self.lstm_output=T.concatenate([forward_result,backward_result],axis=2)
        #Note: self.all_output.shape=(timestep,batch_num,hidden_size*2*2)
        #self.lstm_output=T.concatenate([self.forward_output.max(axis=0),self.backward_output.max(axis=0)],axis=1)
        self.all_output=T.concatenate([self.forward_output,self.backward_output],axis=2)
        #self.output=T.concatenate([self.all_output.max(axis=0),self.all_output.min(axis=0)],axis=1)
        self.output=self.all_output.max(axis=0)
        self.output_size=layers[-1]*2*2
     #   self.lstm_output=T.concatenate([self.forward_output,self.backward_output],axis=2).dimshuffle(1,2,0).max(axis=2)
    #    step=60


def double_max_with_entity(input,e_pos,sen_len,input_size):
    #make sure input.shape=(batch,dim,step)
    #e.shape=(batch,[e1,e2])
    #sen_len.shape=(batch,)
    def step(h,e,s_len):
        e1_index=e[0]
        e2_index=e[1]
        max1=h[:,:e2_index].max(axis=1)
        max2=h[:,e1_index+1:s_len].max(axis=1)
        double_max=T.concatenate([max1,max2])
        
        return double_max
    result,update=theano.scan(
                            step,
                            sequences=[input,e_pos,sen_len],
                            outputs_info=None
                           )
    return (result,input_size*2) 
    pass
    
def double_max_without_entity(input,e_pos,sen_len,input_size,add_total=True):
    #make sure input.shape=(batch,dim,step)
    #e.shape=(batch,[e1,e2])
    #sen_len.shape=(batch,)
    def step(h,e,s_len):
        e1_index=e[0]
        e2_index=e[1]
        # 3 segment, [0-e1);(e1-e2);(e2,end_of_sen)
        pad=T.zeros_like(h[:,0])
        pad_min=h.min(axis=1).dimshuffle(0,'x') 
        area1=T.concatenate([pad_min,h[:,:e2_index],pad_min],axis=1)
        #area2=h[:,e2_index:s_len+1]
        area2=T.concatenate([h[:,e2_index:s_len],pad_min],axis=1) 
        area=T.concatenate([area1,area2],axis=1)
        new_e1_index=e1_index+1
        new_e2_index=e2_index+2
        new_s_len=s_len+2+1 #the last 1 is natrual pad from the sentence,but make sure sen_len < maxlen
        
        #concatenate [0,e1_index) (e1_index,e2_index). skip e1
        max1_area=T.concatenate([area[:,:new_e1_index],area[:,new_e1_index+1:new_e2_index]],axis=1)
        #max1_area.shape[1]==2 mean no word before e2 other than e1,only 2 pad left if remove e1 from the area, condition can also be new_e2_index==3
        max1=T.switch(T.gt(max1_area.shape[1],2),max1_area.max(axis=1),pad)
        
        #concateante (e1_index,e2_index) (e2_index,s_len) skip e2
        max2_area=T.concatenate([area[:,new_e1_index+1:new_e2_index],area[:,new_e2_index+1:new_s_len]],axis=1)
        #same as max1, if max2_area.shape[1]==2 mean no word after e1 other than e2, only 2 pad left if remove e2 from thea area (one pad is padmin,one is from sentence pad),condition can also e1_index+2=s_len
        max2=T.switch(T.gt(max2_area.shape[1],2),max2_area.max(axis=1),pad)
       
        total_max=T.concatenate([max1_area,max2_area],axis=1).max(axis=1)
        if add_total:
            max_result=T.concatenate([max1,max2,total_max])
        else:
            max_result=T.concatenate([max1,max2])
        return max_result
    result,update=theano.scan(
                            step,
                            sequences=[input,e_pos,sen_len],
                            outputs_info=None
                           )
    if add_total:
       return (result,input_size*3) 
    else:
       return (result,input_size*2) 
        
def tri_max_without_entity(input,e_pos,sen_len,input_size):
    #make sure input.shape=(batch,dim,step)
    #e.shape=(batch,[e1,e2])
    #sen_len.shape=(batch,)
    def step(h,e,s_len):
        e1_index=e[0]
        e2_index=e[1]
        # 3 segment, [0-e1);(e1-e2);(e2,end_of_sen)
        pad=T.zeros_like(h[:,0])
        pad_min=h.min(axis=1).dimshuffle(0,'x') 
        area1=T.concatenate([pad_min,h[:,:e2_index],pad_min],axis=1)
        #area2=h[:,e2_index:s_len+1]
        area2=T.concatenate([h[:,e2_index:s_len],pad_min],axis=1) 
        area=T.concatenate([area1,area2],axis=1)
        new_e1_index=e1_index+1
        new_e2_index=e2_index+2
        new_s_len=s_len+2+1
       
        max1=T.switch(T.lt(0,e1_index),area[:,:new_e1_index].max(axis=1),pad)
        max2=T.switch(T.lt(e1_index,e2_index-1),area[:,new_e1_index+1:new_e2_index].max(axis=1),pad)
        max3=T.switch(T.lt(e2_index,s_len-2),area[:,new_e2_index+1:new_s_len].max(axis=1),pad)
        tri_max=T.concatenate([max1,max2,max3])
        return tri_max
    result,update=theano.scan(
                            step,
                            sequences=[input,e_pos,sen_len],
                            outputs_info=None
                           )
    return (result,input_size*3) 

def entity_fea(input,e_pos,input_size):
    #make sure input.shape=(batch,dim,step)
    #e.shape=(batch,[e1,e2])
    def step(h,e):
        e1=e[0]
        e2=e[1]
        return T.concatenate([h[:,e1],h[:,e2]])
    result,update=theano.scan(
                            step,
                            sequences=[input,e_pos],
                            outputs_info=None
                            )
    return (result,input_size*2)


           # minibatch_avg_cost=coling.train(sen_x,pos_label_x,ner_x,wnsyn_x,dep_x,dep_path_x,pos_x,y,e_pos,sen_len,np.cast['int32'](1))
class RMM_BLSTM(object):
    '''
    Relative mul-max based on BLSTM for semVal2010 task8
    '''
    def __init__(self,rng,word_num,emb_dim,sen_len,cs,pos_max,pos_emb_len,lex_len,ex_fea_len,conv_output_col,n_hidden,label_num,sen_dropout,lex_dropout,layers,max_word_fea_size,word_fea_emb):
        '''
        RMM_BLSTM
        '''
        ######################
        #  var def
        self.pos_idx=T.itensor3('pos_idx')
        self.sen_idx=T.itensor3('sen_idx')

        self.pos_label_idx=T.imatrix('pos_label')
        self.ner_idx=T.imatrix('ner_idx')
        self.wnsyn_idx=T.imatrix('wnsyn_idx')
        self.dep_idx=T.imatrix('dep_idx')
        self.dep_path_idx=T.itensor3('dep_path_idx')
        self.com_dep_path_idx=T.imatrix('com_dep_path_idx')

        self.e_pos=T.imatrix()
        self.y=T.ivector('y')
        self.sen_len=T.ivector('len')
        self.is_train=T.iscalar('is_train')
        ###################################
        #      pos embedding init
        #pos can be (-pos_max,pos_max)
        self.e1_pos_emb=Embeddings(rng,pos_max*2,pos_emb_len,'e1_pos_emb')
        self.e2_pos_emb=Embeddings(rng,pos_max*2,pos_emb_len,'e2_pos_emb')
        #state.word_fea_emb={'pos':5,'pos_label':20,'ner':20,'wnsyn':20,'dep':20,'dep_path:':5}
        self.pos_label_emb=Embeddings(rng,max_word_fea_size,word_fea_emb['pos_label'],'pos_label_emb')
        self.ner_emb=Embeddings(rng,max_word_fea_size,word_fea_emb['ner'],'ner_emb')
        self.wnsyn_emb=Embeddings(rng,max_word_fea_size,word_fea_emb['wnsyn'],'wnsyn_emb')
        self.dep_emb=Embeddings(rng,max_word_fea_size,word_fea_emb['dep'],'dep_emb')
        self.dep_path_emb=Embeddings(rng,max_word_fea_size,word_fea_emb['dep_path'],'dep_path_emb')
        self.com_dep_path_emb=Embeddings(rng,max_word_fea_size,word_fea_emb['com_dep_path'],'com_dep_path_emb')
        
        #############################
        #     entity embedding
        #self.emb=Embeddings(rng,word_num+1,emb_dim,'emb')
        self.emb=Embeddings(rng,word_num+1+3,emb_dim,'emb')# 1 for pad(0),2 for e1 and e2,(-1,-2)
        self.pos_sen_x=self.get_embedding_sen(
                                sen_idx=self.sen_idx,
                                word_emb=self.emb,
                                pos_idx=self.pos_idx,
                                pos_max=pos_max,
                                cs=cs,
                                e1_pos_emb=self.e1_pos_emb,
                                e2_pos_emb=self.e2_pos_emb,
                                pos_label_idx=self.pos_label_idx,
                                pos_label_emb=self.pos_label_emb,
                                ner_idx=self.ner_idx,
                                ner_emb=self.ner_emb,
                                wnsyn_idx=self.wnsyn_idx,
                                wnsyn_emb=self.wnsyn_emb,
                                dep_idx=self.dep_idx,
                                dep_emb=self.dep_emb,
                                dep_path_idx=self.dep_path_idx,
                                dep_path_emb=self.dep_path_emb,
                                )
        #self.pos_sen_x.shape=(batch,input_size,len_of_sen) 
        #########################
        #  model building
        #state.word_fea_emb={'pos':5,'pos_label':20,'ner':20,'wnsyn':20,'dep':20,'dep_path:':5}
        self.input_size=self.emb.D*cs+pos_emb_len*2+word_fea_emb['ner']+word_fea_emb['pos_label']+word_fea_emb['wnsyn']+word_fea_emb['dep']+word_fea_emb['dep_path']*3
         
        ##############
        #remove slip window
        self.pos_sen_x=T.concatenate([self.pos_sen_x[:,self.emb.D:self.emb.D*2,:],self.pos_sen_x[:,self.emb.D*3:,:]],axis=1)
        self.input_size=self.input_size-self.emb.D*2
        ##############
        print 'input_size for word:%d'%self.input_size
        self.lex_x,self.lex_x_size=entity_fea(self.pos_sen_x,self.e_pos,self.input_size)
        ####################
        # build lstm layers
        #self.pos_sen_x.shape=(batch_num,input_size,len_of_sentence),change it to (len_of_sentence,batch_num,input_size)
        #only dep_path_x for sen
        #sen_word=self.pos_sen_x[:,:self.emb.D]
        #dep_path=self.dep_path_emb.E[self.dep_path_idx].dimshuffle(0,2,1) 
        #self.sen_input=T.concatenate([self.pos_sen_x[:,:self.emb.D],self.pos_sen_x[:,-word_fea_emb['dep_path']:]],axis=1)
        #self.sen_input=T.concatenate([sen_word,dep_path],axis=1)
        #self.sen_input_size=self.emb.D+word_fea_emb['dep_path']

        self.lstm_input=self.pos_sen_x.dimshuffle(2,0,1)
        self.lstm_input=T.switch(T.neq(self.is_train,0),model._dropout_from_layer(rng,self.lstm_input,p=sen_dropout),self.lstm_input)

        lstm_input_size=self.input_size
        self.lstm_layer=BLSTMLayer(
                                rng=rng,
                               input=self.lstm_input,
                               input_size=lstm_input_size,
                                layers=layers,
                                activation=model.rectify
                                )
        #lstm.all_output.shape=(step,batch,output_size)
       # self.debug_check_lstm_output=theano.function([self.sen_idx,self.pos_idx],[self.lstm_layer.all_output])
        (lstm_lex,lstm_lex_size)=entity_fea(self.lstm_layer.all_output.dimshuffle(1,2,0),self.e_pos,self.lstm_layer.output_size)
        self.lstm_output_list=[lstm_lex[:,:lstm_lex_size/2],lstm_lex[:,lstm_lex_size/2:]]
        self.attention_value_list=[]
        self.pos_value_list=[]
        
        self.attention_layer=qz_model.attention_chain(rng=rng,n=self.lstm_layer.output_size+lstm_lex_size,n_out=1,act=T.tanh)
        for attetion_count in range(10):
            attention_fea=T.concatenate(self.lstm_output_list[-2:],axis=1).dimshuffle('x',0,1)
            self.attention_value_list.append(self.attention_layer.get_output(input=T.concatenate([self.lstm_layer.all_output,T.concatenate([attention_fea for i in range(95)],axis=0)],axis=2)))
            self.pos_value_list.append(self.attention_value_list[-1].argmax(axis=0))
            self.lstm_output_list.append(qz_model.attention_pooling(self.lstm_layer.all_output.dimshuffle(1,0,2),self.pos_value_list[-1])[:,0,:])
        
        
        lstm_output=T.concatenate(self.lstm_output_list,axis=1)
        
        
        
        lstm_output_size=self.lstm_layer.output_size*len(self.lstm_output_list)
        dropout_lstm_output=model._dropout_from_layer(rng,lstm_output,p=sen_dropout)
        
        self.debug_f1=theano.function(
                    inputs=[self.sen_idx,self.pos_label_idx,self.ner_idx,self.wnsyn_idx,self.dep_idx,self.dep_path_idx,self.com_dep_path_idx,self.pos_idx,self.e_pos,self.sen_len,self.is_train],
                    outputs=[
                            self.lstm_layer.all_output.dimshuffle(1,2,0),
                            self.e_pos,
                            self.sen_len,
                            lstm_output
                            ],
                    on_unused_input='ignore'
                    )
        lex_fea_input=self.lex_x
        dropout_lex_fea_input=model._dropout_from_layer(rng,lex_fea_input,p=sen_dropout)
        lex_fea_input_size=self.lex_x_size
        self.dropout_hiddenLayer=DropoutHiddenLayer(
                                    rng=rng,
                                    input=T.concatenate([dropout_lstm_output],axis=1),
                                    n_in=lstm_output_size,
                                    #input=dropout_lstm_output,
                                    #n_in=lstm_output_size,
                                    n_out=n_hidden,
                                    dropout_rate=sen_dropout,
                                    activation=model.rectify
                                    )

        self.hiddenLayer=HiddenLayer(
                            rng=rng,
                            input=T.concatenate([lstm_output],axis=1),
                            n_in=lstm_output_size,
                            #input=lstm_output,
                            #n_in=lstm_output_size,
                            n_out=n_hidden,
                            W=self.dropout_hiddenLayer.W*(1-sen_dropout),
                            b=self.dropout_hiddenLayer.b,
                            activation=model.rectify
                            )
        ###############################
       # build other layers
       # logic_W_values=np.asarray(
       #             rng.uniform(
                        #low=-np.sqrt(6./(n_hidden+lex_fea_input_size+label_num)),
                        #high=np.sqrt(6./(n_hidden+lex_fea_input_size+label_num)),
                        #size=(n_hidden+lex_fea_input_size,label_num)
       #                 low=-np.sqrt(6./(n_hidden+label_num)),
       #                 high=np.sqrt(6./(n_hidden+label_num)),
       #                 size=(n_hidden,label_num)
       #                 ),
       #             dtype=theano.config.floatX
       #     )
       # logic_W_values*=4
       # logic_W=theano.shared(value=logic_W_values,name='W',borrow=True)
        
        self.dropout_output_layer=LogisticRegression(
                #input=T.concatenate([self.dropout_hiddenLayer.output,dropout_lex_fea_input],axis=1),
                #n_in=n_hidden+lex_fea_input_size,
                input=self.dropout_hiddenLayer.output,
                n_in=n_hidden,
               # W=logic_W,
                n_out=label_num
            )

        self.output_layer=LogisticRegression(
                #input=T.concatenate([self.hiddenLayer.output,lex_fea_input],axis=1),
                #n_in=n_hidden+lex_fea_input_size,
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                W=self.dropout_output_layer.W,#*(1-sen_dropout),
                b=self.dropout_output_layer.b,
                n_out=label_num
               )
        
        layer_params=self.lstm_layer.params+self.dropout_hiddenLayer.params+self.dropout_output_layer.params
        emb_params=[self.e1_pos_emb.E,self.e2_pos_emb.E,self.pos_label_emb.E,self.ner_emb.E,self.wnsyn_emb.E,self.dep_emb.E,self.dep_path_emb.E,self.emb.E]
        self.params=emb_params+layer_params
        ####################
        #    debug function
       # self.debug_f=theano.function([self.sen_idx,self.pos_idx],[lstm_output],on_unused_input='ignore')
        ########################
        #     theano functions
        self.errors=self.output_layer.errors
        #####################
        #  train ,valid ,test,classify function
       

        self.classify=theano.function(
                    inputs=[self.sen_idx,self.pos_label_idx,self.ner_idx,self.wnsyn_idx,self.dep_idx,self.dep_path_idx,self.com_dep_path_idx,self.pos_idx,self.e_pos,self.sen_len,self.is_train],
                    outputs=self.output_layer.y_pred,
                    on_unused_input='ignore'
                    )
        self.validate=theano.function(
                    inputs=[self.sen_idx,self.pos_label_idx,self.ner_idx,self.wnsyn_idx,self.dep_idx,self.dep_path_idx,self.com_dep_path_idx,self.pos_idx,self.y,self.e_pos,self.sen_len,self.is_train],
                    outputs=[self.errors(self.y),self.output_layer.y_pred,self.output_layer.p_y_given_x],
                    on_unused_input='ignore'
                    )
        self.test=self.validate
        #get train function
        self.train=self.get_train_fn()
    
    #------------------------
    def get_train_fn(self):
        '''
        get trian function
        '''
        ##################
        # get cost
        L2=(self.dropout_output_layer.W**2).sum()
        self.cost=self.dropout_output_layer.negative_log_likelihood(self.y)
        ###########
        #  update
        updates,self.lr_dict=sgd_updates_adadelta(self.params,self.cost)
        #updates, gsums, xsums, lr, max_norm = create_optimization_updates(self.cost, params, method='adadelta')
        #pdb.set_trace()
        train_fn=theano.function(
                inputs=[self.sen_idx,self.pos_label_idx,self.ner_idx,self.wnsyn_idx,self.dep_idx,self.dep_path_idx,self.com_dep_path_idx,self.pos_idx,self.y,self.e_pos,self.sen_len,self.is_train],
                outputs=[self.errors(self.y),self.output_layer.y_pred,self.output_layer.p_y_given_x],
                updates=updates,
                on_unused_input='ignore')
        return train_fn


    def get_embedding_sen(self,sen_idx,word_emb,pos_idx,pos_max,cs,e1_pos_emb,e2_pos_emb,pos_label_idx,pos_label_emb,ner_idx,ner_emb,wnsyn_idx,wnsyn_emb,dep_idx,dep_emb,dep_path_idx,dep_path_emb):
        #################
        # pos emb
        e1_pos_idx=pos_idx[:,:,0]+pos_max
        e2_pos_idx=pos_idx[:,:,1]+pos_max
        e1_pos_x=e1_pos_emb.E[e1_pos_idx]
        e2_pos_x=e2_pos_emb.E[e2_pos_idx]
        #e1_pos_x=e1_pos_idx.astype(theano.config.floatX)
        #e2_pos_x=e2_pos_idx.astype(theano.config.floatX)
        #################
        # sen emb
        sen_x=word_emb.E[sen_idx].reshape((sen_idx.shape[0],sen_idx.shape[1],word_emb.D*cs))
        pos_label=pos_label_emb.E[pos_label_idx]
        ner=ner_emb.E[ner_idx]
        wnsyn=wnsyn_emb.E[wnsyn_idx]
        dep_x=dep_emb.E[dep_idx]
        dep_path_x=dep_path_emb.E[dep_path_idx].reshape((dep_path_idx.shape[0],dep_path_idx.shape[1],dep_path_emb.D*3))

        word_fea=T.concatenate([pos_label,ner,wnsyn,dep_x,dep_path_x],axis=2)
        pos_sen_x=T.concatenate((sen_x,word_fea,e1_pos_x,e2_pos_x),axis=2).dimshuffle(0,2,1)#dimshuffle(1,0,2)#.dimshuffle(0,2,1)
        return pos_sen_x

    def init_emb(self,idx2emb,voca,emb):
        #self.emb.load_init(idx2emb)
        assert emb.N>voca
        emb_word=0
        W_value=emb.E.get_value()
        for i in range(voca):
            if idx2emb.has_key(i):
                W_value[i]=idx2emb[i]
                emb_word+=1
            else:# use unkown emb 
                W_value[i]=idx2emb[0]
        emb.E.set_value(W_value)
        #W=np.array(W_value)
        #self.lex_emb.E.set_value(W)
        print 'all word:%d (including one pad word),embedding word:%d'%(voca,emb_word)
        return emb

    def normalize(self):
        pass
        self.emb.std_normalize()
        self.e1_pos_emb.std_normalize()
        self.e2_pos_emb.std_normalize()
    
    def save_model(self,m_file):
        models=[]
        for param in self.params:
            models.append(param.get_value())
        save_data(models,m_file)
    
    def load_model(self,m_file):
        models=load_data(m_file)
        for value,param in zip(models,self.params):
            param.set_value(value)
        
def get_e_mask(pos_x,row,col):
    e_pos=0-pos_x[:,0]
    masks=[]
    for e in e_pos:
        mask=np.ones((col,row))
        mask[e]=0
        masks.append(mask.T)
    masks=np.array(masks)
    return (e_pos.astype('int32'),masks.astype('float32'))

def get_e_lex(e_pos,pos_x,ner_x,wnsyn_x):
    '''
    get pos ner and wnsyn of e1,e2
    '''
    #pdb.set_trace()
    my_lex_fea=[]
    for i in range(len(e_pos)):
        e1=e_pos[i][0]
        e2=e_pos[i][1]
        lex_fea=[]
        lex_fea.append(pos_x[i][e1])
        lex_fea.append(pos_x[i][e2])
        lex_fea.append(ner_x[i][e1])
        lex_fea.append(ner_x[i][e2])
        lex_fea.append(wnsyn_x[i][e1])
        lex_fea.append(wnsyn_x[i][e2])
        my_lex_fea.append(lex_fea)
    return np.array(my_lex_fea)

def epoch_evaluate(y_pred,t_y,idx2rel):
    total_TP=0
    total_FP=0
    total_FN=0
    for i in idx2rel.keys():
        rel=idx2rel[i]
        TP=((y_pred==i)*(t_y==i)).sum()
        FP=((y_pred==i)*(t_y!=i)).sum()
        FN=((y_pred!=i)*(t_y==i)).sum()
        total_TP+=TP
        total_FP+=FP
        total_FN+=FN
        P=100.0*TP/(TP+FP+0.0000001)
        R=100.0*TP/(TP+FN+0.0000001)
        F1=2*P*R/(P+R+0.00000001)
        print 'rel %d (%s)\t:P:%.3f(%d/%d)\tR:%.3f(%d/%d)\tF1:%.3f'%(i,rel,P,TP,(TP+FP),R,TP,(TP+FN),F1)
    total_P=100.0*total_TP/(total_TP+total_FP+0.0000001)
    total_R=100.0*total_TP/(total_TP+total_FN+0.0000001)
    total_F1=2*total_P*total_R/(total_P+total_R+0.000001)
    
    print 'Total\t:P:%.3f\tR:%.3f\tF1:%.3f'%(total_P,total_R,total_F1)
    print '='*30



def SemVal_evaluate(test_id,test_y,y_pred,idx2rel,script,temp_dir):
    pro_ans=[]
    true_ans=[]
    error=0
    for i in range(len(test_id)):
        pro_rel=idx2rel[y_pred[i]]
        true_rel=idx2rel[test_y[i]]
        if pro_rel!=true_rel:
            error+=1
        pro_ans.append('%s\t%s'%(test_id[i],pro_rel))
        true_ans.append('%s\t%s'%(test_id[i],true_rel))
    #print 'evulate error: %d/%d ==%f %%'%(error,len(test_id),error*1.0/len(test_id)*100)
    prop_key=os.path.join(temp_dir,'proposed_key')
    ans_key=os.path.join(temp_dir,'ans_key')
    with open(prop_key,'w') as f:
        f.write('\n'.join(pro_ans)+'\n')
    with open(ans_key,'w') as f:
        f.write('\n'.join(true_ans)+'\n')
    cmd='perl %s %s %s'%(script,prop_key,ans_key)
    result=os.popen(cmd).readlines() 
    desc='.'.join(result[-15:])
    F1=re.compile(r'= *([0-9.]+).*%').search(result[-1]).group(1)
    return (desc,float(F1))

# Experiment function --------------------------------------------------------
def SEexp(state, channel):
    # Show experiment parameters
    print >> sys.stderr, state
    np.random.seed(state.seed)
    tic=time.time()
    ########################
    #   load data
    print 'loading data....'
    #pdb.set_trace()
    dicts=load_data(os.path.join(state.datapath,state.dicts))
    elem2idx,idx2elem=dicts['word_dict']
    rel2idx,idx2rel=dicts['rel_dict']
    
    data_set=load_data(os.path.join(state.datapath,state.dataset))
    idx2emb=load_data(os.path.join(state.datapath,state.idx2emb))
    print 'load data cost %d sec'%(time.time()-tic)
    max_word_fea_size=data_set['max_word_fea_size']

    train_sen_x=data_set['train']['sen_x']
    train_pos_label_x=data_set['train']['pos_label_x']
    train_ner_x=data_set['train']['ner_x']
    train_wnsyn_x=data_set['train']['wnsyn_x']
    train_pos_x=data_set['train']['pos_x']
    train_lex_x=data_set['train']['lex_x']
    train_sen_len=data_set['train']['sen_len']
    train_ex_fea_x=data_set['train']['ex_fea_x']
    train_id=data_set['train']['id']
    train_y=data_set['train']['y']
    #dependency fea
    train_dep_x=data_set['train']['dep_x']
    train_dep_path_x=data_set['train']['dep_path_x']
    train_com_dep_path_x=data_set['train']['com_dep_path_x']
    train_dep_e_parent=data_set['train']['dep_e_parent']
    #train_my_lex=get_e_lex(0-train_pos_x[:,0],train_pos_label_x,train_ner_x,train_wnsyn_x)

    test_sen_x=data_set['test']['sen_x']
    test_pos_label_x=data_set['test']['pos_label_x']
    test_ner_x=data_set['test']['ner_x']
    test_wnsyn_x=data_set['test']['wnsyn_x']
    test_pos_x=data_set['test']['pos_x']
    test_lex_x=data_set['test']['lex_x']
    test_sen_len=data_set['test']['sen_len']
    test_ex_fea_x=data_set['test']['ex_fea_x']
    test_id=data_set['test']['id']
    test_y=data_set['test']['y']
    #dependency fea
    test_dep_x=data_set['test']['dep_x']
    test_dep_path_x=data_set['test']['dep_path_x']
    test_com_dep_path_x=data_set['test']['com_dep_path_x']
    test_dep_e_parent=data_set['test']['dep_e_parent']
    #test_my_lex=get_e_lex(0-test_pos_x[:,0],test_pos_label_x,test_ner_x,test_wnsyn_x)
    print 'train num:%d test:%d'%(len(train_sen_x),len(test_sen_x))
    #pdb.set_trace()
    #####################
    #   get model
    #pdb.set_trace()
    voca=len(elem2idx)
    rng=np.random.RandomState(1234)
    begin_build=time.time()
    #pdb.set_trace()
    if True:
        coling=RMM_BLSTM(
            rng=rng,
            word_num=voca,
            emb_dim=100,
            sen_len=95,
            cs=3,#context wind size=3
            pos_max=95,#relative pos max len
            pos_emb_len=5,
            lex_len=6, #len of lex_feaure, 6 word:
            ex_fea_len=178,# two one-hot representation for e1,e2 wnsyn
            conv_output_col=200,
            n_hidden=1000,
            label_num=19,
            sen_dropout=state.sen_dropout,
            lex_dropout=state.lex_dropout,
            layers=state.layers,
            max_word_fea_size=max_word_fea_size,
            word_fea_emb=state.word_fea_emb
           )
        coling.emb=coling.init_emb(idx2emb,voca,coling.emb)
        model_file=__file__+"_model"
        if os.path.isfile(model_file) and state.loadmodel:
            coling.load_model(model_file)
            print 'sucessfully load model'
    else:
        coling=DD()
    #coling.conv_emb=coling.init_emb(idx2emb,voca,coling.conv_emb)
    ###########################
    # training
    print 'build cost %d sec'%(time.time()-begin_build)
    batchsize =state.batchsize
    nbatches=int(np.ceil(len(train_sen_x)*1.0/batchsize))
    best_test_loss=np.inf
    print >> sys.stderr, "BEGIN TRAINING"
    n_epoches=state.n_epoches
    train_loss_list=[]
    test_loss_list=[]
    test_f1_list=[]

    script='./result/script/semeval2010_task8_scorer-v1.2.pl'
    temp_dir='.'

    for epoch_count in xrange(1, n_epoches + 1):
        epoch_start=time.time()
        # Shuffling
        #pdb.set_trace()
        order = np.random.permutation(len(train_sen_x))
        train_sen_x=train_sen_x[order]
        train_pos_label_x= train_pos_label_x[order]
        train_ner_x=train_ner_x[order]
        train_wnsyn_x=train_wnsyn_x[order]
        train_dep_x=train_dep_x[order]
        train_dep_path_x=train_dep_path_x[order]
        train_com_dep_path_x=train_com_dep_path_x[order] 
        train_sen_len=train_sen_len[order]
        train_pos_x=train_pos_x[order]
        train_y=train_y[order]
       
        if epoch_count%30==0:
            print '-'*5+'train loss:'+'-'*10
            print train_loss_list
            print '-'*5+'test loss:'+'-'*10
            print test_loss_list
            print '-'*5+'test f1:'+'-'*10
            print test_f1_list
            save_data((train_loss_list,test_loss_list,test_f1_list),state.result_list_file)
        
        if epoch_count%10==0:
            print >> sys.stderr, state
      #  pdb.set_trace()
        train_start=time.time()
        train_loss=0
        for i in range(nbatches):
            sen_x=train_sen_x[i*batchsize:(i+1)*batchsize]
            pos_x=train_pos_x[i*batchsize:(i+1)*batchsize]
            pos_label_x=train_pos_label_x[i*batchsize:(i+1)*batchsize]  
            ner_x=train_ner_x[i*batchsize:(i+1)*batchsize]  
            wnsyn_x=train_wnsyn_x[i*batchsize:(i+1)*batchsize]  
            dep_x=train_dep_x[i*batchsize:(i+1)*batchsize]
            dep_path_x=train_dep_path_x[i*batchsize:(i+1)*batchsize] 
            com_dep_path_x=train_com_dep_path_x[i*batchsize:(i+1)*batchsize]
            y=train_y[i*batchsize:(i+1)*batchsize]
            sen_len=train_sen_len[i*batchsize:(i+1)*batchsize]
            e_pos=0-pos_x[:,0]
            #pdb.set_trace()
            #all_out,ep,sl,result=coling.debug_f1(sen_x,pos_label_x,ner_x,wnsyn_x,dep_x,dep_path_x,com_dep_path_x,pos_x,e_pos,sen_len,np.cast['int32'](1))
            minibatch_avg_cost,train_y_pred,train_y_x=coling.train(sen_x,pos_label_x,ner_x,wnsyn_x,dep_x,dep_path_x,com_dep_path_x,pos_x,y,e_pos,sen_len,np.cast['int32'](1))
            train_loss+=minibatch_avg_cost
            if state.normalize:
                coling.normalize()
        #pdb.set_trace()
        train_loss/=nbatches
        print 'epoch %d train cost %f min,train loss:%f %%'%(epoch_count,(time.time()-train_start)/60.0,train_loss*100)
        train_loss_list.append((epoch_count,round(train_loss,4)))
        if (epoch_count % state.test_freq)==0:
            test_start = time.time()
            #pdb.set_trace()
            #e_pos=0-test_pos_x[:,0]
            #test_loss,y_pred=coling.test(test_sen_x,test_pos_label_x,test_ner_x,test_wnsyn_x,test_dep_x,test_dep_path_x,test_pos_x,test_y,e_pos,test_sen_len,np.cast['int32'](0))
            ###################
            #batch test
            size=1000
            batchs=int(np.ceil(len(test_sen_x)*1.0/size))
            y_pred=None    
            test_loss=0
            y_x=None
            for i in range(batchs):
                sen_x=test_sen_x[i*size:(i+1)*size]
                pos_label_x=test_pos_label_x[i*size:(i+1)*size]
                ner_x=test_ner_x[i*size:(i+1)*size]
                wnsyn_x=test_wnsyn_x[i*size:(i+1)*size]
                dep_x=test_dep_x[i*size:(i+1)*size]
                dep_path_x=test_dep_path_x[i*size:(i+1)*size]
                com_dep_path_x=test_com_dep_path_x[i*size:(i+1)*size]
                pos_x=test_pos_x[i*size:(i+1)*size]
                y=test_y[i*size:(i+1)*size]
                sen_len=test_sen_len[i*size:(i+1)*size]
                e_pos=0-pos_x[:,0] 
                avg_test_loss,avg_y_pred,avg_y_x=coling.test(sen_x,pos_label_x,ner_x,wnsyn_x,dep_x,dep_path_x,com_dep_path_x,pos_x,y,e_pos,sen_len,np.cast['int32'](0))
                test_loss+=avg_test_loss
                
                if y_pred is None:
                    y_pred=avg_y_pred
                else:
                    y_pred=np.concatenate([y_pred,avg_y_pred])
                if y_x is None:
                    y_x=avg_y_x
                else:
                    y_x=np.concatenate([y_x,avg_y_x],axis=0)
            test_loss/=batchs
            epoch_evaluate(y_pred,test_y,idx2rel)
            
            #pdb.set_trace()
            desc,f1=SemVal_evaluate(test_id,test_y,y_pred,idx2rel,script,temp_dir)
            print '\nepoch %d\ttest loss:%f\ttrain_loss:%f\t\nF1:%s'%(epoch_count,test_loss*100,train_loss*100,desc)
            
            test_loss_list.append((epoch_count,round(test_loss,4)))
            test_f1_list.append((epoch_count,f1))
            if test_loss<best_test_loss:
                best_test_loss=test_loss
                analyse_pack=(test_y,y_pred,y_x,idx2rel,test_id)
                save_data(analyse_pack,state.analyse_file)
                #test_results.append(test_loss)
                state.best_test_epoch=epoch_count
                #evaluate(coling,test_sen_x,test_pos_label_x,test_ner_x,test_wnsyn_x,test_pos_x,test_lex_x,test_my_lex,test_ex_fea_x,test_y,e_pos,test_sen_len,test_id,idx2rel,state)
                print >> sys.stderr, "\t\t##### NEW BEST test >>test error: %f %%,take %s sec" % (test_loss*100,time.time()-tic)
            print >> sys.stderr, "\t(the test took %s seconds)" % (round(time.time() - test_start, 3))
        print 'save model to %s'%__file__+'_model'
        coling.save_model(__file__+'_model')
        print >> sys.stderr, "\t(epoch %d took %s min)" % (epoch_count,round((time.time()-epoch_start)/60.0, 3))
    print '-'*5+'train loss:'+'-'*10
    print train_loss_list
    print '-'*5+'test loss:'+'-'*10
    print test_loss_list
    print '-'*5+'test f1:'+'-'*10
    print test_f1_list
    save_data((train_loss_list,test_loss_list,test_f1_list),state.result_list_file)
    return channel.COMPLETE



def launch(datapath='./data/', dataset='coling_v3_input', dicts='dicts.pkl',idx2emb='idx2emb.pkl',rel_dict='rel_dict.pkl',Nrel=19,batchsize=400,n_epoches=1000,sen_dropout=0.4,lex_dropout=0.1,valid_freq=1,seed=123,savepath='./result',loadmodel=False,normalize=False,layers=[10],suffix='',word_fea_emb=None):

    # Argument of the experiment script
    state = DD()

    state.datapath = datapath
    state.dataset = dataset
    state.dicts =dicts
    state.idx2emb=idx2emb
    state.rel_dict=rel_dict
    state.Nrel = Nrel
    state.batchsize=batchsize
    state.n_epoches=n_epoches
    #pdb.set_trace()
    state.valid_freq=valid_freq
    state.seed=seed
    state.savepath = savepath
    state.loadmodel=loadmodel
    state.proposed_key='proposed_key_'+str(state.n_epoches)+'_'+str(batchsize)+"_"+dataset+"-"+suffix
    state.ans_key='ans_key_'+str(state.n_epoches)+'_'+str(batchsize)+'_'+dataset+'-'+suffix
    state.analyse_file='ForAnalyse_'+__file__+'_'+suffix
    state.result_list_file=__file__+'_result_list_'+suffix
    state.valid_freq=10
    state.test_freq=1
    state.sen_dropout=sen_dropout
    state.lex_dropout=lex_dropout
    state.normalize=normalize
    state.layers=layers
    #state.word_fea_emb_len=word_fea_emb_len
    if word_fea_emb is None:
        state.word_fea_emb={'pos':5,'pos_label':20,'ner':20,'wnsyn':20,'dep':20,'dep_path':10,'com_dep_path':20}
    else:
        state.word_fea_emb=word_fea_emb

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)
    channel = Channel(state)
    SEexp(state, channel)

if __name__ == '__main__':
     
    tic=time.time()
    test=False
    if test:
        launch(loadmodel=True,idx2emb='glove_100_idx2emb.pkl',datapath='./data',dataset='blstm_complete_input_len95_mini',n_epoches=3,batchsize=13,
        word_fea_emb={'pos':5,'pos_label':2,'ner':2,'wnsyn':2,'dep':2,'dep_path':2, 'com_dep_path':2})
        #launch(datapath='./data',dataset='single_coling_input_mini',n_epoches=11,batchsize=1)
    else:
        launch(loadmodel=False,idx2emb='glove_100_idx2emb.pkl',datapath='./data',dataset='blstm_complete_input_len95',n_epoches=350,batchsize=200,sen_dropout=0.5,lex_dropout=0.0,layers=[400],suffix='',word_fea_emb={'pos':5,'pos_label':20,'ner':20,'wnsyn':20,'dep':20,'dep_path':10, 'com_dep_path':20})
    print 'total time:%d'%(time.time()-tic)
