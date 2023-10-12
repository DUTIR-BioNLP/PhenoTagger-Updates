# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:08:09 2021

@author: luol2
"""
import tensorflow as tf
from represent_tc import Hugface_RepresentationLayer,NN_RepresentationLayer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad,Nadam
from transformers import TFBertModel, BertConfig,TFElectraModel,TFAutoModel
import numpy as np
import sys



class NN_TC():
    def __init__(self, model_files):
        self.model_type='cnn'
        self.model_test_type='cnn'
        self.fea_dict = {'word': 1,
                         'char': 1,
                         }
    
        self.hyper = {'sen_max'      :256,
                      'word_max'     :30,
                      'charvec_size' :50}
             
        self.w2vfile=model_files['w2vfile']      
        self.charfile=model_files['charfile']
        self.labelfile=model_files['labelfile']
          
        vocab={'char':self.charfile,'label':self.labelfile}
        print('loading w2v model.....') 
        self.rep = NN_RepresentationLayer(self.w2vfile,vocab_file=vocab, frequency=400000)
        
    def build_model(self): 
        print('building  model......')
        all_fea = []
        fea_list = []
        
        if self.fea_dict['word'] == 1:
            word_input = Input(shape=(self.hyper['sen_max'],), dtype='int32', name='word_input')  
            all_fea.append(word_input)
            word_fea = Embedding(self.rep.vec_table.shape[0], self.rep.vec_table.shape[1], weights=[self.rep.vec_table], trainable=False,mask_zero=False, input_length=self.hyper['sen_max'], name='word_emd')(word_input)
            fea_list.append(word_fea)
    
        if self.fea_dict['char'] == 1:
            char_input = Input(shape=(self.hyper['sen_max'],self.hyper['word_max']), dtype='int32', name='char_input')
            all_fea.append(char_input)
            char_fea = TimeDistributed(Embedding(self.rep.char_table_size, self.hyper['charvec_size'], trainable=True,mask_zero=False),  name='char_emd')(char_input)
            char_fea = TimeDistributed(Conv1D(self.hyper['charvec_size']*2, 3, padding='same',activation='relu'), name="char_cnn")(char_fea)
            char_fea_max = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling_max")(char_fea)
            fea_list.append(char_fea_max)
            
        
        if len(fea_list) == 1:
            concate_vec = fea_list[0]
        else:
            concate_vec = Concatenate()(fea_list)
    
        concate_vec = Dropout(0.2)(concate_vec)
    
        # model
        if self.model_test_type=='cnn':    
            cnn = Conv1D(512, 3, padding='valid', activation='relu',name='cnn1')(concate_vec)
            cnn = Conv1D(256, 3, padding="valid", activation="relu",name='cnn2')(cnn)
            cnn = GlobalMaxPooling1D()(cnn)
        elif self.model_test_type=='lstm':
            bilstm = Bidirectional(LSTM(200, return_sequences=True, implementation=2, dropout=0.4, recurrent_dropout=0.4), name='bilstm1')(concate_vec)
            cnn = GlobalMaxPooling1D()(bilstm)

    
        dense = Dense(128, activation='relu')(cnn)
        dense= Dropout(0.2)(dense)
        output = Dense(self.rep.label_table_size, activation='softmax')(dense)
        self.model = Model(inputs=all_fea, outputs=output)

        # opt = Adam(lr=1e-5) 
        # opt = RMSprop(lr=0.001)
        # self.model.compile(
        #     optimizer=opt,
        #     loss='binary_crossentropy',
        #     metrics=['accuracy'],
        # )
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.summary()


    def load_model(self,model_file):
        self.model.load_weights(model_file)
        self.model.summary()        
        print('load model done!')

    

class HUGFACE_TC(): #huggingface transformers
    def __init__(self, model_files):
        self.model_type='HUGFACE'
        self.maxlen = 256 #sent 256 doc-512,pretrain-sent 128
        self.checkpoint_path = model_files['checkpoint_path']
        self.label_file=model_files['labelfile']
        self.lowercase=model_files['lowercase']
        self.rep = Hugface_RepresentationLayer(self.checkpoint_path, self.label_file, lowercase=self.lowercase)
        
            
    def build_model(self):
        print('...vocab len:',self.rep.vocab_len)
        plm_model = TFAutoModel.from_pretrained(self.checkpoint_path, from_pt=True, trainable=False)
        # plm_model.resize_token_embeddings(self.rep.vocab_len) 
        x1_in = Input(shape=(self.maxlen,),dtype=tf.int32, name='input_ids')
        x2_in = Input(shape=(self.maxlen,),dtype=tf.int32, name='token_type_ids')
        x3_in = Input(shape=(self.maxlen,),dtype=tf.int32, name='attention_mask')
        # x = plm_model(x1_in, token_type_ids=x2_in, attention_mask=x3_in)[0]
        # x = plm_model(x1_in, token_type_ids=x2_in, attention_mask=x3_in)[1]
        x = plm_model(x1_in, token_type_ids=x2_in, attention_mask=x3_in)[0][:,0,:]
        #dense = TimeDistributed(Dense(512, activation='relu'), name='dense1')(x)
        # x = GlobalMaxPooling1D()(x)
        x = Dense(64, activation='relu', name='dense2')(x)
        # x= Dropout(0.2)(x)
        # output = Dense(self.rep.label_table_size, activation='softmax', name='softmax')(x)
        output = Dense(self.rep.label_table_size, activation='softmax', name='softmax')(x)
        self.model = Model(inputs=[x1_in,x2_in,x3_in], outputs=output, name="hugface_softmax")

       
        # opt = Adam(learning_rate = lr_schedule)
        opt = Adam(lr=1e-4) 
        # opt = RMSprop(lr=0.001)
        self.model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        # self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.summary()
        
        
    def load_model(self,model_file):
        self.model.load_weights(model_file)
        self.model.summary()  
        print('load HUGFACE model done!')
