# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:59:07 2023

@author: luol2
"""

import argparse
import os
import time
import re
import io
import bioc
import subprocess

from tensorflow.keras.models import load_model
from model_tc import HUGFACE_TC,NN_TC
from ssplit_tokenzier import ssplit_token
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import sys


ENTag = [' entss ',' entee ']

def ml_intext_fn(alltexts):
    data_list=[]
    label_list=[]

    for doc in alltexts:
        seg=doc.split('\t')
        data_list.append(seg[1])
        label_list.append(seg[0])
    return data_list,label_list
#input model predict, out the top index
def tc_out(results):
    output_result=[]
    for ele in results:
        output_result.append(ele.argsort()[-1])
    return output_result

def tc_out_label(results,index_2_label):
    output_result=[]
    for ele in results:
        # print(ele,str(ele.argsort()[-1]),index_2_label[str(ele.argsort()[-1])])
        output_result.append(index_2_label[str(ele.argsort()[-1])])
    return output_result
    

def NegDec(text,model_tc,ent_result):
    
    #generate input instance
    neg_instace = []
    context_win = 6 #6words
    for ele in ent_result:
        before_seg = text[0:int(ele[0])].split(' ')
        if len(before_seg)>= context_win:
            before_txt = ' '.join(before_seg[(-1)*context_win:])
        else:
            before_txt = ' '.join(before_seg[0:])
        after_seg = text[int(ele[1]):].split(' ')
        if len(after_seg)>= context_win:
            after_txt = ' '.join(after_seg[0:context_win])
        else:
            after_txt = ' '.join(after_seg[0:])
        # _ins_text = text[0:int(ele[0])]+ENTag[0]+text[int(ele[0]):int(ele[1])]+ENTag[1]+text[int(ele[1]):]
        _ins_text = before_txt+ENTag[0]+text[int(ele[0]):int(ele[1])]+ENTag[1]+after_txt
        # print(_ins_text)
        neg_instace.append('NEG\t'+ssplit_token(_ins_text))
    
    if neg_instace!=[]:
        # NN TC
        if model_tc.model_type=='cnn':
            test_set,test_label = ml_intext_fn(neg_instace)
            test_x, test_y = model_tc.rep.represent_instances_all_feas(test_set,test_label,word_max_len=model_tc.hyper['sen_max'],char_max_len=model_tc.hyper['word_max'])
            input_test = []
        
            if model_tc.fea_dict['word'] == 1:
                input_test.append(test_x[0])
        
            if model_tc.fea_dict['char'] == 1:
                input_test.append(test_x[1])
        
        
            test_score = model_tc.model.predict(input_test,batch_size=256,verbose=0)
            ml_out=tc_out_label(test_score,model_tc.rep.index_2_label) #classification label 
        else:
            test_set,test_label = ml_intext_fn(neg_instace)
            test_x,test_y=model_tc.rep.load_data_hugface(test_set,test_label,word_max_len=model_tc.maxlen)
            test_score = model_tc.model.predict(test_x)
            ml_out=tc_out_label(test_score,model_tc.rep.index_2_label) #classification label 
        
        final_result=[]
        # print(ent_result,ml_out)
        for i in range(0,len(ent_result)):
            
            final_result.append(ent_result[i]+[ml_out[i]])
    else:
        final_result=[]


    return final_result


if __name__=="__main__":
    
 
    pass
    
    # if not os.path.exists(outpath):
    #     os.makedirs(outpath)
    # model_type='cnn'
    # infile='../trainingset/BioCreativeVIII3_TrainSet_ENTag-neg.tsv'
    # modelfile='../neg_models/cnn-neg-train-best.h5'
    # outfile='../_temp_results/train_bioformer-TC.tsv'
    # main(infile,model_type,outfile,modelfile)

    
