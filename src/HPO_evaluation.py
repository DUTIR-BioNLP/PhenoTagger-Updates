# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:33:22 2020

@author: luol2
"""
import argparse
from nn_model import bioTag_CNN,bioTag_BERT
from dic_ner import dic_ont
from evaluate import GSCplus_corpus,JAX_corpus, GSCplus_corpus_hponew
from tagging_text import bioTag, bioTag_dic, bioTag_ml
import os
import time
import json
import tensorflow as tf
from embedding_process import embedding_load
from tqdm import tqdm
# 将代码放在CPU上运行
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
session = tf.Session(config=config) 
'''
def run_gsc_test(files,biotag_dic,nn_model):
    

    fin_test=open(files['testfile'],'r',encoding='utf-8')
    all_test=fin_test.read().strip().split('\n\n')
    fin_test.close()
    test_out=open(files['outfile'],'w',encoding='utf-8')
    #i=0
    for doc_test in tqdm(all_test):
        #i+=1
        #print(i)
        lines=doc_test.split('\n')
        pmid = lines[0]
        test_result = bioTag(lines[1],biotag_dic,nn_model,onlyLongest=False,abbrRecog=False,Threshold=0.95)
        # test_result = bioTag_ml(lines[1],nn_model,onlyLongest=False,abbrRecog=False, Threshold=0.95)
        # test_result = bioTag_dic(lines[1],biotag_dic,onlyLongest=False, abbrRecog=False)
        test_out.write(pmid+'\n'+lines[1]+'\n')
        for ele in test_result:
            test_out.write(ele[0]+'\t'+ele[1]+'\t'+lines[1][int(ele[0]):int(ele[1])]+'\t'+ele[2]+'\t'+ele[3]+'\n')
        test_out.write('\n')
    test_out.close()
    GSCplus_corpus_hponew(files['outfile'],files['testfile'],subtree=True)

def run_jax_test(files,biotag_dic,nn_model):
    inpath=files['testfile']
    test_out=open(files['outfile'],'w',encoding='utf-8')
    i=0
    preds_result={}
    for file in os.listdir(inpath):
        i+=1
        print(i)
        pmid=file[:-4]
        temp_result=[]
        fin=open(inpath+file,'r',encoding='utf-8')
        intext=fin.read().rstrip()
        fin.close()     
        test_result=bioTag(intext,biotag_dic,nn_model,onlyLongest=False,abbrRecog=True,Threshold=0.95)
        for ele in test_result:
            if ele not in temp_result:
                temp_result.append(ele)
        preds_result[pmid]=temp_result
    json.dump(preds_result, test_out ,indent=2)
    test_out.close()
    JAX_corpus(files['outfile'], files['goldfile'])


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='build weak training corpus, python build_dict.py -i infile -o outpath')
    parser.add_argument('--modeltype', '-m', help="the model type (pubmedbert or biobert or bioformer?)",default='biobert')
    parser.add_argument('--output', '-o', help="the output prediction file ",default='../example/output2/gsc_bioformer_new2.tsv')
    
    args = parser.parse_args()
    model_type=args.modeltype
    test_set=args.corpus

    gpu = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) > 0:
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)
    
    ontfiles={'dic_file':'../dicts/dict202402_sys_2/noabb_lemma.dic',
              'word_hpo_file':'../dicts/dict202402_sys_2/word_id_map.json',
              'hpo_word_file':'../dicts/dict202402_sys_2/id_word_map.json'}
    biotag_dic=dic_ont(ontfiles)
    
    emb = embedding_load(filename='../emb_2024/transR_512.emb')
        
    if model_type=='biobert':
        vocabfiles={'labelfile':'../emb_label/transR_label.vocab',
                    'checkpoint_path':'../models_v1.2/biobert-base-cased-v1.2',
                    'lowercase':False}
        modelfile='../models/biobert_transR.h5'
        nn_model=bioTag_BERT(vocabfiles, emb)
        nn_model.load_model(modelfile)
    elif model_type=='bioformer':
        vocabfiles={'labelfile':'../emb_label/transR_label.vocab',
                    'checkpoint_path':'../models_v1.2/bioformer-cased-v1.0',
                    'lowercase':False}
        modelfile='../models/bioformer_transR.h5'
        nn_model=bioTag_BERT(vocabfiles, emb)
        nn_model.load_model(modelfile)
    else:
        vocabfiles={'labelfile':'../emb_label/transR_label.vocab',
                    'checkpoint_path':'../models_v1.2/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                    'lowercase':True}
        modelfile ='../models/pubmedbert_transR.h5'
        nn_model=bioTag_BERT(vocabfiles, emb)
        nn_model.load_model(modelfile)
    
    if test_set=='gsc':
        files={
                'testfile':'../data/GSC_2024/GSC_2024_test.tsv',
                'outfile':'../example/output2/gsc_bioformer_new2.tsv'}
        files['outfile']=args.output
        start_time=time.time()
        run_gsc_test(files,biotag_dic,nn_model)
        print('gsc done:',time.time()-start_time)
