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

def run_gsc_test(input, output, biotag_dic, nn_model, para_set):
    

    fin_test=open(input,'r',encoding='utf-8')
    all_test=fin_test.read().strip().split('\n\n')
    fin_test.close()
    test_out=open(output,'w',encoding='utf-8')
    #i=0
    for doc_test in tqdm(all_test):
        #i+=1
        #print(i)
        lines=doc_test.split('\n')
        pmid = lines[0]
        test_result = bioTag(lines[1],biotag_dic,nn_model,onlyLongest=para_set['onlyLongest'],abbrRecog=para_set['abbrRecog'],Threshold=para_set['ML_Threshold'])
        # test_result = bioTag_ml(lines[1],nn_model,onlyLongest=False,abbrRecog=False, Threshold=0.95)
        # test_result = bioTag_dic(lines[1],biotag_dic,onlyLongest=False, abbrRecog=False)
        test_out.write(pmid+'\n'+lines[1]+'\n')
        for ele in test_result:
            test_out.write(ele[0]+'\t'+ele[1]+'\t'+lines[1][int(ele[0]):int(ele[1])]+'\t'+ele[2]+'\t'+ele[3]+'\n')
        test_out.write('\n')
    test_out.close()
    GSCplus_corpus_hponew(output,input,subtree=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeltype', '-m', help="the model type (pubmedbert or biobert or bioformer?)",default='biobert')
    parser.add_argument('--input', '-i', help="the input prediction file ")
    parser.add_argument('--output', '-o', help="the output prediction file ")
    
    args = parser.parse_args()

    gpu = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) > 0:
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    para_set={
              'onlyLongest':False, # False: return overlap concepts, True only longgest
              'abbrRecog':False,# False: don't identify abbr, True: identify abbr
            #   'negation': True, #True:negation detection
              'ML_Threshold':0.95,# the Threshold of deep learning model
              }
    
    ontfiles={'dic_file':'../dicts/dict202402_sys_2/noabb_lemma.dic',
              'word_hpo_file':'../dicts/dict202402_sys_2/word_id_map.json',
              'hpo_word_file':'../dicts/dict202402_sys_2/id_word_map.json'}
    biotag_dic=dic_ont(ontfiles)
    
    emb = embedding_load(filename='../emb_2024/transR_512.emb')

    if args.modeltype=='biobert':
        vocabfiles={'labelfile':'../emb_label/transR_label.vocab',
                    'checkpoint_path':'../models_v1.2/biobert-base-cased-v1.2',
                    'lowercase':False}
        modelfile='../models/biobert_transR.h5'
        nn_model=bioTag_BERT(vocabfiles, emb)
        nn_model.load_model(modelfile)
    elif args.modeltype=='bioformer':
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


    run_gsc_test(args.input, args.output, biotag_dic, nn_model, para_set)