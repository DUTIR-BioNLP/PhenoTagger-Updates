import argparse
from nn_model import bioTag_CNN,bioTag_BERT
from dic_ner import dic_ont
from evaluate import GSCplus_corpus,JAX_corpus, GSCplus_corpus_hponew
from tagging_text import bioTag, bioTag_dic, bioTag_ml
import os
import time
import json
import tensorflow as tf
import bioc
from embedding_process import embedding_load
from tqdm import tqdm
from negbio2.negbio_run import negbio_load, negbio_main

def neg_iden(infile, outfile):
    # gsc to xml
    fin = open(infile, 'r', encoding='utf-8')
    all_in = fin.read().strip().split('\n\n')
    fin.close()
    collection = bioc.BioCCollection()
    for doc in all_in:
        lines = doc.split('\n')
        pmid = lines[0]
        text = lines[1]

        document = bioc.BioCDocument()
        document.id = pmid

        passage = bioc.BioCPassage()
        passage.offset = 0

        passage.text = text.strip()
        document.add_passage(passage)

        mention_num = 0
        for i in range(2, len(lines)):
            ele = lines[i].split('\t')
            bioc_node = bioc.BioCAnnotation()
            bioc_node.id = str(mention_num)
            if len(ele) == 1:
                continue
            bioc_node.infons['identifier'] = ele[3]
            bioc_node.infons['type'] = "Phenotype"
            bioc_node.infons['score'] = ele[4]
            start = int(ele[0])
            last = int(ele[1])
            loc = bioc.BioCLocation(offset=str(passage.offset+start), length= str(last-start))
            bioc_node.locations.append(loc)
            bioc_node.text = passage.text[start:last]
            passage.annotations.append(bioc_node)
            mention_num += 1
        collection.add_document(document)
    with open(outfile, 'w') as fp:
        bioc.dump(collection, fp)
    
    pipeline, argv = negbio_load()
    negbio_main(pipeline, argv, outfile, './')

def turn_xml_to_bc8(output):
    fin = open('./BC8_neg.neg2.xml', 'r', encoding='utf-8')
    neg2_results = {}
    collection = bioc.load(fin)
    fin.close()

    for document in collection.documents:
        pmid = document.id
        _mention = {}
        for passage in document.passages:
            for men_node in passage.annotations:
                if 'uncertainty' in men_node.infons.keys():
                    _mention[men_node.id] = 'uncertainty'
                elif 'negation' in men_node.infons.keys():
                    _mention[men_node.id] = 'negation'
                else:
                    _mention[men_node.id] = 'positive'
        neg2_results[pmid] = _mention


    with open('./BC8_temp.tsv', 'r', encoding='utf-8') as fin:
        all_in = fin.read().strip().split('\n\n')
    

    fout = open(output, 'w', encoding='utf-8')
    fout.write(f'ObservationID\tText\tHPO Term\n')

    for doc in all_in:
        lines = doc.split('\n')
        pmid, text = lines[0], lines[1]

        if neg2_results[pmid] == {}:
            fout.write(f'{pmid}\t{text}\tNA\n')
        elif len(neg2_results[pmid]) == 1:
            # print(neg2_results[pmid])
            # print(lines)
            # break
            if neg2_results[pmid]['0'] == 'positive':
                hpid = lines[2].split('\t')[3]
                fout.write(f'{pmid}\t{text}\t{hpid}\n')
            else:
                fout.write(f'{pmid}\t{text}\tNA\n')
        else:
            if len(list(set(neg2_results[pmid].values()))) == 1 and list(set(neg2_results[pmid].values()))[0] != 'positive':
                fout.write(f'{pmid}\t{text}\tNA\n')
            for anno_id in neg2_results[pmid].keys():
                if neg2_results[pmid][anno_id] == 'positive':
                    hpid = lines[2+int(anno_id)].split('\t')[3]
                    fout.write(f'{pmid}\t{text}\t{hpid}\n')
                else:
                    continue
            
    fout.close()

def run_gsc_test(input, output, biotag_dic, nn_model, para_set):
    

    fin_test=open(input,'r',encoding='utf-8')
    all_test=fin_test.read().strip().split('\n\n')
    fin_test.close()
    if para_set['negation'] == True:
        test_out=open('./BC8_temp.tsv','w',encoding='utf-8')
    else:
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
    # GSCplus_corpus_hponew(output,input,subtree=True)
    if para_set['negation'] == True:
        neg_iden(infile='./BC8_temp.tsv', outfile='./BC8_neg.xml')
        turn_xml_to_bc8(output)


        os.remove('./BC8_temp.tsv')
        os.remove('./BC8_neg.xml')
        os.remove('./BC8_neg.neg2.xml')


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
              'negation': False, #True:negation detection
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