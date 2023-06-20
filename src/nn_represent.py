# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:02:20 2020

@author: luol2
"""
import time
import os, sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras_bert import Tokenizer
from transformers import AutoTokenizer


class CNN_RepresentationLayer(object):
    
    
    def __init__(self, wordvec_file,  vocab_file=[],\
                 vec_size=50, word_size=10000, frequency=10000):
        
        '''
        wordvec_file    ：    the file path of word embedding
        vec_size        :    the dimension size of word vector 
                             learned by word2vec tool
        
        word_size       :    the size of word vocabulary
  
        frequency       :    the threshold for the words left according to
                             their frequency appeared in the text
                             for example, when frequency is 10000, the most
                             frequent appeared 10000 words are considered
        
        '''
        #load word embedding
        file = open(wordvec_file)
        first_line = file.readline().strip()
        file.close()
        self.word_size = int(first_line.split()[0])
        self.vec_size = int(first_line.split()[1])
        self.frequency = frequency
        
        if self.frequency>self.word_size:
            self.vec_table = np.zeros((self.word_size + 2, self.vec_size))
        else:
            self.vec_table = np.zeros((self.frequency + 2, self.vec_size))
        self.word_2_index = {}
        self.load_wordvecs(wordvec_file)
        
        #other fea
        self.char_2_index={}
        self.char_table_size=0
        if 'char' in vocab_file.keys():
            self.load_fea_vocab(vocab_file['char'],self.char_2_index)
            self.char_table_size=len(self.char_2_index)
            print(self.char_table_size)
            #print(self.char_2_index)
        
        self.label_2_index={}
        self.label_table_size=0
        if 'label' in vocab_file.keys():
            self.load_label_vocab(vocab_file['label'],self.label_2_index)
            self.label_table_size=len(self.label_2_index)
            print(self.label_table_size)
            #print(self.char_2_index) 
                        
        self.pos_2_index={}
        self.pos_table_size=0
        if 'pos' in vocab_file.keys():
            self.load_fea_vocab(vocab_file['pos'],self.pos_2_index)
            self.pos_table_size=len(self.pos_2_index)
            print(self.pos_table_size)
            

    
    def load_wordvecs(self, wordvec_file):
        
        file = open(wordvec_file,'r',encoding='utf-8')
        file.readline()
        print(self.word_size)
        print(self.vec_size)
        row = 0
        self.word_2_index['padding_0'] = row #oov-zero vector
        row+=1
        for line in file:
            if row <= self.word_size and row <= self.frequency:
                line_split = line.strip().split(' ')
                self.word_2_index[line_split[0]] = row
                for col in range(self.vec_size):
                    self.vec_table[row][col] = float(line_split[col + 1])
                row += 1
            else:
                break
        
        self.word_2_index['sparse_vectors'] = row #oov-zero vector      
        file.close()        

    def load_fea_vocab(self,fea_file,fea_index):
        fin=open(fea_file,'r',encoding='utf-8')
        i=0
        fea_index['padding_0']=i
        i+=1
        fea_index['oov_padding']=i
        i+=1
        for line in fin:
            fea_index[line.strip()]=i
            i+=1
        fin.close()
        
    def load_label_vocab(self,fea_file,fea_index):
        fin=open(fea_file,'r',encoding='utf-8')
        i=0
        for line in fin:
            fea_index[line.strip()]=i
            i+=1
        fin.close()
    
    '''
    def generate_label_list(self,labels):
        label_list=[]
        
        for label in labels:
            temp_label=[0]*self.label_table_size
            temp_label[self.label_2_index[label]]=1
            label_list.append(temp_label)
        return label_list
    '''
    def generate_label_list(self,labels):
        sparse_labels=[]
        for ele in labels:
            sparse_labels.append(self.label_2_index[ele])
        return(sparse_labels)
    
    def represent_instances_all_feas(self, instances, labels, word_max_len=100, char_max_len=50, training=False):

        x_text_list=[]
        x_word_list=[]
        x_char_list=[]
        x_lemma_list=[]
        x_pos_list=[]

        y_list=[]

        for sentence in instances:           
            sentence_list=[]
            sentence_word_list=[]
            sentence_lemma_list=[]
            sentence_pos_list=[]
            sentence_text=[]   
            for j in range(0,len(sentence)):
                word=sentence[j]
                #char fea               
                char_list=[0]*char_max_len
                for i in range(len(word[0])):
                    if i<char_max_len:
                        if word[0][i] in self.char_2_index.keys():
                            char_list[i]=self.char_2_index[word[0][i]]
                        else:
                            char_list[i]=self.char_2_index['oov_padding']
                sentence_word_list.append(char_list)
                
                #word fea
                sentence_text.append(word[0].lower())
                if word[0].lower() in self.word_2_index.keys():
                    sentence_list.append(self.word_2_index[word[0].lower()])                        
                else:
                    sentence_list.append(self.word_2_index['sparse_vectors'])
                
                #lemma fea
                if word[1].lower() in self.word_2_index.keys():
                    sentence_lemma_list.append(self.word_2_index[word[1].lower()])
                else:
                    sentence_lemma_list.append(self.word_2_index['sparse_vectors'])
             
                #pos fea
                if word[3] in self.pos_2_index.keys():
                    sentence_pos_list.append(self.pos_2_index[word[3]])
                else:
                    sentence_pos_list.append(self.pos_2_index['oov_padding'])
                
            x_text_list.append(sentence_text)
            x_word_list.append(sentence_list)
            x_char_list.append(sentence_word_list)
            x_lemma_list.append(sentence_lemma_list)
            x_pos_list.append(sentence_pos_list)
        
        if training==True:
            y_list=self.generate_label_list(labels)
            x_word_np = pad_sequences(x_word_list, word_max_len, value=0, padding='post',truncating='post')  # right padding
            x_char_np = pad_sequences(x_char_list, word_max_len, value=0, padding='post',truncating='post')
            x_lemma_np = pad_sequences(x_lemma_list, word_max_len, value=0, padding='post',truncating='post')                
            x_pos_np = pad_sequences(x_pos_list, word_max_len, value=0, padding='post',truncating='post')
            y_np = np.array(y_list)
            
        else:
            x_word_np = pad_sequences(x_word_list, word_max_len, value=0, padding='post',truncating='post')  # right padding
            x_char_np = pad_sequences(x_char_list, word_max_len, value=0, padding='post',truncating='post')
            x_lemma_np=[]
            x_pos_np=[]
            y_np=[]
            
        return [x_word_np, x_char_np, x_lemma_np,  x_pos_np, x_text_list], y_np        



class BERT_RepresentationLayer(object):
    
    
    def __init__(self, tokenizer_name_or_path, label_file,lowercase=True):
        

        #load vocab
        self.model_type='bert'
        #self.model_type='roberta'
        if self.model_type in {"gpt2", "roberta"}:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True,do_lower_case=lowercase)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True,do_lower_case=lowercase)

        #load label
        self.label_2_index={}
        self.index_2_label={}
        self.label_table_size=0
        self.load_label_vocab(label_file,self.label_2_index,self.index_2_label)
        self.label_table_size=len(self.label_2_index)
        self.vocab_len=len(self.tokenizer)

    def load_label_vocab(self,fea_file,fea_index,index_2_label):
        
        fin=open(fea_file,'r',encoding='utf-8')
        all_text=fin.read().strip().split('\n')
        fin.close()
        for i in range(0,len(all_text)):
            fea_index[all_text[i]]=i
            index_2_label[str(i)]=all_text[i]
            
    def generate_label_list(self,labels):
        sparse_labels=[]
        for ele in labels:
            sparse_labels.append(self.label_2_index[ele])       
        return(sparse_labels)
    
    def load_data(self,instances, labels,  word_max_len=100,training=False):
    
        x_index=[]
        x_seg=[]
        x_mask=[]
        y_list=[]
        
        for sentence in instances:                           
            sentence_text_list=[]
            for j in range(0,len(sentence)):
                sentence_text_list.append(sentence[j][0].lower()) #input lower
                
            token_result=self.tokenizer(
                sentence_text_list,
                max_length=word_max_len,
                truncation=True,is_split_into_words=True)
            
            bert_tokens=self.tokenizer.convert_ids_to_tokens(token_result['input_ids'])
            word_index=token_result.word_ids(batch_index=0)
            

            x_index.append(token_result['input_ids'])
            if self.model_type in {"gpt2", "roberta"}:
                x_seg.append([0]*len(token_result['input_ids']))
            else:
                x_seg.append(token_result['token_type_ids'])
            x_mask.append(token_result['attention_mask'])              
        
        if training==True:
            y_list=self.generate_label_list(labels)
        
            x1_np = pad_sequences(x_index, word_max_len, value=0, padding='post',truncating='post')  # right padding
            x2_np = pad_sequences(x_seg, word_max_len, value=0, padding='post',truncating='post')
            x3_np = pad_sequences(x_mask, word_max_len, value=0, padding='post',truncating='post')
            y_np = np.array(y_list)
        
        else:
            x1_np = pad_sequences(x_index, word_max_len, value=0, padding='post',truncating='post')  # right padding
            x2_np = pad_sequences(x_seg, word_max_len, value=0, padding='post',truncating='post')
            x3_np = pad_sequences(x_mask, word_max_len, value=0, padding='post',truncating='post')
            y_np=[]

        return [x1_np, x2_np, x3_np], y_np  

if __name__ == '__main__':
    pass
 
            
