# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 19:54:17 2021

@author: luol2
"""



import os, sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

class NN_RepresentationLayer(object):
    
    
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
        self.index_2_label={}
        self.load_label_vocab(vocab_file['label'],self.label_2_index,self.index_2_label)
        self.label_table_size=len(self.label_2_index)
 


    def load_label_vocab(self,fea_file,fea_index,index_2_label):
        
        fin=open(fea_file,'r',encoding='utf-8')
        all_text=fin.read().strip().split('\n')
        fin.close()
        for i in range(0,len(all_text)):
            fea_index[all_text[i]]=i
            index_2_label[str(i)]=all_text[i]
                        
    
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
        
    
    def generate_label_list(self,labels):
        sparse_labels=[]
        for ele in labels:
            if self.label_2_index[ele] == 0:
                # print('POS')
                # sparse_labels.append(self.label_2_index[ele])
                sparse_labels.append([1.,0.])
            else:
                # print('NEG')
                sparse_labels.append([0.,1.])
        return(sparse_labels)
    
    def represent_instances_all_feas(self, instances, labels, word_max_len=100, char_max_len=50, training=False):

        x_text_list=[]
        x_word_list=[]
        x_char_list=[]
        x_lemma_list=[]
        x_pos_list=[]

        y_list=[]

        for ele in instances:           
            sentence_list=[]
            sentence_word_list=[]
            sentence_text=[] 
            sentence = ele.split()
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
                sentence_text.append(word.lower())
                if word.lower() in self.word_2_index.keys():
                    sentence_list.append(self.word_2_index[word.lower()])                        
                else:
                    sentence_list.append(self.word_2_index['sparse_vectors'])
                
                
            x_text_list.append(sentence_text)
            x_word_list.append(sentence_list)
            x_char_list.append(sentence_word_list)

        
        if training==True:
            y_list=self.generate_label_list(labels)
            x_word_np = pad_sequences(x_word_list, word_max_len, value=0, padding='post',truncating='post')  # right padding
            x_char_np = pad_sequences(x_char_list, word_max_len, value=0, padding='post',truncating='post')
            y_np = np.array(y_list)
            
        else:
            x_word_np = pad_sequences(x_word_list, word_max_len, value=0, padding='post',truncating='post')  # right padding
            x_char_np = pad_sequences(x_char_list, word_max_len, value=0, padding='post',truncating='post')
            y_np=[]
            
        return [x_word_np, x_char_np,], y_np        


         
class Hugface_RepresentationLayer(object):
    
    
    def __init__(self, tokenizer_name_or_path, label_file,lowercase=True):
        

        #load vocab
        #self.bert_vocab_dict = {}
        #self.cased=cased
        #self.load_bert_vocab(vocab_path,self.bert_vocab_dict)
        self.model_type='bert'
        #self.model_type='roberta'
        if self.model_type in {"gpt2", "roberta"}:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True,do_lower_case=lowercase)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True,do_lower_case=lowercase)
        
        # self.tokenizer.add_tokens(["<ENT>","</ENT>"])

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
            if self.label_2_index[ele] == 0:
                # print('POS')
                # sparse_labels.append(self.label_2_index[ele])
                sparse_labels.append([1.,0.])
            else:
                # print('NEG')
                sparse_labels.append([0.,1.])
        return(sparse_labels)
        
    
    
    def load_data_hugface(self,instances, labels,  word_max_len=100):
    
        x_index=[]
        x_seg=[]
        x_mask=[]

        bert_text_labels=[]
        max_len=0
        over_num=0
        maxT=word_max_len
        ave_len=0

        #print('instances:', instances)
        #print('labels:',labels)
        
        
        for sentence in instances:                           
            sentence_text_list=sentence.split()

            token_result=self.tokenizer(
                sentence_text_list,
                max_length=word_max_len,
                truncation=True,is_split_into_words=True)
            
            bert_tokens=self.tokenizer.convert_ids_to_tokens(token_result['input_ids'])
            # print(bert_tokens)
            word_index=token_result.word_ids(batch_index=0)
            ave_len+=len(bert_tokens)
            if len(bert_tokens)>max_len:
                max_len=len(bert_tokens)
            if len(bert_tokens)==maxT:
                over_num+=1

            x_index.append(token_result['input_ids'])
            if self.model_type in {"gpt2", "roberta"}:
                x_seg.append([0]*len(token_result['input_ids']))
            else:
                x_seg.append(token_result['token_type_ids'])
            x_mask.append(token_result['attention_mask'])

            #print('label:',label_list)
        label_list=self.generate_label_list(labels)
        
        x1_np = pad_sequences(x_index, word_max_len, value=0, padding='post',truncating='post')  # right padding
        x2_np = pad_sequences(x_seg, word_max_len, value=0, padding='post',truncating='post')
        x3_np = pad_sequences(x_mask, word_max_len, value=0, padding='post',truncating='post')
        y_np = np.array(label_list)
        
        #print('x1_np:',x1_np[0:2])
        #print('\nx2_np:',x2_np[0:2])
        #print('\ny_np:',y_np[0:2])

        print('bert max len:',max_len,',Over',maxT,':',over_num,'ave len:',ave_len/len(instances),'total:',len(instances))
      
        
        return [x1_np, x2_np,x3_np], y_np


            

if __name__ == '__main__':
    pass
    
 
            
