'''
将transR训练后格式转化为PhenoTagger需要的格式
'''

import numpy as np

array_length = 512
my_array = np.random.uniform(low=-1, high=1, size=array_length)

femb = open('./transR_emb_2024/TransR_pytorch_entity_512dim_batch400', 'r', encoding='utf-8')
embs = femb.read().strip().split('\n')
femb.close()

entity2id = {}

file = open('./HPO_2024/entity2id.txt', 'r', encoding='utf-8')
allids = file.readlines()
file.close()
for line in allids:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[1]] = line[0]
# print(emb[0])
# print(entity2id)
fout = open('./emb_2024/transR_512.emb', 'w', encoding='utf-8')

all_label = open('./dict/lable.vocab', 'r', encoding='utf-8')
labels = all_label.read().strip().split('\n')
all_label.close()

for per_emb in embs:
    id, emb = per_emb.split('\t')
    emb = emb.replace('[', '')
    emb = emb.replace(']', '')
    emb = emb.split(', ')
    # print(id, emb)
    if entity2id[id] in labels:
        fout.write(entity2id[id]+' ')
        for embedding in emb:
            fout.write(embedding)
            if embedding == emb[-1]:
                fout.write('\n')
            else:
                fout.write(' ')
fout.write("HP:None" + ' ')        
for i in range(array_length):

    fout.write(str(my_array[i]))
    if i == array_length-1:
        fout.write('\n')
    else:
        fout.write(' ')
fout.close()
    # break