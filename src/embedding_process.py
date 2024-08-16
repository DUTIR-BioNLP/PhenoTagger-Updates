import numpy as np
import pandas as pd

def embedding_load(filename):
    # filename = '/home/Users/qjw/PhenoTagger-Updates/emb/new_emb_all.emb'  # 替换为你的实际文件路径
    df = pd.read_csv(filename, sep=' ', header=None)
    df = df.drop(df.columns[0],axis=1)
    # print(df)
    numpy_array = df.to_numpy().astype(np.float32)
    # print(numpy_array)

    return numpy_array

def NCR_embedding_load():
    from onto import Ontology

    ont = Ontology('../ontology/hp_20191108.obo','HP:0000118')
    # emb = tf.sparse.reorder(tf.SparseTensor(
    #     indices = ont.sparse_ancestors,
    #     values = ont.sparse_ancestors_values,
    #     dense_shape=[14370, 14370]))
    
    return ont.sparse_ancestors, ont.sparse_ancestors_values

if __name__ == '__main__':
    emb = embedding_load(filename='/home/Users/qjw/PhenoTagger-Updates/emb/new_emb_all_256.emb')
    f = open('/home/Users/qjw/PhenoTagger-Updates/emb/new_emb_all_256.emb', 'r', encoding='utf-8')
    datas = f.readlines()
    data = datas[1]
    data = data.strip().split(' ')
    print(len(data))
    for data in datas:
        data = data.strip().split(' ')
        if len(data) != 257:

            print(len(data))
    # from onto import Ontology

    # ont = Ontology('../ontology/hp_20191108.obo','HP:0000118')
    # fout = open('/home/Users/qjw/PhenoTagger-Updates/dict/lable_NCR.vocab', 'w', encoding='utf-8')
    # for concept in ont.concepts:
    #     fout.write(concept+'\n')
    # fout.close()