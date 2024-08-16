# An Improved Method for Phenotype Concept Recognition Using Rich HPO Information

## Content

- [Dependency package](#package)
- [Data and model preparation](#preparation)
- [Instructions for tagging](#tagging)
- [Instructions for training](#training)


## Dependency package

<a name="package"></a>
PhenoTagger has been tested using Python3.9.19 on CentOS and uses the following dependencies on a CPU and GPU:

- [TensorFlow 2.12.0](https://www.tensorflow.org/)
- [Transformers 4.30.1](https://huggingface.co/docs/transformers/index)
- [NLTK 3.8.1](www.nltk.org)


To install all dependencies automatically using the command:

```
$ pip install -r requirements.txt
```

## Data and model preparation

<a name="preparation"></a>

1. To run this code, you need to create a model folder named "models" in the PhenoTagger folder, then download the model files ( four trained models for HPO concept recognition are released, i.e., CNN, Bioformer, BioBERT, PubMedBERT) into the model folder.

   - First download original files of the pre-trained language models (PLMs): [Bioformer](https://huggingface.co/bioformers/bioformer-8L/), [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2), [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
2. The two typo-corpora are provided in */data/ 

## Tagging

<a name="tagging"></a>

You can identify the HPO concepts from biomedical texts by the *HPO_evaluation.py* file.


The file requires 2 parameters:

- --modeltype, -m, help="the model type (pubmedbert or biobert or bioformer?)"
- --output, -o, help="output folder to save the tagged results"

Example:

```
$ CUDA_VISIBLE_DEVICES=0 python HPO_evaluation.py -m biobert -o ../example/output/
```

## Training

<a name="training"></a>

### 1. Build the ontology dictionary using the *Build_dict.py* file

The file requires 3 parameters:

- --input, -i, help="input the ontology .obo file"
- --output, -o, help="the output folder of dictionary"
- --rootnode, -r, help="input the root node of the ontogyly"

Example:

```
$ python Build_dict.py -i ../ontology/hp.obo -o ../dict/ -r HP:0000118
```

After the program is finished, 6 files will be generated in the output folder.

- id\_word\_map.json
- lable.vocab
- noabb\_lemma.dic
- obo.json
- word\_id\_map.json
- alt\_hpoid.json

### 2. Build the distant supervised training dataset using the *Build_distant_corpus.py* file

The file requires 4 parameters:

- --dict, -d, help="the input folder of the ontology dictionary"
- --fileneg, -f, help="the text file used to generate the negatives" (You can use our negative text ["mutation_disease.txt"](https://ftp.ncbi.nlm.nih.gov/pub/lu/PhenoTagger/mutation_disease.zip) )
- --negnum, -n, help="the number of negatives, we suggest that the number is the same with the positives."
- --output, -o, help="the output folder of the distantly-supervised training dataset"

Example:

```
$ python Build_distant_corpus.py -d ../dict/ -f ../data/mutation_disease.txt -n 50000 -o ../data/distant_train_data/
```

After the program is finished, 3 files will be generated in the outpath:

- distant\_train.conll       (distantly-supervised training data)
- distant\_train\_pos.conll  (distantly-supervised training positives)
- distant\_train\_neg.conll  (distantly-supervised training negatives)

### 3. Training Ontology Vector

The ontology vector was trained using *TransE.py* and *TransR.py*. For the ConvE methods, please refer to [https://github.com/TimDettmers/ConvE].

After training, the vectors were processed using emb_process.py for format handling.

### 4. Training using the *training.py* file

The file requires 4 parameters:

- --trainfile, -t, help="the training file"
- --devfile, -d, help="the development set file. If don't provide the dev file, the training will be stopped by the specified EPOCH"
- --modeltype, -m, help="the deep learning model type (cnn, biobert, pubmedbert or bioformer?)"
- --output, -o, help="the output folder of the model"

Example:

```
$ CUDA_VISIBLE_DEVICES=0 python training.py -t ../data/distant_train_data/distant_train.conll -d ../data/corpus/GSC/GSC-2024_dev.tsv -m biobert -o ../models/
```
