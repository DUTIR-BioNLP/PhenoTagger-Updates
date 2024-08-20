import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from nlpaug.util import Action
import random
import argparse

random.seed(777)

def aug_make(name):

    ins_name = ins_aug.augment(name)[0]
    del_name = del_aug.augment(name)[0]
    sub_name = sub_aug.augment(name)[0]
    swap_name = swap_aug.augment(name)[0]

    return [ins_name, del_name, sub_name, swap_name]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make typo data. -i input_path, -o outpath')
    parser.add_argument('--input', '-i', help="Input ontology path.",default='../ontology/hp20240208.obo')
    parser.add_argument('--output', '-o', help="Output typo_ontology path.",default='../ontology/typo_hpo.obo')
    args = parser.parse_args()

    with open(args.input, 'r') as hpof:
        hpo_datas = hpof.read().strip().split('\n\n')
    fout = open(args.output, 'w')
    # 增删替换
    ins_aug = nac.RandomCharAug(action=Action.INSERT, aug_char_max=2, aug_char_p=0.2,
                 aug_word_p=0.2, aug_word_max=1, include_upper_case=False, include_numeric=False,spec_char='')
    del_aug = nac.RandomCharAug(action=Action.DELETE, aug_char_max=2, aug_char_p=0.2,
                 aug_word_p=0.2, aug_word_max=1, include_upper_case=False, include_numeric=False,spec_char='')
    sub_aug = nac.RandomCharAug(action=Action.SUBSTITUTE, aug_char_max=2, aug_char_p=0.2,
                 aug_word_p=0.2, aug_word_max=1, include_upper_case=False, include_numeric=False,spec_char='')
    swap_aug = nac.RandomCharAug(action=Action.SWAP, aug_char_max=2, aug_char_p=0.2,
                 aug_word_p=0.2, aug_word_max=1, include_upper_case=False, include_numeric=False,spec_char='')
    fout.write(hpo_datas[0]+'\n\n')
    for i in range(1,len(hpo_datas)):
        fout.write(hpo_datas[i]+'\n')
        lines = hpo_datas[i].split('\n')
        name = lines[2][len('name: '):]
        aug_names = aug_make(name)
        # print(aug_names)
        fout.write(f'synonym: "{aug_names[0]}"\n')
        fout.write(f'synonym: "{aug_names[1]}"\n')
        fout.write(f'synonym: "{aug_names[2]}"\n')
        fout.write(f'synonym: "{aug_names[3]}"\n')
        fout.write('\n')
        # if i == 10:
        #     break
    fout.close()