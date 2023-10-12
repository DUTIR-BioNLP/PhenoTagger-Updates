import spacy
from negspacy.negation import Negex
from spacy.tokens import Token
import en_core_sci_sm

nlp = en_core_sci_sm.load(disable=["tagger", "parser", "lemmatizer"])
# nlp.add_pipe(nlp.create_pipe('sentencizer'))
nlp.add_pipe('sentencizer')
# negex = Negex(nlp, language='en', chunk_prefix=["no"])
nlp.add_pipe('negex', last=True)
Token.set_extension('negex', default=False, force=True)

#label of the negation detection 
POS_LABEL='POS'
NEG_LABEL='NEG'

def check_overlap(phrase1, phrase2):
    words1 = phrase1.split()
    words2 = phrase2.split()
    
    overlapping_words = set(words1) & set(words2)

    return len(overlapping_words) > 0

def neg_iden(text, final_result):
    neg_list = []
    for e in nlp(text).ents:
        # print(e.text,e._.negex)
        if e._.negex:
            neg_list.append(e.text)
            # print(e.text)
    # print(neg_list)
    for i in range(len(final_result)):
        if len(neg_list) == 0:
            final_result[i].append(POS_LABEL)
        else:
            start = int(final_result[i][0])
            end = int(final_result[i][1])
            term = text[start:end]
            for j in range(len(neg_list)):
                is_neg = check_overlap(term, neg_list[j])
                if is_neg:
                    final_result[i].append(NEG_LABEL)
                    
                    break
            if len(final_result[i]) == 4:
                final_result[i].append(POS_LABEL)
                
    return final_result
    # print(final_result)        
# neg_iden("MOUTH: no telangiectases noted",
#          [['10', '24', 'HP:0001009', '1.00']])