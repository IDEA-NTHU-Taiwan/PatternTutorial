import patternlib.utils as utils
import multiprocessing
import pandas as pd
import nltk

def token2cwsw(tokens, cw_list, sw_list, cw_token='cw', 
               sw_token='sw', none_token='_', both_token='both'):
    """
    The rule here is very flexible.
    """
    cwsw_list = []
    for token in tokens:
        if token in cw_list and token in sw_list:
            cwsw_list.append(both_token)
        elif token in cw_list:
            cwsw_list.append(cw_token)
        elif token in sw_list:
            cwsw_list.append(sw_token)
        else:
            cwsw_list.append(none_token)
    return cwsw_list


def list_match(l1, l2):
    if len(l1) != len(l2):
        return False
    for idx in range(len(l1)):
        if l1[idx] != l2[idx]:
            return False
    return True



def build_patternDict(df, label_list, pattern_templates, n_jobs=-1):

    if n_jobs == 1:
        ## no mp
        pattern_dict = dict()
        build_patternDict_(df=df, pattern_dict=pattern_dict, 
                           label_list=label_list, pattern_templates=pattern_templates)
    else:
        if n_jobs == -1:
            cores = multiprocessing.cpu_count() 
        else:
            cores = int(n_jobs)

        # a memory-shared dictionary `pattern_dict`
        manager = multiprocessing.Manager()
        pattern_dict = manager.dict()

        # build pattern-dictionary
        utils.apply_by_multiprocessing(df, build_patternDict_, 
                                       pattern_dict=pattern_dict, label_list=label_list, 
                                       pattern_templates=pattern_templates, workers=cores)
    return pattern_dict



def build_patternDict_(df, pattern_dict, label_list, pattern_templates):

    cwsw_text = df['cwsw_text']
    tokenized_text = df['tokenized_text']
    label = df['emotion']

    for target_pattern in pattern_templates:
        cwsw_ngrams = list(nltk.ngrams(cwsw_text, len(target_pattern))) 
        token_ngrams = list(nltk.ngrams(tokenized_text, len(target_pattern))) 

        ## pattern rules defined here
        ## (I don't consider the `both` token here)
        for idx, cwsw_gram in enumerate(cwsw_ngrams):
            if list_match(cwsw_gram, target_pattern):
                token_ngram = token_ngrams[idx]
                general_token_ngram = []
                for j, cwsw in enumerate(cwsw_gram):
                    if cwsw == 'sw':
                        general_token_ngram.append('*')
                    else:
                        general_token_ngram.append(token_ngram[j])
                pattern_get = ' '.join(general_token_ngram)

                ## be careful of updating the `pattern_dict`
                if pattern_get not in pattern_dict:
                    default_dict = {}
                    default_dict['template'] = ' '.join(cwsw_gram)
                    default_dict['diversity'] = []
                    for label_ in label_list:
                        default_dict[label_] = 0
                    pattern_dict[pattern_get] = default_dict

                temp_dict = pattern_dict[pattern_get]
                temp_dict[label] += 1
                temp_dict['diversity'].append(token_ngram)
                pattern_dict[pattern_get] = temp_dict



def build_patternDF(pattern_dict, label_list):
    df_pattern = pd.DataFrame()
    pattern_list = list(pattern_dict.keys())
    df_pattern['pattern'] = pattern_list
    col_list = ['template'] + label_list + ['diversity']
    temp_list = [[] for _ in range(len(col_list))]
    for pattern in pattern_list:
        for i, col in enumerate(col_list):
            temp_list[i].append(pattern_dict[pattern][col])
    for i, col in enumerate(col_list):
        df_pattern[col] = temp_list[i]
    return df_pattern






