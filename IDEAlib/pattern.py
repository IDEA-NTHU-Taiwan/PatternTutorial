import IDEAlib.utils as utils
import multiprocessing
import pandas as pd
import numpy as np
import nltk

"""
TODO:
1. update the `patternDict` multiprocessing method


"""


def listcwsw(df, threshold, key_col='token', val_col='value'):
    """
    return the `np.appry` type of CW/SW list
    """
    return df[df[val_col]>=threshold][key_col].values


def token2cwsw(tokens, cw_list, sw_list, cw_token='cw', 
               sw_token='sw', none_token='_', both_token='both'):
    """
    The rule here is very flexible.
    """
    cwsw_list = []
    for token in tokens:
        if token in cw_list:
            if token in sw_list:
                cwsw_list.append(both_token)
            else:
                cwsw_list.append(cw_token)
        elif token in sw_list:
            cwsw_list.append(sw_token)
        else:
            cwsw_list.append(none_token)
    return cwsw_list


def text2pattern(row, pattern_templates, token_column='tokenized_text', cwsw_column='cwsw_text', 
                  cw_token='cw', sw_token='sw', none_token='_', both_token='both'):
    pattern_list = []
    pattern_content_list = []
    cwsw_text = row[cwsw_column]
    tokenized_text = row[token_column]
    pattern_lens = list(set([len(pattern_template) for pattern_template in pattern_templates]))
    for idx in range(len(tokenized_text)):
        for target_pattern in pattern_templates:
            target_pattern_len = len(target_pattern)
            if idx + target_pattern_len >= len(tokenized_text):
                continue
            patterns = pattern_match(cwsw=cwsw_text[idx:idx+target_pattern_len], 
                                     token=tokenized_text[idx:idx+target_pattern_len], 
                                     cw_token=cw_token, sw_token=sw_token, none_token=none_token,
                                     target_pattern=target_pattern, both_token=both_token)
            if patterns:
                for pattern in patterns:
                    pattern_list.append(pattern)
                    pattern_content_list.append(tokenized_text[idx:idx+target_pattern_len])
    return pattern_list, pattern_content_list

def replace_both(token_list, cw_token, sw_token, both_token):
    rep_token_list = [token_list.copy() for _ in range(2)]
    both_idx = [idx for idx in range(len(token_list)) if token_list[idx]==both_token][0]
    rep_token_list[0][both_idx], rep_token_list[1][both_idx] = cw_token, sw_token
    return rep_token_list

def pattern_match(cwsw, token, target_pattern, cw_token, sw_token, none_token, both_token='both'):
    if len(cwsw) != len(target_pattern):
        return False
    # replace both_token into cw, sw token
    both_idx = [idx for idx in range(len(cwsw)) if cwsw[idx]==both_token]
    both_n = len(both_idx)
    both_flatten = [cwsw]
    for _ in range(2**both_n):
        for idx, token_list in enumerate(both_flatten):
            if both_token in token_list:
                both_flatten += replace_both(token_list, cw_token, sw_token, both_token)
                both_flatten.pop(idx)
                break
    patterns = []
    for cwsw_ in both_flatten:
        pattern = []
        for idx in range(len(cwsw_)):
            if cwsw_[idx] != target_pattern[idx]:
                break
            elif cwsw_[idx] == cw_token:
                pattern.append(token[idx])
            elif cwsw_[idx] == sw_token:
                pattern.append('*')
        if len(pattern) == len(cwsw_):
            patterns.append(pattern)
    return patterns


def patternDict(pattern_col=None, workers=-1, **kwargs):
    if 'df' in kwargs:
        print('\n**Warning**: \nyou are using the past (slower) version of `patternDict()`, remember to update your IDEAlib & see the new example on github')
        df, label_list = kwargs.pop('df'), kwargs.pop('label_list')
        pattern_templates, n_jobs = kwargs.pop('pattern_templates'), kwargs.pop('n_jobs')
        return patternDict_past(df=df, label_list=label_list, pattern_templates=pattern_templates, n_jobs=n_jobs)
    if type(pattern_col) == np.ndarray:
        pattern_array = pattern_col
    elif type(pattern_col) == pd.Series:
        pattern_array = pattern_col.values
    else:
        print('error')
        return dict()
    ## TODO !

def patternDF(pattern_dict, label_list):
    df_pattern = pd.DataFrame()
    pattern_list = list(pattern_dict.keys())
    df_pattern['pattern'] = pattern_list
    col_list = ['template'] + label_list + ['contents']
    temp_list = [[] for _ in range(len(col_list))]
    for pattern in pattern_list:
        for i, col in enumerate(col_list):
            temp_list[i].append(pattern_dict[pattern][col])
    for i, col in enumerate(col_list):
        df_pattern[col] = temp_list[i]
    return df_pattern


## ------------------------------------------------------------------------ ##

def list_match(l1, l2):
    if len(l1) != len(l2):
        return False
    for idx in range(len(l1)):
        if l1[idx] != l2[idx]:
            return False
    return True



def patternDict_past(df, label_list, pattern_templates, label_col='emotion', n_jobs=-1,
                     token_column='tokenized_text', cwsw_column='cwsw_text'):

#     if n_jobs == 1:
#         ## no mp
#         pattern_dict = dict()
#         patternDict_(df=df, pattern_dict=pattern_dict, label_list=label_list, 
#                      pattern_templates=pattern_templates, label_col=label_col,
#                      token_column=token_column, cwsw_column=cwsw_column
#                     )
#         print(len(pattern_dict))
#     else:
    
    if n_jobs == -1:
        cores = multiprocessing.cpu_count() 
    else:
        cores = int(n_jobs)

    # a memory-shared dictionary `pattern_dict`
    manager = multiprocessing.Manager()
    pattern_dict = manager.dict()

    # build pattern-dictionary
    utils.apply_by_multiprocessing(df, patternDict_, 
                                   pattern_dict=pattern_dict, label_list=label_list, 
                                   pattern_templates=pattern_templates, 
                                   label_col=label_col, workers=cores,
                                   token_column=token_column, cwsw_column=cwsw_column,
                                   axis=1
                                   )
    return pattern_dict



def patternDict_(df, pattern_dict, label_list, pattern_templates, 
                 label_col='emotion', token_column='tokenized_text', cwsw_column='cwsw_text'):

    cwsw_text = df[cwsw_column]
    tokenized_text = df[token_column]
    label = df[label_col]
    for target_pattern in pattern_templates:
                
        cwsw_ngrams = list(nltk.ngrams(cwsw_text, len(target_pattern))) 
        token_ngrams = list(nltk.ngrams(tokenized_text, len(target_pattern))) 
        ## pattern rules defined here
        ## (I don't consider the `both` token here)
        for idx, cwsw_gram in enumerate(cwsw_ngrams):
            
            # match `cwsw_gram` with `pattern_template` (target_pattern)
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
                    default_dict['contents'] = []
                    for label_ in label_list:
                        default_dict[label_] = 0
                    pattern_dict[pattern_get] = default_dict

                temp_dict = pattern_dict[pattern_get]
                temp_dict[label] += 1
                temp_dict['contents'].append(token_ngram)
                pattern_dict[pattern_get] = temp_dict


# def text2pattern(df, pattern_templates, token_column='tokenized_text', cwsw_column='cwsw_text', 
#                  cw_token='cw', sw_token='sw', none_token='_', both_token='both',
#                  pattern_column='pattern', n_jobs=-1):
#     if n_jobs == -1:
#         cores = multiprocessing.cpu_count() 
#     else:
#         cores = int(n_jobs)

#     df[pattern_column] = utils.apply_by_multiprocessing(df, text2pattern_, 
#                                                         pattern_templates=pattern_templates,
#                                                         token_column=token_column, cwsw_column=cwsw_column,
#                                                         cw_token='cw', sw_token='sw', none_token='_', both_token='both',
#                                                         workers=cores, axis=1)
#     return df



def weight_pattern(df, label_list=['anger','sadness'], pattern_contents='contents',
                   weight='pfief', diversity=True, nums_threh=2):
    
    if weight =='pfief':
        df['total'] = df.loc[:,label_list].sum(axis=1)
        df = df[df.total > nums_threh]

        pf = df[label_list].div(df.total, axis=0)
        label_exist = df[label_list].apply(lambda n: n>0)
        ief = label_exist.div(label_exist.sum(axis=1), axis=0)

        pfief = pf.mul(ief)

        if diversity:
            divrsty = np.log(df.contents.apply(lambda contents: len(set(contents))))
            pfief = pfief.mul(divrsty, axis=0)

        return df.join(pfief, rsuffix='_pfief')
    else:
        print('\nThere is no other weighting yet ^^"\n')
    