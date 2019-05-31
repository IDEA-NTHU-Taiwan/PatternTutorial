import IDEAlib.utils as utils
from functools import partial
import multiprocessing
import pandas as pd
import numpy as np
import nltk

"""
TODO:
1. update the `patternDict` multiprocessing method


"""


class PatternTree():
    def __init__(self, df_pattern_weight, pattern_column, weight_columns):
        self.pattern_tree = {'sub_tree':{}}
        self.pattern_vectors = np.zeros((df_pattern_weight.shape[0], len(weight_columns)))
        self.id2pattern = []
        self.pattern_max_len = max([len(pattern.split())
                                    for pattern in df_pattern_weight[pattern_column].tolist()]
                                  )
        self.build_tree(df_pattern_weight, pattern_column, weight_columns)
    
    def build_tree(self, df_pattern_weight, pattern_column, weight_columns):
        for i, (pattern, vector) in enumerate(zip(df_pattern_weight[pattern_column],
                                              df_pattern_weight[weight_columns].values)
                                             ):
            token_pattern = pattern.split()
            token_len = len(token_pattern)
            sub_tree = self.pattern_tree.get('sub_tree')
            for w_i in range(token_len-1):
                if sub_tree.get(token_pattern[w_i]) == None:
                    sub_tree[token_pattern[w_i]] = {'pattern_idx':-1, 'sub_tree':{}}
                sub_tree = sub_tree.get(token_pattern[w_i]).get('sub_tree')
            # last token of pattern, insert pattern idx
            if sub_tree.get(token_pattern[-1]) == None:
                sub_tree[token_pattern[-1]] = {'pattern_idx':i, 'sub_tree':{}}
            else:
                sub_tree[token_pattern[-1]]['pattern_idx'] =i
            self.id2pattern.append(pattern)
            self.pattern_vectors[i] = vector   
        print('\nPattern tree has been built.', '\nWith totally {} patterns.\n'.format(i+1))
    
    def get_pattern_by_id(self, pid):
        return self.id2pattern[pid]

    def search_sub_tree(self, tokens, pat_tree):
        patterns = []
        len_tokens = len(tokens)
        token = tokens[0] # check first word first
        if token in pat_tree: # if token in token tree, it is a cw. search for following possible template
            if  pat_tree.get(token).get('pattern_idx')!= -1:
                patterns.append(pat_tree.get(token).get('pattern_idx'))
            if len_tokens >1: # if not end, search next
                sub_tree = pat_tree.get(token).get('sub_tree')
                patterns += self.search_sub_tree(tokens[1:], sub_tree)
        if '*' in pat_tree: # it is a '*'
            token = '*'
            if  pat_tree.get(token).get('pattern_idx')!= -1:
                patterns.append(pat_tree.get(token).get('pattern_idx'))
            sub_tree = pat_tree.get(token).get('sub_tree')
            if len_tokens >1: # if not end, search next
                patterns += self.search_sub_tree(tokens[1:], sub_tree)
        return patterns
    
    def tree_match(self, text_token):
        patterns = []
        text_len = len(text_token)
        # search pattern
        for token_i in range(text_len-self.pattern_max_len+1): # get pivot
            tokens = text_token[token_i: token_i+self.pattern_max_len]
            patterns += self.search_sub_tree(tokens, self.pattern_tree.get('sub_tree'))
        return patterns



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
    pattern_template_list = []
    cwsw_text = row[cwsw_column]
    tokenized_text = row[token_column]
    pattern_lens = list(set([len(pattern_template) for pattern_template in pattern_templates]))
    for idx in range(len(tokenized_text)):
        for target_pattern in pattern_templates:
            target_pattern_len = len(target_pattern)
            if idx + target_pattern_len > len(tokenized_text):
                continue
            patterns, pattern_temps = pattern_match(cwsw=cwsw_text[idx:idx+target_pattern_len], 
                                                    token=tokenized_text[idx:idx+target_pattern_len], 
                                                    cw_token=cw_token, sw_token=sw_token, none_token=none_token,
                                                    target_pattern=target_pattern, both_token=both_token)
            if patterns:
                for pattern, pattern_temp in zip(patterns, pattern_temps):
                    pattern_list.append(pattern)
                    pattern_content_list.append(tokenized_text[idx:idx+target_pattern_len])
                    pattern_template_list.append(pattern_temp)
    return pattern_list, pattern_content_list, pattern_template_list

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
    pattern_temps = []
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
            pattern_temps.append(cwsw_)
    return patterns, pattern_temps


def patternDict(df, pattern_col=None, pattern_content_col=None, pattern_template_col=None, 
                label_col=None, workers=-1, **kwargs):
    if 'label_list' in kwargs:
        print('\n**Warning**: \nyou are using the past (slower) version of `patternDict()`, remember to update your IDEAlib & see the new example on github')
        label_list = kwargs.pop('label_list')
        pattern_templates, n_jobs = kwargs.pop('pattern_templates'), kwargs.pop('n_jobs')
        return patternDict_past(df=df, label_list=label_list, pattern_templates=pattern_templates, n_jobs=n_jobs)

    col_list = [pattern_col, pattern_content_col, pattern_template_col, label_col]
    df = df[col_list]
    if workers == -1:
        workers = multiprocessing.cpu_count() 
    pool = multiprocessing.Pool(processes=workers)
    func = partial(patternDict_, col_list=col_list)
    part_dicks = pool.map(func, [(d) for d in np.array_split(df, workers)])
    pool.close()

    ## merge dict
    label_unique = np.unique(df[label_col].values)
    print('label_unique: ', label_unique)
    pattern_dict = dict()
    for part_dict in part_dicks:
        for key in part_dict:
            if key not in pattern_dict:
                pattern_dict[key] = part_dict[key].copy()
            else:
                pattern_dict[key]['contents'] += part_dict[key]['contents']
                for label_key in label_unique:
                    pattern_dict[key][label_key] += part_dict[key][label_key]
    return pattern_dict

def patternDict_(df, col_list, str_type=True):
    """
    ** Now using for-loop iter, maybe there are some optimized methods. **
    """
    pattern_col, pattern_content_col, pattern_template_col, label_col = col_list
    # df['emo_n'] = df.apply(lambda row: [row[label_col] for _ in range(len(row[pattern_col]))], axis=1)
    pattern_col = df[pattern_col].values
    pattern_content_col = df[pattern_content_col].values
    pattern_template_col = df[pattern_template_col].values
    label_col = df[label_col].values
    label_unique = np.unique(label_col)
    # print('label_unique: ', label_unique)

    """
    part_dict[`pattern`][`template`(str), `contents`(str in list), `each label`]
    (`pattern` is the key of part_dict)
    """
    part_dict = dict()
    for idx_row, patterns in enumerate(pattern_col):
        label = label_col[idx_row]
        for idx_in, pattern in enumerate(patterns):
            pattern_content = pattern_content_col[idx_row][idx_in]
            pattern_template = pattern_template_col[idx_row][idx_in]
            if str_type:
                pattern = ' '.join(pattern)
                pattern_template = ' '.join(pattern_template)
            if pattern not in part_dict: # init
                part_dict[pattern] = {}
                for each_label in label_unique:
                    part_dict[pattern][each_label] = 0
                part_dict[pattern]['contents'] = []
                part_dict[pattern]['template'] = pattern_template
            part_dict[pattern][label] += 1
            part_dict[pattern]['contents'].append(pattern_content)
    return part_dict



def weight_pattern(df, label_list=['anger','sadness'], pattern_contents='contents',
                   weight='pfief', diversity=True, nums_threshold=2):
    import warnings
    warnings.filterwarnings("ignore")
    if weight =='pfief':
        df['total'] = df.loc[:,label_list].sum(axis=1)
        df = df[df.total > nums_threshold]
        pf = df[label_list].div(df.total, axis=0)
        label_exist = df[label_list].apply(lambda n: n>0)
        ief = label_exist.div(label_exist.sum(axis=1), axis=0)
        pfief = pf.mul(ief)
        if diversity:
            df[pattern_contents] = df[pattern_contents].apply(lambda x: [tuple(t) for t in x])
            divrsty = np.log(df[pattern_contents].apply(lambda contents: len(set(contents))))
            pfief = pfief.mul(divrsty, axis=0)
        return df.join(pfief, rsuffix='_pfief')
    else:
        print('\nThere is no other weighting yet ^^"\n')

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

def pattern_check(row, pattern_col, pattern_df):
    pattern_list = row[pattern_col]
    # if type(pattern_list[0]) is list:
    #     pattern_list = [' '.join(x) for x in pattern_list]
    pattern_list = [' '.join(x) for x in pattern_list]
    pattern_pool = pattern_df['pattern'].values
    check_list = []
    for pattern in pattern_list:
        if pattern in pattern_pool:
            check_list.append(True)
        else:
            check_list.append(False)
    return check_list

def pattern_filter(row, pattern_col, patt_check_col):
    pattern = row[pattern_col]
    patt_check = row[patt_check_col]
    pattern_final = [patt for i, patt in enumerate(pattern) if patt_check[i]]
    return pattern_final



## ------------------------------------------------------------------------ ##
##    past function    ##
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
    utils.apply_by_multiprocessing(df, patternDict_past_, 
                                   pattern_dict=pattern_dict, label_list=label_list, 
                                   pattern_templates=pattern_templates, 
                                   label_col=label_col, workers=cores,
                                   token_column=token_column, cwsw_column=cwsw_column,
                                   axis=1
                                   )
    return pattern_dict



def patternDict_past_(df, pattern_dict, label_list, pattern_templates, 
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



