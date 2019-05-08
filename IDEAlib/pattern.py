import IDEAlib.utils as utils
import multiprocessing
import pandas as pd
import nltk

"""
TODO:
1. update the `patternDict` multiprocessing method


"""


def listcwsw(df, threshold, key_col='token', val_col='value'):
    """
    return the `np.appry` type of CW/SW list
    """
    return df[df[val_col] >= threshold][key_col].values


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


def list_match(l1, l2):
    if len(l1) != len(l2):
        return False
    for item_at_l1, item_at_l2 in zip(l1, l2):
        if item_at_l1 != item_at_l2:
            return False
    return True


def patternDict(df, label_list, pattern_templates,
                label_col='emotion', n_jobs=-1,
                token_column='tokenized_text',
                cwsw_column='cwsw_text'):

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
    utils.patternDict_multiprocessing(df, patternDict_,
                                      pattern_dict=pattern_dict, label_list=label_list,
                                      pattern_templates=pattern_templates,
                                      label_col=label_col, workers=cores,
                                      token_column=token_column, cwsw_column=cwsw_column
                                      )
    return pattern_dict


def patternDict_(df, pattern_dict, label_list, pattern_templates,
                 label_col='emotion',
                 token_column='tokenized_text',
                 cwsw_column='cwsw_text'):

    cwsw_text = df[cwsw_column]
    tokenized_text = df[token_column]
    label = df[label_col]
    for target_pattern in pattern_templates:

        cwsw_ngrams = list(nltk.ngrams(cwsw_text, len(target_pattern)))
        token_ngrams = list(nltk.ngrams(tokenized_text, len(target_pattern)))
        # pattern rules defined here
        # (I don't consider the `both` token here)
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

                # be careful of updating the `pattern_dict`
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
