import multiprocessing
import pandas as pd
import numpy as np
import re


def tweet_replaceToken(tweet):
    tweet = re.sub('((https?:\/\/)|(pic.twitter))\S+','URLTOK',tweet.lower().strip()) # url
    tweet = re.sub('@(?:[a-zA-Z0-9_]+)', '<M>', tweet) # mention
    tweet = re.sub('#(?:[a-zA-Z0-9_]+)', '<H>', tweet) # hashtag
    tweet = tweet.replace('\n'," ")
    return tweet


def dict2df(dict_, key_col='token', val_col='value'):
    df = pd.DataFrame()
    df[key_col] = list(dict_.keys())
    df[val_col] = list(dict_.values())
    df.sort_values(by=val_col, axis=0, ascending=False, inplace=True)
    return df


def token2CWSWlist(df, threshold, key_col='token', val_col='value'):
    """
    return the `np.appry` type of CW/SW list
    """
    return df[df[val_col]>=threshold][key_col].values


## multiprocessing
def _apply_df(args):
    df, func, pattern_dict, label_list, pattern_templates = args
    return df.apply(lambda x: func(x, pattern_dict, label_list, pattern_templates), axis=1)

def apply_by_multiprocessing(df, func, pattern_dict, label_list, pattern_templates, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    pool.map(_apply_df, [(d, func, pattern_dict, label_list, pattern_templates) for d in np.array_split(df, workers)])
    pool.close()
    # return pd.concat(list(result))


















