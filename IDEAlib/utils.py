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


## multiprocessing (could be more pretty QQ)
def _apply_df(args):
    df, func, kwargs = args
    if 'axis' in kwargs:
        axis = kwargs.pop('axis')
        return df.apply(func, **kwargs, axis=axis)
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    if workers == -1:
        workers = multiprocessing.cpu_count() 
    coln = 1    
    if 'coln' in kwargs:
        coln = kwargs.pop('coln')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    series = pd.concat(list(result))

    ## TODO: make here beautiful
    if coln == 1:
        return series
    elif coln == 2:
        series_0, series_1 = series.apply(lambda x: x[0]), series.apply(lambda x: x[1])
        return series_0, series_1
    elif coln == 3:
        series_0, series_1, series_2 = series.apply(lambda x: x[0]), series.apply(lambda x: x[1]), series.apply(lambda x: x[2])
        return series_0, series_1, series_2    













