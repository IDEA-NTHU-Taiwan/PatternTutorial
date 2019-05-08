import multiprocessing
import pandas as pd
import numpy as np
import re


def tweet_replaceToken(tweet):
    tweet = re.sub('((https?:\/\/)|(pic.twitter))\S+',
                   'URLTOK', tweet.lower().strip())  # url
    tweet = re.sub('@(?:[a-zA-Z0-9_]+)', '<M>', tweet)  # mention
    tweet = re.sub('#(?:[a-zA-Z0-9_]+)', '<H>', tweet)  # hashtag
    tweet = tweet.replace('\n', " ")
    return tweet


def dict2df(dict_, key_col='token', val_col='value'):
    df = pd.DataFrame()
    df[key_col] = list(dict_.keys())
    df[val_col] = list(dict_.values())
    df.sort_values(by=val_col, axis=0, ascending=False, inplace=True)
    return df


# multiprocessing
def pD_apply_df(args):
    (df,
     func,
     pattern_dict,
     label_list,
     pattern_templates,
     label_col,
     token_column,
     cwsw_column) = args

    return df.apply(lambda x: func(x, pattern_dict, label_list,
                                   pattern_templates, label_col,
                                   token_column, cwsw_column),
                    axis=1)


def patternDict_multiprocessing(df, func, pattern_dict, label_list,
                                pattern_templates, label_col, **kwargs):
    workers = kwargs.pop('workers')
    token_column = kwargs.pop('token_column')
    cwsw_column = kwargs.pop('cwsw_column')
    pool = multiprocessing.Pool(processes=workers)
    pool.map(pD_apply_df, [(d, func, pattern_dict,
                            label_list, pattern_templates,
                            label_col, token_column, cwsw_column)

                           for d in np.array_split(df, workers)
                           ])
    pool.close()
    # return pd.concat(list(result))


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    if workers == -1:
        workers = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
                                  for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))
