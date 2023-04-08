import pandas as pd
import numpy as np

def conf_dtype(df:pd.DataFrame, test:pd.DataFrame, return_dtypes_list: bool = False):
    """
        convert dtypes to time, int (discrete), float (continu), and categorical (others)
        return: converted dtypes
            df, test
            if return_dtypes_list: return list of columns that has these types
                return df, test, ([time], [continu], [discret], [categorical])
    """
    _df, _test = df.copy(), test.copy()
    time, continu, discret, categorical = [], [], [], []
    for col in set(_df.columns) & set(_test.columns):
        if isinstance(df[col], np.datetime64):
            time.append(col)
            continue
        try:
            if all(_df[col].map(lambda x: int(x) == x)) and all(_test[col].map(lambda x: int(x) == x)):
                _df[col], _test[col] = _df[col].astype(np.int32), _test[col].astype(np.int32)
                discret.append(col)
                continue

            all(_df[col].map(lambda x: float(x))) and all(_test[col].map(lambda x: float(x)))
            _df[col], _test[col] = _df[col].astype(np.float32), _test[col].astype(np.float32)
            continu.append(col)
            continue

        except ValueError:
            _df[col], _test[col] = _df[col].astype('category'), _test[col].astype('category')
            categorical.append(col)
    if return_dtypes_list:
        return _df, _test, (time, continu, discret, categorical)
    return _df, _test
