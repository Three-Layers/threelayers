import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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


class MICEImputer():
    """
    Attributes:
    - estimator: The estimator used to predict missing values. Default is LinearRegression.
    - mice_imputer: The IterativeImputer object that performs the MICE imputation. 
    """
    estimator = LinearRegression()
    
    mice_imputer = IterativeImputer(estimator = estimator,
                           missing_values = np.nan,
                           max_iter = 5,
                           initial_strategy = 'mean',
                           imputation_order = 'ascending',
                           verbose = 1,
                           random_state = 42)


class Imputer():
    """
        This is a class for imputing missing values in a pandas DataFrame using two different methods: 
        iterative imputation and aggregation. 
        The class takes in a training DataFrame, a test DataFrame, and a list of columns with missing values.
    """
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, missing_cols: list[str]):
        self.train_df= train_df
        self.test_df = test_df
        self.all_df = pd.concat([train_df, test_df])
        self.missing_cols = missing_cols

    def __split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_data = data.iloc[:self.train_df.shape[0], :]
        test_data = data.iloc[self.train_df.shape[0]:, :]
        return train_data, test_data

    def iterative_imputer(self, used_cols: list[str]=None, drop_cols: list[str]=None,
                          mice_imputer: IterativeImputer=MICEImputer.mice_imputer) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Imputes missing values in the dataset using an iterative imputation method.

        Args:
        - used_cols (list[str], optional): List of column names to use for imputation. If provided, only these columns
          will be used for imputation. Defaults to None.
        - drop_cols (list[str], optional): List of column names to drop from the dataset before imputation. If provided,
          these columns will not be used for imputation. Defaults to None.
        - mice_imputer (IterativeImputer, optional): An instance of the IterativeImputer class used for imputation.
          Defaults to MICEImputer.mice_imputer.

        Returns:
        - Tuple of two pandas DataFrames

        Raises:
        - TypeError: If both the 'used_cols' and 'drop_cols' parameters are None.
        """
        
        all_df = self.all_df.copy()
        if used_cols:
            df = all_df[used_cols]
        elif drop_cols:
            df = all_df.drop(drop_cols, axis=1)
        elif not used_cols and not drop_cols:
            raise TypeError("'used_cols' or 'drop_cols' parameters cannot be None.")
        
        mice_imputed_df = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)
        for col in self.missing_cols:
            all_df[col] = mice_imputed_df[col]

        return self.__split_data(all_df)
    
    def agg_imputer(self, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Imputes missing values in the dataset using an aggregation method.

        Args:
        - method (str): The imputation method to use. Must be one of "mean", "median", or "mode".

        Returns:
        - Tuple of two pandas DataFrames
        """

        all_df = self.all_df.copy()
        for col in self.missing_cols:
            if method == "mean":
                all_df[col] = all_df[col].fillna(all_df[col].mean())
            elif method == "median":
                all_df[col] = all_df[col].fillna(all_df[col].median())
            elif method == "mode":
                all_df[col] = all_df[col].fillna(all_df[col].mode())

        return self.__split_data(all_df)
