import pandas as pd
import numpy as np
from cleaning import conf_dtype
from sklearn.model_selection import KFold, StratifiedKFold

# NOT TESTED
def validate_FE_train_test(train: pd.DataFrame, test: pd.DataFrame, target: str):
    # assert time == [] and cat == []
    _, _, (time, continu, discrete, cat) = conf_dtype(train, test, True)
    if time != []:
        print('='*20)
        print('Non number features : ', time)
        print('dtypes : time')
    if cat != []:
        print('='*20)
        print('Non number features : ', cat)
        print('dtypes : category')
    
    # assert (set(train.columns) ^ set(test.columns)) == [target]
    if (set(train.columns) ^ set(test.columns)) == [target]:
        print('='*20)
        print('not in train : ', set(test.columns)-set(train.columns))
        print('not in test : ', set(train.columns)-set(test.columns))


def kfold_ml(estimator, X: pd.DataFrame, y: pd.Series, loss_fn: function, n_folds: int=5, 
             normal_kfold: bool=True, not_eval: bool=False, shuffle: bool=False) -> tuple[float, pd.Series]:
    """
    Perform K-Fold Cross Validation on a given estimator and return the mean of the loss function over all folds,
    along with a pandas Series containing the full predictions.

    Parameters:
    -----------
    estimator : estimator object
        A scikit-learn compatible estimator.

    X : pandas DataFrame
        The input features for the model.

    y : pandas Series
        The target variable for the model.

    loss_fn : function
        The loss function to use for evaluation.

    n_folds : int, optional
        The number of folds to use for cross validation. Default is 5.

    not_eval : bool, optional
        If True, the estimator will be fit on the entire training set and not evaluated. Default is False.

    shuffle : bool, optional
        Whether or not to shuffle the data before splitting into folds. Default is False.

    Returns:
    --------
    mean_loss : float
        The mean loss over all folds.

    full_pred : pandas Series
        A pandas Series containing the full predictions.
    """
    
    full_pred = pd.Series([x for x in range(len(y))])
    if normal_kfold:
        kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=42)
    else:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=42)
    pred_val, actual_val, lst_loss = [], [], []
    k = 1

    for train_idx, val_idx in kfold.split(X, y):
        X_train = X.iloc[train_idx,:]
        X_val = X.iloc[val_idx,:]
        y_train = y.iloc[train_idx,:]
        y_val = y.iloc[val_idx,:]
        
        model = estimator
        if not_eval:
            model = model.fit(X_train, y_train)
        else:
            model = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        pred = model.predict(X_val)
        full_pred.loc[X_val.index] = pred
        pred_val.append(pred)
        actual_val.append(y_val.values)

        loss = np.sqrt(loss_fn(y_val, pred))
        lst_loss.append(loss)
        print(f"Fold {k}: {loss:5f}")
        k += 1

    print(f"Mean KFold: {np.mean(lst_loss):5f}")
    return np.mean(lst_loss), full_pred
