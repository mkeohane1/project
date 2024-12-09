import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def fill_missing_values(df, numerical_strategy='mean', categorical_strategy='most_frequent', numeric_cols=None, cat_cols=None):
    """
    Fill missing values in numerical and categorical columns using specified strategies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    numerical_strategy : str, default='mean'
        Strategy for numerical columns. Options: 'mean', 'median', 'most_frequent', 'constant'.
    categorical_strategy : str, default='most_frequent'
        Strategy for categorical columns. Options: 'most_frequent', 'constant'.
    numeric_cols : list, optional
        List of column names to treat as numeric. If None, automatically inferred.
    cat_cols : list, optional
        List of column names to treat as categorical. If None, automatically inferred.
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with missing values filled.
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cat_cols is None:
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill numeric
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy=numerical_strategy)
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    # Fill categorical
    if len(cat_cols) > 0:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy, fill_value='missing')
        df[cat_cols] = categorical_imputer.fit_transform(df[cat_cols])

    return df

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers from specified numeric columns using the IQR (Interquartile Range) method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    columns : list, optional
        List of numeric columns to check for outliers. If None, all numeric columns are used.
    multiplier : float, default=1.5
        The IQR multiplier defining what is considered an outlier.
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with outliers removed.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if df[col].dtype.kind in 'bifc':  # numeric types
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df

def encode_categorical(df, columns=None, method='one-hot'):
    """
    Encode categorical variables using one-hot encoding or label encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    columns : list, optional
        List of columns to encode. If None, all non-numeric columns will be encoded.
    method : str, default='one-hot'
        Encoding method. Options: 'one-hot', 'label'.
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with categorical features encoded.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if method == 'one-hot':
        # Use pandas get_dummies for simplicity
        df = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        # Use sklearn's LabelEncoder for each column
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    else:
        raise ValueError("method must be 'one-hot' or 'label'")

    return df

def preprocess_data(df,
                    numeric_strategy='mean',
                    categorical_strategy='most_frequent',
                    outlier_method='iqr',
                    outlier_cols=None,
                    outlier_multiplier=1.5,
                    encode_method='one-hot',
                    encode_cols=None):
    """
    An all-in-one preprocessing function that:
    1. Fills missing values in numerical and categorical columns.
    2. Removes outliers based on IQR.
    3. Encodes categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    numeric_strategy : str
        Strategy for numerical imputation ('mean', 'median', etc.)
    categorical_strategy : str
        Strategy for categorical imputation ('most_frequent', 'constant')
    outlier_method : str, default='iqr'
        Method to remove outliers. Currently only 'iqr' supported.
    outlier_cols : list or None
        Columns to apply outlier removal. If None, applies to all numeric columns.
    outlier_multiplier : float
        IQR multiplier for outlier detection.
    encode_method : str, default='one-hot'
        Encoding method ('one-hot' or 'label').
    encode_cols : list or None
        Columns to encode. If None, encodes all categorical columns.
    
    Returns:
    --------
    pd.DataFrame
        A fully preprocessed DataFrame ready for EDA and modeling.
    """
    # Step 1: Handle missing values
    df = fill_missing_values(df,
                             numerical_strategy=numeric_strategy,
                             categorical_strategy=categorical_strategy)

    # Step 2: Remove outliers
    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df, columns=outlier_cols, multiplier=outlier_multiplier)
    else:
        raise ValueError("Currently only 'iqr' outlier method is supported.")

    # Step 3: Encode categorical variables
    df = encode_categorical(df, columns=encode_cols, method=encode_method)

    return df