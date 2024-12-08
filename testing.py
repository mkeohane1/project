import unittest
import pandas as pd
import numpy as np
from cleaning import fill_missing_values, remove_outliers_iqr, encode_categorical, preprocess_data

class TestDataCleaning(unittest.TestCase):

    def test_fill_missing_values(self):
        # Create a sample DataFrame with missing values
        df = pd.DataFrame({
            'numeric_col': [1, np.nan, 3, np.nan, 5],
            'categorical_col': ['A', 'B', None, 'B', 'A']
        })

        # Fill missing values
        filled_df = fill_missing_values(df, numerical_strategy='mean', categorical_strategy='most_frequent')

        # Check that no missing values remain
        self.assertFalse(filled_df.isnull().values.any())
        # Check numeric column imputation result
        # Mean of [1,3,5] = 3, so missing should be replaced with 3
        self.assertEqual(filled_df['numeric_col'].iloc[1], 3)
        self.assertEqual(filled_df['numeric_col'].iloc[3], 3)
        # Check categorical column imputation result
        # Most frequent in ['A','B','B','A'] is 'A' and 'B' appear twice each.
        # SimpleImputer with 'most_frequent' will pick one of them (depends on sorting/order).
        # We can just assert that it's not None
        self.assertIn(filled_df['categorical_col'].iloc[2], ['A', 'B'])

    def test_remove_outliers_iqr(self):
        # Create a DataFrame with obvious outliers
        df = pd.DataFrame({
            'values': [10, 12, 11, 1000, 13, 9, 10]
        })

        # Remove outliers based on IQR
        cleaned_df = remove_outliers_iqr(df, columns=['values'], multiplier=1.5)

        # Check that the outlier (1000) is removed
        self.assertNotIn(1000, cleaned_df['values'].values)
        # Check we still have the majority of normal values
        self.assertTrue(len(cleaned_df) < len(df))
        self.assertTrue(all(v in [9,10,10,11,12,13] for v in cleaned_df['values'].values))

    def test_encode_categorical_one_hot(self):
        # Create a DataFrame with categorical columns
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'green'],
            'size': ['S', 'M', 'M', 'L']
        })

        # One-hot encode
        encoded_df = encode_categorical(df, method='one-hot')

        # Check that original categorical columns are removed
        self.assertTrue('color' not in encoded_df.columns)
        self.assertTrue('size' not in encoded_df.columns)

        # Check that one-hot columns exist
        self.assertTrue(any(col.startswith('color_') for col in encoded_df.columns))
        self.assertTrue(any(col.startswith('size_') for col in encoded_df.columns))

    def test_encode_categorical_label(self):
        # Create a DataFrame with categorical columns
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'green']
        })

        # Label encode
        encoded_df = encode_categorical(df, method='label')

        # Check that original column still exists and is now numeric
        self.assertIn('color', encoded_df.columns)
        self.assertTrue(np.issubdtype(encoded_df['color'].dtype, np.integer))

    def test_preprocess_data(self):
        # Create a DataFrame with missing values, outliers, and categorical columns
        df = pd.DataFrame({
            'numeric_col': [1, np.nan, 3, 1000, 5],
            'cat_col': ['A', 'B', None, 'B', 'A'],
            'another_cat': ['X', 'Y', 'X', 'Z', 'X']
        })

        # Run the full preprocessing pipeline
        preprocessed_df = preprocess_data(
            df,
            numeric_strategy='mean',
            categorical_strategy='most_frequent',
            outlier_method='iqr',
            encode_method='one-hot'
        )

        # Check no missing values remain
        self.assertFalse(preprocessed_df.isnull().values.any())

        # Check outlier removal: 1000 should be gone
        self.assertNotIn(1000, preprocessed_df['numeric_col'].values)

        # Check that categorical columns have been one-hot encoded
        self.assertTrue(any(col.startswith('cat_col_') for col in preprocessed_df.columns))
        self.assertTrue(any(col.startswith('another_cat_') for col in preprocessed_df.columns))
        
        # Check that numeric_col still exists and is numeric
        self.assertIn('numeric_col', preprocessed_df.columns)
        self.assertTrue(np.issubdtype(preprocessed_df['numeric_col'].dtype, np.number))


if __name__ == '__main__':
    unittest.main()
