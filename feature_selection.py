import pandas as pd
import statsmodels.api as sm

def p_values(X_train, y_train):
    """
    Calculate the p-values for each feature in the dataset
    using OLS regression.

    X_train: Features (input data)
    y_train: Target variable

    Returns: p-values for each feature
    """
    # Add a constant (intercept) to the model
    X_train = sm.add_constant(X_train)
    
    # Fit the OLS model
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    
    # Get p-values for each feature
    p_values = results.pvalues
    return p_values


def backward_elimination(X_train, y_train, threshold=0.05):
    """
    Perform backward elimination to remove features with p-values greater than a threshold.

    X_train: Features (input data)
    y_train: Target variable
    threshold: p-value threshold for feature removal (default is 0.05)

    Returns: Reduced feature set
    """
    # Add a constant (intercept) to the model
    X_train = sm.add_constant(X_train)
    
    # Fit the OLS model
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    
    # Get p-values of the features
    p_values = results.pvalues
    
    # Loop to remove features with p-values above the threshold
    while max(p_values) > threshold:
        # Get the feature with the highest p-value
        max_p_value_feature = p_values.idxmax()
        
        # Drop the feature with the highest p-value
        X_train = X_train.drop(columns=[max_p_value_feature])
        
        # Fit the model again with the reduced features
        model = sm.OLS(y_train, X_train)
        results = model.fit()
        
        # Get new p-values
        p_values = results.pvalues
    
    # Return the reduced feature set
    return X_train


# Example Usage (Optional, to test the code in isolation):
if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    # Load the California housing dataset (alternative to Boston housing)
    california_housing = fetch_california_housing()

    # Extract features and target
    X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    y = pd.Series(california_housing.target)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Calculate p-values for features
    print("Calculating p-values for features...")
    p_vals = p_values(X_train, y_train)
    print("\nP-values:\n", p_vals)

    # 2. Perform backward elimination
    print("\nPerforming backward elimination...")
    X_train_reduced = backward_elimination(X_train, y_train)
    print("\nReduced features after backward elimination:\n", X_train_reduced.columns)