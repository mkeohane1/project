from eda import calculate_mean, calculate_variance
from model import fit, predict

def calculate_mse(y, predicted_y):
	"""
	Measures the average of the squares of the errors between
	estimated and actual values (mean squared error).
	Parameters:
		y: an iterable of actual y values
		predicted_y: an iterable of predicted y values
	Returns: the mean squared error as a float
	"""

	if len(y) != len(predicted_y):
		raise ValueError("inputs must be same length")
	if len(y) == 0 or len(predicted_y) == 0:
		raise ValueError("inputs cannot be empty")

	squared_errors = [(actual - predicted) ** 2 for actual, predicted\
				   in zip(y, predicted_y)]

	mse = sum(squared_errors) / len(y)

	return mse


def calculate_r_squared(y, predicted_y):
	"""
	Calculates the coefficient of determination (R squared) for
	the proportion of the variation in the dependent variable
	from the independent variable.
	Parameters:
		y: an iterable of actual y values
		predicted_y: an iterable of predicted y values
	Returns: r squared as a float
	"""

	if len(y) != len(predicted_y):
		raise ValueError("inputs must be same length")

	total_variance = calculate_variance(y)
	explained_variance = calculate_variance(predicted_y)

	r_squared = explained_variance / total_variance
	
	return r_squared
	