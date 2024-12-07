from eda import *

def fit(x, y):
	"""
	Uses the ordinary least squares (OLS) method to find line of best fit.
	Parameters:
		x: an iterable of x values
		y: an iterable of y values
	Returns: slope and y intercept as floats
	"""

	if not isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
		raise TypeError("x and y must be iterables")
	if len(x) != len(y):
		raise ValueError("x and y must be same length")

	x_mean = calculate_mean(x)
	y_mean = calculate_mean(y)

	x_difference = [value - x_mean for value in x]
	y_difference = [value - y_mean for value in y]

	squares = [difference ** 2 for difference in x_difference]

	product = [x_diff * y_diff for x_diff, y_diff in\
			zip(x_difference, y_difference)]

	slope = sum(product) / sum(squares)

	y_intercept = y_mean - (slope * x_mean)

	# print(f"{slope}, {y_intercept}")
	return (slope, y_intercept)


def predict(x, slope, y_intercept):
	"""
	A function to predict y-values using OLS
	Parameters:
		x: an iterable of x-values
		slope: the slope of the line
		y_intercept: the y-intercept of the line
	Returns: an iterable of predicted y-values
	"""

	if not isinstance(x, (list, tuple)):
		raise TypeError("x must be an iterable")
	if not all(isinstance(i, (int, float)) for i in x):
		raise TypeError("all elements of x must be numeric")
	if not isinstance(slope, (int, float)):
		raise TypeError("slope must be a numeric value")
	if not isinstance(y_intercept, (int, float)):
		raise TypeError("y intercept must be a numeric value")

	predicted_y = [((slope * i) + y_intercept) for i in x]

	print(predicted_y)
	return predicted_y


