from eda import *

def fit(x, y):
	"""
	Uses the ordinary least squares (OLS) method to find line of best fit.
	Parameters:
		x: an iterable of x values
		y: an iterable of y values
	Returns: 
	"""

	x_mean = calculate_mean(x)
	y_mean = calculate_mean(y)

	x_difference = [value - x_mean for value in x]
	y_difference = [value - y_mean for value in y]

	squares = [difference ** 2 for difference in x_difference]

	product = [x_diff * y_diff for x_diff, y_diff in\
			zip(x_difference, y_difference)]

	slope = sum(product) / sum(squares)

	y_intercept = y_mean - (slope * x_mean)

	print(f"{slope}, {y_intercept}")
	return (slope, y_intercept)