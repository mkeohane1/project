import numpy as np

def calculate_mean(x):
	"""
	Calculates the arithmetic mean of an iterable x.
	Parameters:
		x: an iterable of numerical values
	Returns: the mean of the input x as a float
	"""

	if len(x) == 0:
		raise ValueError("Empty dataset")
	for i in x:
		if not isinstance(i, (int, float)):
			raise TypeError("All elements must be numerical values")

	# print(np.mean(x))
	# print(np.mean(y))
	return np.mean(x)


def calculate_median(x):
	"""
	Calculates and returns the median value of an iterable x.
	Parameters:
		x: an iterable of numerical values
	Returns: the median of the input x
	"""

	if len(x) == 0:
		raise ValueError("Empty dataset")
	for i in x:
		if not isinstance(i, (int, float)):
			raise TypeError("All elements must be numerical values")

	# print(np.median(x))
	# print(np.median(y))
	return np.median(x)


def calculate_variance(x):
	"""
	Calculates the variance of an iterable x.
	Parameters:
		x: an iterable of numerical values
	Returns: the variance of the input x as a float
	"""

	if len(x) == 0:
		raise ValueError("Empty dataset")
	for i in x:
		if not isinstance(i, (int, float)):
			raise TypeError("All elements must be numerical values")

	# print(np.var(x))
	# print(np.var(y))
	return np.var(x)


def calculate_std(x):
	"""
	Calculates the standard deviation of an iterable x.
	Parameters:
		x: an iterable of numerical values
	Returns: the standard deviation of the input x
	"""

	if len(x) == 0:
		raise ValueError("Empty dataset")
	for i in x:
		if not isinstance(i, (int, float)):
			raise TypeError("All elements must be numerical values")

	# print(np.std(x))
	# print(np.std(y))
	return np.std(x)


