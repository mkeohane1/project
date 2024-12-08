import unittest
from eda import calculate_mean, calculate_median, calculate_variance, calculate_std
from model import fit, predict
from evaluation import calculate_mse, calculate_r_squared

class TestFunctions(unittest.TestCase):
  
	def test_calculate_mean(self):
		
		# test with typical integers
		x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		self.assertEqual(calculate_mean(x), 5.5)
		
		# test with negative values
		x = [-3, -2, -1, 0, 1, 2, 3]
		self.assertEqual(calculate_mean(x), 0)

		# test with empty list
		x = []
		with self.assertRaises(ValueError):
			calculate_mean(x)
		
		# test with non-numeric values
		x = ["a", "b", "c", "d"]
		with self.assertRaises(TypeError):
			calculate_mean(x)
		
	def test_calculate_median(self):
		 
		# test with typical integers
		x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		self.assertEqual(calculate_median(x), 5.5)
		 
		# test with negative values
		x = [-3, -2, -1, 0, 1, 2, 3]
		self.assertEqual(calculate_median(x), 0)
		 
		# test with empty list
		x = []
		with self.assertRaises(ValueError):
			calculate_median(x)
		 
		# test with non-numeric values
		x = ["a", "b", "c", "d"]
		with self.assertRaises(TypeError):
			calculate_median(x)
			
	def test_calculate_variance(self):
		 
		# test with typical integers
		x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		expected = sum((xi - 5.5) ** 2 for xi in x) / len(x)
		self.assertEqual(calculate_variance(x), expected)
		 
		# test with negative values
		x = [-3, -2, -1, 0, 1, 2, 3]
		expected = sum((xi - 0) ** 2 for xi in x) / len(x)
		self.assertEqual(calculate_variance(x), expected)
		  
		# test with empty list
		x = []
		with self.assertRaises(ValueError):
			calculate_variance(x)
			   
		# test with non-numeric values
		x = ["a", "b", "c", "d"]
		with self.assertRaises(TypeError):
			calculate_variance(x)
		
	def test_calculate_std(self):

		# test with typical integers
		x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		expected = (sum((xi - 5.5) ** 2 for xi in x) / len(x)) ** 0.5
		self.assertEqual(calculate_std(x), expected)

		# test with negative values
		x = [-3, -2, -1, 0, 1, 2, 3]
		expected = (sum((xi - 0) ** 2 for xi in x) / len(x)) ** 0.5
		self.assertEqual(calculate_std(x), expected)

		# test with empty list
		x = []
		with self.assertRaises(ValueError):
			calculate_std(x)

		# test with non-numeric values
		x = ["a", "b", "c", "d"]
		with self.assertRaises(TypeError):
			calculate_std(x)
		
	def test_fit_typical(self):
		
		# test with typical x and y values
		x = [1, 2, 3, 4, 5]
		y = [2, 4, 6, 8, 10]
		expected_slope = 2.0
		expected_y_intercept = 0.0
		self.assertEqual(fit(x, y), (expected_slope, expected_y_intercept))
		
	def test_fit_negative(self):
		  
		# test with negative x and y values
		x = [-1, -2, -3, -4, -5]
		y = [-2, -4, -6, -8, -10]
		expected_slope = 2.0
		expected_y_intercept = 0.0
		self.assertEqual(fit(x, y), (expected_slope, expected_y_intercept))
		  
	def test_fit_bad_input(self):
			
		# test with non-numeric data types
		x = ["a", "b", "c"]
		y = [1, 2, 3]
		with self.assertRaises(TypeError):
			fit(x, y)

	def test_predict_typical(self):

		# test with typical data
		x = [1, 2, 3, 4, 5]
		slope = 2.0
		y_intercept = 1.0
		expected_y = [3.0, 5.0, 7.0, 9.0, 11.0]
		self.assertEqual(predict(x, slope, y_intercept), expected_y)

	def test_predict_negative(self):

		# test with negative x and slope values
		x = [-1, -2, -3, -4]
		slope = -2.0
		y_intercept = 3.0
		expected_y = [5.0, 7.0, 9.0, 11.0]
		self.assertEqual(predict(x, slope, y_intercept), expected_y)

	def test_predict_bad_input(self):

		# test with non-numeric x value
		x = ["a", "b", "c", "d"]
		slope = 2.0
		y_intercept = 1.0
		with self.assertRaises(TypeError):
			predict(x, slope, y_intercept)

		# test with non-numeric slope
		x = [1, 2, 3]
		slope = "a"
		y_intercept = 1.0
		with self.assertRaises(TypeError):
			predict(x, slope, y_intercept)

		# test with non-numeric y intercept
		x = [1, 2, 3]
		slope = 2.0
		y_intercept = "a"
		with self.assertRaises(TypeError):
			predict(x, slope, y_intercept)

	def test_mse(self):

		# test with typical data
		y = [3, -0.5, 2, 7]
		predicted_y = [2.5, 0.0, 2, 8]
		expected_mse = sum([(actual - predicted) ** 2 for actual, predicted\
							in zip(y, predicted_y)]) / len(y)
		self.assertEqual(calculate_mse(y, predicted_y), expected_mse)
	
		# test with non-numeric data
		y = [1, 2, 3]
		predicted_y = ["a", "b", "c"]
		with self.assertRaises(TypeError):
			calculate_mse(y, predicted_y)

	def test_rsq(self):

		# test with typical data
		y = [3, -0.5, 2, 7]
		predicted_y = [2.5, 0.0, 2, 8]
		total_variance = calculate_variance(y)
		explained_variance = calculate_variance(predicted_y)
		expected_r_squared = explained_variance / total_variance
		self.assertEqual(calculate_r_squared(y, predicted_y), expected_r_squared)
	
		# test with non-numeric data
		y = [1, 2, 3]
		predicted_y = ["a", "b", "c"]
		with self.assertRaises(TypeError):
			calculate_r_squared(y, predicted_y)


if __name__ == '__main__':
	unittest.main()