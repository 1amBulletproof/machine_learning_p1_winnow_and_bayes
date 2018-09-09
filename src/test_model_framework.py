#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Test Framework for machine learning algorithms

import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator
from winnow_2 import Winnow2

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to run tests')
	parser = argparse.ArgumentParser(description='Learn & Verify machine learning algorithms')
	parser.add_argument('file_path', type=str, help='full path to input file')
	parser.add_argument('fraction', type=float, default=0.66, nargs='?', help='fraction of data to learn')
	args = parser.parse_args()

	file_path = args.file_path
	input_data = FileManager.get_csv_file_data_array(file_path)

	fraction_of_data_for_learning = args.fraction
	data_sets = DataManipulator.split_data_in_2_randomly(input_data, fraction_of_data_for_learning)

	learning_data = data_sets[0]
	#print('learning data size: ', len(learning_data))
	test_data  = data_sets[1]
	#print('test data size: ', len(test_data))

	#@TODO: here simply "run" a test_runner given the input data & test data?
	winnow2 = Winnow2() # default values for alpha, threshold, & start weight

	#Learn the model on learning data
	winnow2_learned_weights = winnow2.learn_winnow2_model(learning_data)
	print('learned weights:')
	print(winnow2_learned_weights)

	#Test the model on test data
	print()
	print('Testing the model')
	winnow2_test_results = winnow2.test_winnow2_model(test_data) #Should get this right since it's the training data!

	#Output final weights & results (possibly graph some stuff)
	print('classification attempts(', winnow2_test_results[0], '), \
fails(', winnow2_test_results[1], '), \
success(' , winnow2_test_results[2], ')')

	#@TODO: use NaiveBayes to learn the data set
	#@TODO: use NaiveBayes to test model against remaining data
	#@TODO: Output final percents & results (possibly graph some stuff)


if __name__ == '__main__':
	main()
