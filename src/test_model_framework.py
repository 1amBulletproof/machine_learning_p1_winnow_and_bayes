#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Test Framework for machine learning algorithms

import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator
from winnow_2 import Winnow2
from naive_bayes import NaiveBayes

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to run tests')

	parser = argparse.ArgumentParser(description='Learn & Verify machine learning algorithms')
	parser.add_argument('file_path', type=str, help='full path to input file')
	parser.add_argument('fraction', type=float, default=0.66, nargs='?', help='fraction of data to learn')
	args = parser.parse_args()

	#INPUTS
	print()
	print('INPUTS')
	file_path = args.file_path
	print('filepath: ' + file_path)
	fraction_of_data_for_learning = args.fraction
	print('fraction of data to train: ',  fraction_of_data_for_learning)

	#READ INPUT DATA
	input_data = FileManager.get_csv_file_data_array(file_path)
	print('number of input vectors: ', len(input_data))

	#SPLIT INPUT DATA (learning & test sets)
	print()
	data_sets = DataManipulator.split_data_in_2_randomly(input_data, fraction_of_data_for_learning)
	learning_data = data_sets[0]
	print('learning data size: ', len(learning_data))
	test_data  = data_sets[1]
	print('test data size: ', len(test_data))
	print()

	#LEARN THE MODELS

	#Winnow2
	winnow2 = Winnow2() # default values for alpha, threshold, & start weight
	winnow2_learned_weights = winnow2.learn_winnow2_model(learning_data)
	print('Winnow2 learned weights:')
	print(winnow2_learned_weights)
	print()

	#Naive Bayes
	number_of_classes = 2
	naive_bayes = NaiveBayes(number_of_classes)
	naive_bayes_learned_percents = naive_bayes.learn_naive_bayes_model(learning_data)
	print('Naive Bayes learned percentages as input[ class[ (prob0, prob1) ] ]')
	print(naive_bayes_learned_percents)
	print()

	#TEST THE MODELS

	#Winnow2
	print('Testing Winnow2 model')
	winnow2_test_results = winnow2.test_winnow2_model(test_data) #Should get this right since it's the training data!
	print('classification attempts(', winnow2_test_results[0], '), \
#fails(', winnow2_test_results[1], '), \
#success(' , winnow2_test_results[2], ')')
	print()

	#Naive Bayes
	print('Testing Naive Bayes model')
	naive_bayes_test_results = naive_bayes.test_naive_bayes_model(test_data) #Should get this right since it's the training data!
	print('#classification attempts(', naive_bayes_test_results[0], '), \
#fails(', naive_bayes_test_results[1], '), \
#success(' , naive_bayes_test_results[2], ')')
	print()


if __name__ == '__main__':
	main()
