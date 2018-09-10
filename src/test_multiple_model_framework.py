#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Test Framework intended for iris data set but can be extended for using multiple models to classify multiple classes (Winnow2 requires multiple models for multiple classes)

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
	parser.add_argument('all_class_filepath', type=str, help='full path to input file')
	parser.add_argument('class_0_filepath', type=str, help='full path to input file')
	parser.add_argument('class_1_filepath', type=str, help='full path to input file')
	parser.add_argument('class_2_filepath', type=str, help='full path to input file')
	parser.add_argument('fraction', type=float, default=0.66, nargs='?', help='fraction of data to learn')
	parser.add_argument('num_classes', type=int, default=3, nargs='?', help='number of total classes')
	args = parser.parse_args()

	#INPUTS
	print()
	print('INPUTS')
	allclass_filepath = args.all_class_filepath
	print('overall_model_file_path: ' + allclass_filepath)
	class0_filepath = args.class_0_filepath
	print('class0_filepath: ' + class0_filepath)
	class1_filepath = args.class_1_filepath
	print('class1_filepath: ' + class1_filepath)
	class2_filepath = args.class_2_filepath
	print('class2_filepath: ' + class2_filepath)
	fraction_of_data_for_learning = args.fraction
	print('fraction of data to train: ',  fraction_of_data_for_learning)
	number_of_classes = args.num_classes
	print('number of classes: ', number_of_classes)

	#READ INPUT DATA
	allclass_data = FileManager.get_csv_file_data_array(allclass_filepath)
	print('number of input vectors: ', len(allclass_data))
	class0_data = FileManager.get_csv_file_data_array(class0_filepath)
	class1_data = FileManager.get_csv_file_data_array(class1_filepath)
	class2_data = FileManager.get_csv_file_data_array(class2_filepath)

	#SPLIT INPUT DATA (learning & test sets)
	print()
	allclass_data_sets = DataManipulator.split_data_in_2_randomly(allclass_data, fraction_of_data_for_learning)
	class0_data_sets = DataManipulator.split_data_in_2_randomly(class0_data, fraction_of_data_for_learning)
	class1_data_sets = DataManipulator.split_data_in_2_randomly(class1_data, fraction_of_data_for_learning)
	class2_data_sets = DataManipulator.split_data_in_2_randomly(class2_data, fraction_of_data_for_learning)

	allclass_learning_data = allclass_data_sets[0]
	print('learning data size: ', len(allclass_learning_data))
	allclass_test_data  = allclass_data_sets[1]
	print('test data size: ', len(allclass_test_data))
	print()

	class0_learning_data = class0_data_sets[0]
	class0_test_data = class0_data_sets[1]
	class1_learning_data = class1_data_sets[0]
	class1_test_data = class1_data_sets[1]
	class2_learning_data = class2_data_sets[0]
	class2_test_data = class2_data_sets[1]

	#LEARN THE MODELS

	#Winnow2
	winnow2_0 = Winnow2()
	winnow2_1 = Winnow2()
	winnow2_2 = Winnow2()

	winnow2_0_learned_weights = winnow2_0.learn_winnow2_model(class0_learning_data)
	winnow2_1_learned_weights = winnow2_1.learn_winnow2_model(class1_learning_data)
	winnow2_2_learned_weights = winnow2_2.learn_winnow2_model(class2_learning_data)
	print('Winnow2 learned weights for class 0:')
	print(winnow2_0_learned_weights)
	print()
	print('Winnow2 learned weights for class 1:')
	print(winnow2_1_learned_weights)
	print()
	print('Winnow2 learned weights for class 2:')
	print(winnow2_2_learned_weights)
	print()

	#Naive Bayes
	naive_bayes = NaiveBayes(number_of_classes)
	naive_bayes_learned_percents = naive_bayes.learn_naive_bayes_model(allclass_learning_data)
	print('Naive Bayes learned percentages as input[ class[ (prob0, prob1) ] ]')
	print(naive_bayes_learned_percents)
	print()

	#TEST THE MODELS

	#Winnow2
	print('Testing Winnow2 model')
	winnow2_multi_model_test_results = Winnow2.test_multiple_winnow2_models(allclass_test_data, [winnow2_0, winnow2_1, winnow2_2])
	print('classification attempts(', winnow2_multi_model_test_results[0], '), \
#fails(', winnow2_multi_model_test_results[1], '), \
#success(' , winnow2_multi_model_test_results[2], ')')
	print()

	#Naive Bayes
	print('Testing Naive Bayes model')
	naive_bayes_test_results = naive_bayes.test_naive_bayes_model(allclass_test_data)
	print('#classification attempts(', naive_bayes_test_results[0], '), \
#fails(', naive_bayes_test_results[1], '), \
#success(' , naive_bayes_test_results[2], ')')
	print()


if __name__ == '__main__':
	main()
