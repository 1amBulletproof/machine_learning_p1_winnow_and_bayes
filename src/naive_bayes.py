#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Naive Bayes Classified implementation

import numpy as np
import argparse

#@TODO: redo this class (and others) with numpy or pandas?

#=============================
# NaiveBayes
#
# - Class to encapsulate a Naive Bayes model
#=============================
class NaiveBayes:

	def __init__(self, number_of_classes, pseudo_probability=0.001, pseudo_samples=1):
		self.number_of_classes = number_of_classes
		self.pseudo_probability = pseudo_probability # will use 1 pseudo example added w/ probability 
		self.pseudo_samples = pseudo_samples
		self.feature_percents = list()
		self.class_percents = list()

	#=============================
	# LEARN_NAIVE_BAYES_MODEL()
	#
	#	- Learn a naive bayes model (probabilities) from given learn_vectors
	#	- Warning, this will not be a scalable, efficient solution!
	#
	#@param learn_vectors	2D matrix of format [X0,X1...Xn, Class]
	#@param number_of_classes	number of classes to expect
	#@return				percentage table
	#=============================
	def learn_naive_bayes_model(self, learn_vectors):
		sample_size = self._get_number_of_input_vectors(learn_vectors)
		class_totals = list()
		all_feature_totals = list()

		for classification in range(0, self.number_of_classes, 1):
			class_totals.append(0)

		number_of_features = self._get_number_of_inputs(learn_vectors[0])
		for feature in range(0, number_of_features, 1):
			tmplist = []
			for classification in range(0, self.number_of_classes, 1):
				tmplist.append([0,0])
			all_feature_totals.append(tmplist)

		#Count all relevant data
		for learn_vector in learn_vectors:
			classification = self._get_expected_result(learn_vector)
			class_totals[classification] += 1

			for input_itr in range(0, number_of_features, 1):
				input_feature_val = int(learn_vector[input_itr])
				all_feature_totals[input_itr][int(classification)][input_feature_val] += 1

		#print('all_feature_totals')
		#print(all_feature_totals)
		#print('class_totals')
		#print(class_totals)
		
		#Create class percentage table based on counts: for ease of use
		for class_itr in range(0, self.number_of_classes, 1):
			tmp_class_percent = float(class_totals[class_itr]) / float(sample_size)
			self.class_percents.append(tmp_class_percent)
		#print(self.class_percents)

		#Create input percentage table based on counts: for ease of use
		for input_itr in range(0, number_of_features, 1):
			self.feature_percents.append(list())
			for classification in range(0, self.number_of_classes, 1):
				number_iterations_class = class_totals[classification]

				number_iterations_feature_0_for_class = all_feature_totals[input_itr][classification][0]
				probability_of_input_0_for_class = \
					float(number_iterations_feature_0_for_class + \
							self.pseudo_samples * self.pseudo_probability) / \
					float(number_iterations_class + self.pseudo_samples)

				number_iterations_feature_1_for_class = all_feature_totals[input_itr][classification][1]
				probability_of_input_1_for_class = \
					float(number_iterations_feature_1_for_class + \
							self.pseudo_samples * self.pseudo_probability) / \
					float(number_iterations_class + self.pseudo_samples )

				self.feature_percents[input_itr].append(
						(probability_of_input_0_for_class, probability_of_input_1_for_class) )

				#print('input_itr ', input_itr, 'class ', classification)
				#print('number_iterations_feature_0_for_class', number_iterations_feature_0_for_class)
				#print('number_iterations_feature_1_for_class', number_iterations_feature_1_for_class)
				#print('number_iterations_class', number_iterations_class)
				#print('probability of input 0 for class', probability_of_input_0_for_class)
				#print('probability of input 1 for class', probability_of_input_1_for_class)

		#print('probability table:')
		#print(self.feature_percents)

		return self.feature_percents
	
	#=============================
	# TEST_NAIVE_BAYES_MODEL() #
	#
	#	- test the internal Naive Bayes model (percents) for given test_vectors
	#	- Assumes the model has already been learned! Otherwise pointless
	#
	#@param test_vectors	2D matrix of format [X0,X1...Xn, Class]
	#@return				ouput_vector
	#=============================
	def test_naive_bayes_model(self, test_vectors):
		class_attempts = 0
		class_fails = 0
		class_success = 0

		number_of_inputs = self._get_number_of_inputs(test_vectors[0])

		final_product = 1
		for test_vector in test_vectors:
			classification_stats = [-1, -1.0] #[class, calculated probability]
			for classification in range(0, self.number_of_classes, 1):
				current_classification_probability = self.class_percents[classification]
				for input_itr in range(0, number_of_inputs, 1):
					input_val = int(test_vector[input_itr])
					
					current_classification_probability *= \
						self.feature_percents[input_itr][classification][input_val]

				if current_classification_probability > classification_stats[1]:
					classification_stats = [classification, current_classification_probability]

			class_attempts += 1
			if classification_stats[0] == self._get_expected_result(test_vector):
				class_success += 1
			else:
				class_fails += 1

		#4 - choose the largest value

		return (class_attempts, class_fails, class_success)

	#=============================
	# _GET_NUMBER_OF_INPUTS()
	#	-The expected input format is [X1, X2, ... Xn, Fn]
	#=============================
	def _get_number_of_inputs(self, data_vector):
		return len(data_vector) - 1

	#=============================
	# _GET_NUMBER_OF_INPUT_VECTORS()
	#	-The expected input format is [X1, X2, ... Xn, Fn]
	#=============================
	def _get_number_of_input_vectors(self, data_vectors):
		return len(data_vectors)

	#=============================
	# _GET_DATA_INPUTS()
	#	-The expected input format is [X1, X2, ... Xn, Fn]
	#=============================
	def _get_data_inputs(self, data_vector):
		inputs = data_vector[0:self._get_number_of_inputs(data_vector) ] #grabs start:end-1
		return inputs

	#=============================
	# _GET_EXPECTED_RESULT()
	#	-The expected input format is [X1, X2, ... Xn, Fn]
	#=============================
	def _get_expected_result(self, data_vector):
		expected_result = int(data_vector[-1]) #prevent possible string returned
		return expected_result


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main() - testing the learning & predictive ability of Naive Bayes')

	print()
	print('TEST 1: learn the model')
	print('input data:')
	test_data = [[0, 0, 0], [0, 1, 0], [ 1, 0, 1], [1, 1, 1]] #Should be 100% correlated to X1
	print(test_data)

	number_of_classes = 2
	naive_bayes = NaiveBayes(number_of_classes)
	naive_bayes_learned_percents = naive_bayes.learn_naive_bayes_model(test_data)
	print('learned percentages')
	print(naive_bayes_learned_percents)

	print()
	print('TEST 2: test the model')
	print('input data:')
	print(test_data)
	naive_bayes_test_results = naive_bayes.test_naive_bayes_model(test_data) #Should get this right since it's the training data!
	print('classification attempts(', naive_bayes_test_results[0], '), \
#fails(', naive_bayes_test_results[1], '), \
#success(' , naive_bayes_test_results[2], ')')


if __name__ == '__main__':
	main()
