#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Winnow2 algo. implementation

import numpy as np
import argparse

#@TODO: change prints to logs?
#@TODO: redo this class (and others) with numpy or pandas?

#=============================
# WINNOW2
#
# - Class to encapsulate a Winnow2 model
#=============================
class Winnow2:

	def __init__(self, alpha=2, threshold=0.5, default_weight=1, weights=[]):
		self.alpha = alpha
		self.threshold = threshold
		self.default_weight = default_weight
		self.weights = weights
		#@TODO: handle number of weights > number_of_inputs
		#@TODO: vary alpha, try 1.5, 2, 3
		#@TODO: vary threshold, try n/2 instead of just 0.5

	#=============================
	# LEARN_WINNOW2_MODEL()
	#
	#	- Learn a winnow2 model (weights) from given learn_vectors and other parameters
	#
	#@param learn_vectors	2D matrix of format [X0,X1...Xn, Class]
	#@return	weights
	#=============================
	def learn_winnow2_model(self, learn_vectors):
		#print('LOG: learn_winnow2_model START')

		#Default any non-provided weight to 1.0 ... we probalby should be providing default weights for ALL or 0 features
		number_of_inputs = self._get_number_of_inputs(learn_vectors[0])
		for idx in range(len(self.weights), number_of_inputs, 1):
			self.weights.append(float(self.default_weight))

		#Trying to set threshold to relevant value
		self.threshold = self._get_number_of_inputs(learn_vectors[0]) / 2

		for idx in range(0, self._get_number_of_input_vectors(learn_vectors), 1):

			model_result = self._winnow2_classification(learn_vectors[idx], self.weights)
			expected_result = self._get_expected_result(learn_vectors[idx]) 

			if model_result != expected_result:
				input_idx = 0
				for data_input in self._get_data_inputs(learn_vectors[idx]):
					if (model_result == 0) and (expected_result == 1): #Promotion
						self.weights[input_idx] = self._get_promoted_weight(data_input, self.weights[input_idx])
					elif (model_result == 1) and (expected_result == 0): #Demotion
						self.weights[input_idx] = self._get_demoted_weight(data_input, self.weights[input_idx])
					else: #Error
						print('ERROR: A result is bad')
						print('ERROR: Model result(' , model_result , ') or expected result(' , expected_result , ')')
					input_idx += 1

		return self.weights

	#=============================
	# _WINNOW2_CLASSIFICATION()
	#
	#	- Create classification based on winnow2 algorithm && input vector
	#	- Private fcn
	#@param		data_vectors
	#@return	model result (classification)	
	#=============================
	def _winnow2_classification(self, data_vector, weights):
		model_result = 0

		inputs = self._get_data_inputs(data_vector)
		expected_result = self._get_expected_result(data_vector)

		#print()
		#print('CLASS START')
		#print('inputs')
		#print(inputs)
		#print('weights')
		#print(weights)

		fcn_result = self._summation_fcn(inputs, weights)
		#print('fcn_result summation: ', fcn_result)

		if fcn_result > self.threshold:
			model_result = 1
			#print('classification = ', model_result)
		else: 
			model_result = 0
			#print('classification = ', model_result)

		return model_result
	
	#=============================
	# _SUMMATION_FCN()
	#
	#	- do demotion and return the new weight
	#	- Private fcn
	#=============================
	def _summation_fcn(self, variables, weights):
		#ensure there is a weight for every input
		if len(variables) != len(weights):
			print('ERROR: len(variables) ', len(variables),' != len(weights) ', len(weights), ')')
			#Possibly throw an exception!
			return -1
		
		#@TODO: replace this fcn w/ numpy vectorized multiplication (i.e. inputs * weights)
		#Summation of all Xi * Wi
		summation = 0
		for idx in range(0, len(variables), 1):
			summation += float(float(variables[idx]) * float(weights[idx])) # ensure fp arithmetic
		return summation

	#=============================
	# TEST_WINNOW2_MODEL() #
	#	- test the internal winnow2 model (weights) for given test_vectors
	#
	#@param test_vectors	2D matrix of format [X0,X1...Xn, Class]
	#@param	threshold		Value which determines whether output is 0 or 1 
	#@param	weights			Vector of input weights (in case you want to test weights other than your own)
	#@return				ouput_vector
	#=============================
	def test_winnow2_model(self, test_vectors, weights=[]):
		#@TODO: need to support multiple classifications where function value's must be compared for multiple weight vectors/resuts && the largest one chosen!
		if (len(weights) > 0):
			print("LOG: input weights for the testing")
			#@TODO: this will probably assist with testing multiple models?!
			#@TODO: evetually for multiple categories, must support ingesting all the necessary weights and running classification per weight set 

		#Trying to set threshold to relevant value
		self.threshold = self._get_number_of_inputs(test_vectors[0]) / 2

		class_attempts = 0
		class_fails = 0
		class_success = 0
		for idx in range(0, self._get_number_of_input_vectors(test_vectors), 1):
	
			model_result = self._winnow2_classification(test_vectors[idx], self.weights) #assumes we already have weights
			expected_result = self._get_expected_result(test_vectors[idx])
			#print('model_result(', model_result, '), expected_result(', expected_result, ')')

			if model_result != expected_result:
				class_fails += 1
			else:
				class_success += 1

			class_attempts += 1
				
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
	# _GET_PROMOTED_WEIGHT()
	#
	#	- do promotion and return the new weight
	#	- Private fcn
	#=============================
	def _get_promoted_weight(self, data_input, weight):
		#print('LOG: PROMOTION')
		if data_input == 0:
			return weight
		elif data_input == 1:
			return float(weight * self.alpha)
		else:
			print('ERROR: Data input(' , data_input , ') was neither 0 or 1')

	#=============================
	# _GET_DEMOTED_WEIGHT()
	#
	#	- do demotion and return the new weight
	#	- Private fcn
	#=============================
	def _get_demoted_weight(self, data_input, weight):
		#print("LOG: DEMOTION")
		if data_input == 0:
			return weight
		elif data_input == 1:
			return float(weight / self.alpha)
		else:
			print('ERROR: Data input(' , data_input , ') was neither 0 or 1')


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main() - testing the learning & predictive ability of winnow2')

	print()
	print('TEST 1: learn the model')
	print('input data:')
	test_data = [[0, 0, 0], [0, 1, 0], [ 1, 0, 1], [1, 1, 1]] #Should be 100% correlated to X1
	print(test_data)

	winnow2 = Winnow2()
	winnow2_learned_weights = winnow2.learn_winnow2_model(test_data)
	print('learned weights')
	print(winnow2_learned_weights)

	print()
	print('TEST 2: test the model')
	print('input data:')
	print(test_data)
	winnow2_test_results = winnow2.test_winnow2_model(test_data) #Should get this right since it's the training data!
	print('classification attempts(', winnow2_test_results[0], '), \
fails(', winnow2_test_results[1], '), \
success(' , winnow2_test_results[2], ')')


if __name__ == '__main__':
	main()
