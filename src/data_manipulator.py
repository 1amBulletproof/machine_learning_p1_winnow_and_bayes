#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	General Data Manipulation 

import argparse
import copy
import random


#=============================
# DATAMANIPULATOR
#
#	- Contains methods to useful for pre-processing data (typically 2D matrixes)
#@TODO: replace some funcitonality with support for numpy / pandas
#=============================
class DataManipulator:

	#=============================
	# EXPAND_ATTRIBUTE_TO_BINARY_VALUES()
	#
	#	- Expand attribute col to multiple cols for given multi-bin valued attribute
	#	- Creates copy of input data - input data unmodified but slower
	#	- This is sometimes referred as "one-hot coding"
	#
	#@param	data		2D matrix
	#@param	col_idx		column attribute to expand
	#@param	num_bins	max number of bins 
	#@return			modified 2D matrix with ADDITIONAL cols for each bin 
	#=============================
	@staticmethod
	def expand_attributes_to_binary_values(data, col_idx, num_bins):
		#print('LOG: expand_attributes_to_binary_values() START')
		#print('Input data')
		#print(data)
		modified_data = list(data)
		modified_data = copy.deepcopy(data)
		length_of_vector = len(data[0])

		if col_idx >= length_of_vector:
			print('ERROR: Column idx ' , col_idx , ' outside input data cols ' , length_of_vector)
			return 

		row_idx = 0
		for row in modified_data:
			row_val = row[col_idx]
			binary_category_values = DataManipulator._convert_bin_val_into_binary_vector(row_val, num_bins)
			#print('binary_cateogory_vals')
			#print(binary_category_values)

			insert_idx = col_idx
			#replace previous value

			#print('row before modify')
			#print(modified_data[row_idx])
			modified_data[row_idx][insert_idx:(insert_idx+1)] = binary_category_values
			#print('row after modify')
			#print(modified_data[row_idx])

			row_idx += 1

		#print('Output data')
		#print(modified_data)
		#print('LOG: expand_attributes_to_binary_values() START')
		return modified_data

	#=============================
	# _CONVERT_BIN_VAL_INTO_BINARY_VECTOR()
	#
	#	- Private fcn (helper method)
	#	- creates vector of binary values to represent one hot coding
	#	- 0 - base (Assumes 0 is a valid value)
	#	- i.e. val 3 && num_bins 5 -> [0,0,0,1,0]
	#=============================
	@staticmethod
	def _convert_bin_val_into_binary_vector(val, num_bins):
		bin_vals = [0 for val in range(num_bins)]
		bin_vals[val] = 1
		return bin_vals

	#=============================
	# MOVE_COLUMN_TO_END()
	#
	#	- mv given column to be the last column in 2D matrix
	#	- returns a new 2D array (i.e. creates one from scratch, doesn't modify input)
	#
	#@param data		2D data array
	#@param col_idx	column to move to the last column
	#@return			modified data
	#=============================
	@staticmethod
	def move_column_to_end(data, col):
		#print('LOG: move_column_to_end() START')
		modified_data = []
		length_of_vector = len(data[0])

		if col >= length_of_vector:
			print('ERROR: Column idx ' , col , ' outside input data cols ' , length_of_vector)
			return 

		row_idx = 0
		for row in data:
			modified_data.append([])
			col_idx = 0
			swap_val = 'BLAH'
			for val in row:
				if col == col_idx:
					swap_val = val
				else:
					modified_data[row_idx].append(val)
				col_idx += 1

			modified_data[row_idx].append(swap_val)
			row_idx += 1
		
		#print(input_data)
		#print('LOG: move_column_to_end() END')
		return modified_data

	#=============================
	# SPLIT_DATA_IN_2_RANDOMLY()
	#
	#	- Split given input data (expected 2D array) into 2 pieces (one is typically for learning, the other testing) based on given fraction
	#
	#@param data		expected 2d matrix to split up
	#@param fraction	fraction of split in the first group
	#@return			tuple(data_slice1_as_2d_matrix, data_slice2_as_2d_matrix)
	#=============================
	@staticmethod
	def split_data_in_2_randomly(data, fraction):

		#print('data before shuffle:')
		#print(data)

		#Randomize the data
		data_copy = copy.deepcopy(data)
		random.shuffle(data_copy)

		#print('data after shuffle:')
		#print(data)
		
		lines = round((10 * fraction), 1)
		if lines < 1 or lines > 9:
			print('ERROR: bad fraction (too small or large) ' , fraction)
			return

		data_set_1 = list()
		data_set_2 = list()
		row_counter = 0
		for row in data_copy:
			if row_counter <= lines:
				data_set_1.append(row)
			elif row_counter > lines:
				data_set_2.append(row)

			if row_counter == 10:
				row_counter = 0
			row_counter += 1

		return (data_set_1, data_set_2)

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to test class "DataManipulator"')
	parser = argparse.ArgumentParser(description='Manipulate Data')
	#parser.add_argument('file_path', metavar='F', type=str, help='full path to input file')
	#args = parser.parse_args()

	test_data = [['classA', 0, 1, 2], ['classB', 2, 1, 0]]
	print()
	print('TEST 1: move class to end')
	print('test_data:')
	print(test_data)

	col_to_move_to_end = 0
	moved_data = DataManipulator.move_column_to_end(test_data, col_to_move_to_end)
	print('modified_data:')
	print(moved_data)

	print()
	print('TEST 2: expand attributes')
	print('test_data: ')
	print(moved_data)

	num_bins = 3
	col = 0 
	discrete_data = DataManipulator.expand_attributes_to_binary_values(moved_data, col, num_bins)
	col = 3
	discrete_data = DataManipulator.expand_attributes_to_binary_values(discrete_data, col, num_bins)
	col = 6
	discrete_data = DataManipulator.expand_attributes_to_binary_values(discrete_data, col, num_bins)

	print('modified_data:)')
	print(discrete_data)

	print()
	print('TEST 3: split data')
	test_data.append(['classC', 0, 1, 2])
	test_data.append(['classD', 0, 1, 2])
	test_data.append(['classE', 0, 1, 2])
	test_data.append(['classF', 0, 1, 2])
	test_data.append(['classG', 0, 1, 2])
	test_data.append(['classH', 0, 1, 2])
	test_data.append(['classI', 0, 1, 2])
	test_data.append(['classJ', 0, 1, 2])
	print('test_data: ')
	print(test_data)

	data_sets = DataManipulator.split_data_in_2_randomly(test_data, 0.7)
	print('split_data: ')
	print(data_sets[0])
	print(data_sets[1])
	

if __name__ == '__main__':
	main()
