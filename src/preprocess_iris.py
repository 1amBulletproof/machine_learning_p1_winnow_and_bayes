#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Pre-process 'iris.data' (rearrange && binarize the data)

import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator

#=============================
# PROCESS_IRIS_DATA()
#
#	- Pre-process iris data
#		- Turn category values into bins for cm lengths
#	- Outputs final matrix to stdout 
#
#@param	file_path	path to the csv file with voting data
#@return
#=============================
def process_house_votes_data(file_path):
	#print('LOG: process_house_votes_data() START')
	#print(file_path)

	original_data = FileManager.get_csv_file_data_array(file_path)
	#print('LOG: ORIGINAL data')
	#print(original_data[0])

	#print('LOG: group input values into bins')
	num_bins = 4
	#Fix sepal length 4.3 - 7.9
	column = 0
	min_val = 4.3
	max_val = 7.9
	data_in_bins = \
		_bin_input_attribute(original_data, column, min_val, max_val, num_bins)
	#Fix sepal width 2.0 - 4.4
	column = 1
	min_val = 2.0
	max_val = 4.4
	data_in_bins = \
		_bin_input_attribute(original_data, column, min_val, max_val, num_bins)
	#Fix petal length 1.0 - 6.9
	column = 2
	min_val = 1.0
	max_val = 6.9
	data_in_bins = \
		_bin_input_attribute(original_data, column, min_val, max_val, num_bins)
	#Fix petal width 0.1 - 2.5
	column = 3
	min_val = 0.1
	max_val = 2.5
	data_in_bins = \
		_bin_input_attribute(original_data, column, min_val, max_val, num_bins)
	#print()
	#print(data_in_bins)
	#print()

	#print('LOG: turn input features into binary values (0,1)')
	col_idx = 0
	final_data = DataManipulator.expand_attributes_to_binary_values(data_in_bins, col_idx, num_bins)
	for col in data_in_bins[0]:
		col_idx += num_bins #This is because columns are inserted each itr
		#print('length of final_data')
		#print(len(final_data[0]))
		if col_idx == (len(final_data[0]) - 1): #Skip last col b/c it's the class value
			#print('LOG: Stop here - skip the final column which is classification')
			break
		final_data = DataManipulator.expand_attributes_to_binary_values(final_data, col_idx, num_bins)

	#print(final_data[0])
	return final_data


#=============================
# _BIN_INPUT_ATTRIBUTE()
#
#	- Turn values (real numbers, categories, etc) into bins, i.e. 0, 1, 2, ....
#
#@param	data	input data matrix where every value is modified
#=============================
def _bin_input_attribute(data, col_idx, min_val, max_val, number_of_bins):
	#Turn values (real numbers, multiple string categories, etc) into bins, i.e. 0,1,2, ..... 

	val_range = float(max_val - min_val)
	bin_size = float(val_range / number_of_bins)

	for row_idx in range(0, len(data), 1):
		value = data[row_idx][col_idx]
		bin_val_divider = min_val
		
		bin_num = 0
		while(bin_val_divider <= max_val):
			bin_val_divider += bin_size
			if value <= bin_val_divider:
				data[row_idx][col_idx] = bin_num
				break
			bin_num += 1

	return data

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to pre-process House-Votes-84.data file')
	parser = argparse.ArgumentParser(description='Pre-process "iris.data" file')
	parser.add_argument('file_path', type=str, help='full path to input file')
	parser.add_argument('-o', action='store_true', help='output to csv file')
	args = parser.parse_args()
	output_to_csv = args.o

	processed_data = process_house_votes_data(args.file_path)
	if output_to_csv == True:
		filename = 'output.csv'
		FileManager.write_2d_array_to_csv(processed_data, filename)
	else:
		print(processed_data)


if __name__ == '__main__':
	main()
