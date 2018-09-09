#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Pre-process 'House-Votes-84.data' (rearrange && binarize the data)

import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator

#=============================
# PROCESS_HOUSE_VOTES_DATA()
#
#	- Pre-process house votes data
#		- Turn category values into 0 (voted no), 1(voted yes), and 2(did NOT vote)
#			- Move the classification to the final columnn in the matrix
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

	#print('LOG: Class moved to final column')
	original_col_with_class = 0
	columns_moved_data = DataManipulator.move_column_to_end(original_data, original_col_with_class)
	#print(columns_moved_data[0])

	#print('LOG: group input values into bins')
	data_in_bins = _bin_input_attributes(columns_moved_data)
	#print(data_in_bins[0])

	#print('LOG: turn input features into binary values (0,1)')
	num_bins = 3
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
# _BIN_INPUT_ATTRIBUTES()
#
#	- Turn values (real numbers, categories, etc) into bins, i.e. 0, 1, 2, ....
#
#@param	data	input data matrix where every value is modified
#=============================
def _bin_input_attributes(data):
	#Turn values (real numbers, multiple string categories, etc) into bins, i.e. 0,1,2, ..... 
	row_idx = 0
	for row in data:
		col_idx = 0
		for val in row:
			if val == 'n':
				data[row_idx][col_idx] = 0
			elif val == 'y':
				data[row_idx][col_idx] = 1
			elif val == '?':
				data[row_idx][col_idx] = 2
			col_idx += 1
		row_idx += 1
	return data

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to pre-process House-Votes-84.data file')
	parser = argparse.ArgumentParser(description='Pre-process "House-Votes-84.data" file')
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
