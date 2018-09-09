#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Pre-process 'breast-cancer-wisconsin.data'

import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator

#=============================
# PROCESS_BREAST_CANCER_WISCONSIN()
#
#	- Pre-process breast cancer wisconsin data
#		- Ignore column 0 (just ID) , turn classification into either 0 (for '2' or benign) or 1 (for '4' or malignant), && create vectors for the bins provided (1-10)
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

	#Eliminate first column of data, convert class from 2,4 -> 0,1 && make bins are 0-9, not 1-10
	modified_data = list()
	number_of_rows = len(original_data)
	for vector_idx in range(0, number_of_rows, 1):
		tmplist = original_data[vector_idx][1:]
		number_of_cols = len(tmplist)
		for col_idx in range(0, number_of_cols, 1):
			if col_idx == (number_of_cols - 1): #classificiaton value {2,4}
				#Classification value, {2,4} -> {0,1}
				if tmplist[col_idx] == 2:
					tmplist[col_idx] = 0
				elif tmplist[col_idx] == 4:
					tmplist[col_idx] = 1
				else:
					print('ERROR: Expected {2,4} but instead got ', tmplist[col_idx])
					#no modification made
			else: #normal bin value
				tmplist[col_idx] = tmplist[col_idx] - 1 #bins 1-10 -> 0-9
				
		modified_data.append(tmplist)

	#print()
	#print(modified_data)
	#print()

	#print('LOG: turn input features into binary values (0,1)')
	num_bins = 10
	col_idx = 0
	final_data = DataManipulator.expand_attributes_to_binary_values(modified_data, col_idx, num_bins)
	for col in modified_data[0]:
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
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to pre-process House-Votes-84.data file')
	parser = argparse.ArgumentParser(description='Pre-process "breast-cancer-wisonsin.data"')
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
