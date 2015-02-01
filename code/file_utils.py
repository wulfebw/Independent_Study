"""
filename: file_utils.py
date: 3/20/2014
author: Blake

This code is for performing an assortment of file operations. 
"""


import glob
import os
import csv
import shutil
import math
import sys
import psutil
import traceback
import random
import numpy as np
from scipy.io import wavfile
import datetime
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

SAMPLES_PER_FILE = 6
TRAIN_PERCENTAGE = .5
VALIDATION_PERCENTAGE = .3
TEST_PERCENTAGE = .2

""" utilities for diarize.py """

def graph_values(name, values, output_dir, tag='', y_scale=1.5, img_extension='gif'):
	"""
	Method for graphing the values
	"""

	plt.plot(values)
	plt.title(name)
	low = values.min() * y_scale
	high = values.max() * y_scale
	pylab.ylim([low,high])
	plt.savefig('{0}/{1}{2}.{3}'.format(output_dir, name, tag, img_extension))
	plt.clf()

def get_args():
	""" 
	Argument parser for options used in calling file.
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument('-i1', dest='input_file_1', default=DEFAULT_IN)
	parser.add_argument('-i2', dest='input_file_2', default=DEFAULT_IN)
	parser.add_argument('-od', dest='output_dir', default=DEFAULT_OUT)
	parser.add_argument('-e', dest='openSMILE', default=openSMILE)
	args = parser.parse_args()
	return (args.input_file_1, args.input_file_2, args.output_dir, args.openSMILE)

def remove_extension(filename):
	""" 
	Removes extension from ".energy.csv" and ".wav" files
	"""

	if filename.find('.wav'):
		index_of_period = filename.index('.wav')
	elif filename.find('.energy.csv'):
		index_of_period = filename.index('.energy.csv')
	else:
		index_of_period = 0
	return filename[:index_of_period]

def get_file_name(filename):
	""" 
	Method for parsing the filename. Definitely a better way to do this 
	"""
	
	index_last_slash = filename.rindex('/')
	return filename[index_last_slash + 1:]
	
""" end utilities for diarize.py """

""" 
This section is for organizing the files into folders. Don't use it in the future - it complicates things
more than necessary.
"""

GROUPING_EXT = '.avi'

def get_date_name(filename, indicator='/', indicator_len=1):
	f_index = filename.rfind('f')
	m_index = filename.rfind('m')

	if f_index == -1 or m_index == -1:
		raise IndexError

	start = filename.rfind(indicator) + indicator_len
	session = filename[start:start+4]
	male = filename[m_index:m_index+3]
	female = filename[f_index:f_index+3]
	return '{0}_{1}_{2}'.format(session, female, male)

def create_session_dirs(filenames, dirname):
	""" 
	create the session dirs 
	"""
	sessions = set()
	# get list of session names
	for f in filenames:
		start = f.rfind('/') + 1
		end = start + 4
		cur_session = f[start:end]
		sessions.add(cur_session)
	
	for s in sessions:
		if not os.path.exists('{0}/{1}'.format(dirname, s)):
			os.makedirs('{0}/{1}'.format(dirname, s))

def create_date_dirs(filenames, dirname):
	""" 
	create the date dirs 
	"""
	dates = set()
	# get list of date names
	for f in filenames:
		date_name = get_date_name(f)
		dates.add(date_name)

	for d in dates:
		start = d.rfind('/') + 1
		end = start + 4
		session = d[start:end]
		if not os.path.exists('{0}/{1}/{2}'.format(dirname, session, d)):
			os.makedirs('{0}/{1}/{2}'.format(dirname, session, d))

def move_files(filenames):
	for f in filenames:
		filename = os.path.basename(f)
		path = os.path.dirname(f)
		date_name = get_date_name(f)
		start = f.rfind('/') + 1
		session = f[start:start+4]

		shutil.move(f, '{0}/{1}/{2}'.format(path, session, date_name))

def sort_dates(dirname):
	""" 
	Automatically sort individual video files into session folders 

		(1) load in list of file names
		(2) create folder for each session 
		(3) create data folders in each session
		(4) move each file into the appropriate directory
	"""

	files = glob.glob('{0}/*.wav'.format(dirname))
	create_session_dirs(files, dirname)
	create_date_dirs(files, dirname)
	move_files(files)




""" 
This section is for slicing an audio file into 30-second samples 
"""

def get_slices(input_file):
	try:
		samp_rate, input_wav = wavfile.read(input_file)
		slices = []
		slice_len = len(input_wav) / SAMPLES_PER_FILE
		cur_slice = []

		for x in range(SAMPLES_PER_FILE):
			start = slice_len*x
			end = slice_len*(x+1)
			cur_slice = input_wav[start:end]
			slices.append(np.array(cur_slice))

		return slices, samp_rate

	except Exception as e:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		print ''.join('!! ' + line for line in lines)  # Log it or whatever here
		sys.exit()

def store_slices(slices, sample_rate, output_dir, date, filename):
	for index, cur_slice in enumerate(slices):

		slice_dir = '{0}/{1}_{2}'.format(output_dir, date, index)
		output_file = '{0}/{1}_{2}.wav'.format(slice_dir, filename, index)

		if not os.path.exists(slice_dir):
			os.makedirs(slice_dir)

		wavfile.write(output_file, sample_rate, cur_slice)

def get_filename(filename):
	start = 4
	end = -4
	return os.path.basename(filename)[start:end]

def slice_and_store_files(input_dir, output_dir):
	files = glob.glob('{0}/seg_*.wav'.format(input_dir))
	for f in files:
		slices, sample_rate = get_slices(f)
		date = get_date_name(f, 'seg_', 4)
		filename = get_filename(f)
		store_slices(slices, sample_rate, output_dir, date, filename)

def create_samples(dirname):
	for root, subdirs, files in os.walk(dirname):
		for subdir in subdirs:
			if 'segmentation' in subdir:
				output_dir = '{0}/samples'.format(root)
				input_dir = '{0}/{1}'.format(root, subdir)
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
				slice_and_store_files(input_dir, output_dir)

""" 
This section for reading in arff files b/c apparently none of the libraries that exist work. 
"""
def get_arff_data(arff_file):
	try:
		with open(arff_file, 'r') as f:
			lines = f.readlines()
			for l in lines:
				if l.startswith('\'data/inputs/audio/valid'):
					values = l.split(',')
					if len(values) > 1:
						values.pop(0) # get rid of name value here
						return values
	except Exception as e:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		print('\n')
		print ''.join('!! ' + line for line in lines)  # Log it or whatever here
		sys.exit()


""" 
This section is for aggregating the sample arff files.
"""

class Sample(object):
	""" 
	Sample object represents a single sample of extracted features.
	"""

	def __init__(self, filename):
		self.filename = filename
		self.date_name, self.sample_name = self.get_sample_and_date_name()
		self.values = self.load_values()
		self.is_female = self.is_female()

	def set_filename(filename):
		if filename.endswith('arff'):
			self.filename = filename

	def get_sample_and_date_name(self):
		indicator = '/'
		# filename of form .../####_f##_m##_#.arff or .../####_m##_f##_#.arff
		# goal is to extract female
		f_index = self.filename.rfind('_f') + 1
		m_index = self.filename.rfind('m')
		indicator_len = len(indicator)

		if f_index == -1 or m_index == -1:
			raise IndexError

		start = self.filename.rfind(indicator) + indicator_len
		session = self.filename[start:start+4]
		male = self.filename[m_index:m_index+3]
		female = self.filename[f_index:f_index+3]
		sample_num = self.filename[-6]
		return '{0}_{1}_{2}'.format(session, female, male), '{0}_{1}_{2}_{3}'.format(session, female, male, sample_num)

	def load_values(self):
		try:
			values = get_arff_data(self.filename)
			return values
		except Exception as e:
			print('let\'s just not worry about these')
			exc_type, exc_value, exc_traceback = sys.exc_info()
			lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
			print ''.join('!! ' + line for line in lines)  # Log it or whatever here
			sys.exit()

	def is_female(self):
		f_index = self.filename.rindex('_f')
		m_index = self.filename.rindex('m')

		if m_index < f_index:
			return False
		else:
			return True

	def is_valid(self):
		return len(self.values) > 1 

class Pair(object):
	"""
	This class represents a pair of sample files.
	"""

	def __init__(self, date_name):
		self.date_name = date_name
		self.m_sample = None
		self.f_sample = None
		self.outcome = None

	def set_female(self, f_sample):
		self.f_sample = f_sample

	def set_male(self, m_sample):
		self.m_sample = m_sample

	def set_outcome(self, outcome):
		self.outcome = outcome

	def is_valid(self):
		return self.m_sample is not None and self.f_sample is not None and self.outcome != -1

	def get_values(self):
		if self.is_valid():
			return self.f_sample.values + self.m_sample.values + [self.outcome]
		else:
			return []

def get_sample_pairs(dirname):
	# create dict mapping sample_name (includes slice) to a Pair with both female and male samples
	pair_dict = dict()
	count = 0
	for root, subdirs, files in os.walk(dirname):
		for f in files:

			if count % 100 == 0:
				print('{0} files\t'.format(count))
				
			if count > 10000:
				break

			if f.endswith('.arff'):
				count += 1

				try:
					filename = '{0}/{1}'.format(root, f)
					cur_sample = Sample(filename)

					if cur_sample.sample_name in pair_dict:
						cur_pair = pair_dict[cur_sample.sample_name]
					else:
						cur_pair = Pair(cur_sample.date_name)
						pair_dict[cur_sample.sample_name] = cur_pair

					if cur_sample.is_female:
						cur_pair.set_female(cur_sample)
					else:
						cur_pair.set_male(cur_sample)

				except Exception as e:
					exc_type, exc_value, exc_traceback = sys.exc_info()
					lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
					print ''.join('!! ' + line for line in lines)  # Log it or whatever here
					sys.exit()
	return pair_dict

def format_data_value(data, length):
	if len(data) == length:
		return data
	else:
		return '0{0}'.format(data)

def get_outcomes(outcome_file):
	d = dict()
	with open(outcome_file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		reader.next() # remove data labels
		for row in reader:
			try:
				female = format_data_value(row[2], 2)
				male = format_data_value(row[3], 2)
				session = format_data_value(row[6], 4)
				date = '{0}_f{1}_m{2}'.format(session, female, male)
				outcome = row[8]
				d[date] = outcome
			except Exception as e:
				exc_type, exc_value, exc_traceback = sys.exc_info()
				lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
				print ''.join('!! ' + line for line in lines)  # Log it or whatever here
				sys.exit()
	return d

def label_pairs_with_outcomes(pair_dict, outcome_dict):
	for key, value in pair_dict.iteritems():
		if value.date_name in outcome_dict:
			outcome = outcome_dict[value.date_name]
		else:
			outcome = -1

		value.outcome = outcome

def write_samples_to_file(output_file, sample_name_to_pair_dict):
	try:
		with open(output_file, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONE)
			for date_name, pair in sample_name_to_pair_dict.iteritems():
				if pair.is_valid():
					values = [float(i) for i in pair.get_values()]
					values[-1] = int(values[-1])
					writer.writerow(values)
				else:
					print(pair.m_sample)
					print(pair.f_sample)
	except Exception as e:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		print ''.join('!! ' + line for line in lines)  # Log it or whatever here
		sys.exit()

def aggregate_samples(dirname):
	# change .csv to .arff to produce .arff files
	output_file = 'templates/samples_wo_permute.csv'
	template = 'templates/template.csv'
	shutil.copyfile(template, output_file)
	outcome_file = 'extracted_data/interactions.csv'

	sample_name_to_pair_dict = get_sample_pairs(dirname)
	sample_name_to_outcome_dict = get_outcomes(outcome_file)
	label_pairs_with_outcomes(sample_name_to_pair_dict, sample_name_to_outcome_dict)
	write_samples_to_file(output_file, sample_name_to_pair_dict)


def load_csv(input_file, target_index=-1, delim=','):

	data_list = []	# 2d matrix of the extracted data
	with open(input_file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delim)
		reader.next()
		for row in reader:
			try:
				data_list.append(map(float, row))
			except ValueError, e:
				print('error: {0}'.format(e))
				# print('row: {0}'.format(row))

		csvfile.close()

	# in case the data is ordered, shuffle it:
	data_matrix = np.array(data_list)
	data_matrix = np.random.permutation(data_matrix)

	# extract y and x values into different matricies
	y_values = np.array(data_matrix[:, target_index])
	row_len = len(data_matrix[0])
	x_values = data_matrix[:, 1:row_len]


	# preform PCA here
	x_scaled = preprocessing.scale(x_values)
	# pca = PCA(n_components=120)
	# x_values = pca.fit_transform(x_scaled)
	# print(pca.explained_variance_ratio_)
	x_values = x_scaled

	# since the y values need to start at 0 subtract 1 from each
	#y_values -= 1

	# create a tuple of both
	data_tuple = (x_values,y_values)

	# determine the size of each set
	train_size = math.floor(len(data_matrix) * TRAIN_PERCENTAGE)
	validation_size = math.floor(len(data_matrix) * VALIDATION_PERCENTAGE)
	test_size = len(data_matrix) - (train_size + validation_size) # rest of the samples
	
	# portion sets out
	validation_end = train_size + validation_size
	train_set = (data_tuple[0][0 : train_size], data_tuple[1][0 : train_size])
	validation_set = (data_tuple[0][train_size : validation_end], \
					  data_tuple[1][train_size : validation_end])
	test_set = (data_tuple[0][validation_end:], data_tuple[1][validation_end:])

	# create shared datasets
	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(validation_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	# return together
	return_val = [(train_set_x, train_set_y), 
			(valid_set_x, valid_set_y), 
			(test_set_x, test_set_y)]

	return return_val

def load_csv_svm(inputfile, delim=','):
	"""
	This is the csv loading mechanism specifically for the SVMs.
	The commented sections labeled 'network', 'outcome', 'all but selectivity' (removes one feature), 
	and 'all'.

	:type inputfile: string 
	:param inputfile: the csv file used as input. Some have the first line as labels of the data, others do 
	not, so you'll need to set this in the 'labels line'. 

	:type delim: string
	:param delim: the delimiter used in the csv file

	"""
	data_list = []
	with open(inputfile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delim)
		#labels = reader.next()
		for row in reader:
			try:
				# outcome
				data_list.append(map(float,row[5:8]+row[9:12]+[row[-1]]))

				# network
				#data_list.append(map(float,row[1:5]+[row[-1]]))

				# all but selectivity
				# data = row[1:]
				# del(data[4])
				# del(data[10])
				# data_list.append(map(float, data))

				# all
				# data_list.append(map(float,row[:]))
			except ValueError, e:
				print('error: {0}'.format(e))

	# permute in case it's ordered
	data_matrix = np.array(data_list)
	data_matrix = np.random.permutation(data_matrix)

	x = data_matrix[:,:-1]	# [every row, every col but the last]
	y = data_matrix[:,-1]	# [every row, only the last col]

	return x,y

def print_to_file(params, test_performance, output_file, learning_rate, L1_reg, 
					L2_reg, n_epochs, dataset, batch_size, n_hidden):
	hidden_W = params[0]
	hidden_b = params[1]
	log_regress_W = params[2]
	log_regress_b = params[3]
	with open(output_file, 'a') as output:
		output.write('test error percentage: {0} %\n'.format(test_performance))
		output.write('date: {0}\n'.format(datetime.datetime.now()))
		output.write('learning_rate: {0}\nL1_reg: {1}\nL2_reg: {2}\nn_epochs: {3}\ndataset: {4}\nbatch_size: {5}\nn_hidden: {6}\n'.format(\
					learning_rate, L1_reg, L2_reg, n_epochs, dataset, batch_size, n_hidden))
		output.write('\nhidden_W:\n{0}'.format(hidden_W))
		output.write('\n' * 2)
		output.write('hidden_b:\n{0}'.format(hidden_b))
		output.write('\n' * 2)
		output.write('log_regress_W:\n{0}'.format(log_regress_W))
		output.write('\n' * 2)
		output.write('log_regress_b:\n{0}'.format(log_regress_b))
		output.write('\n' * 2)

def graph_learning_curves(train_losses, valid_losses):
	x = range(len(train_losses))
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.scatter(x[:], train_losses[:], s=10, c='b', marker="s")
	ax.scatter(x[:], valid_losses[:], s=10, c='r', marker="o")
	plt.show()

def test_graph():
	graph_learning_curves(range(0, 1000),range(10, 1010))

""" This section for selecting ad hoc features from full feature file """

def create_ad_hoc(inputfile='samples.csv', delim=','):
	all_feature_values = []
	cur_feature_values = []
	with open(inputfile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delim)
		reader.next()
		for row in reader:


			f_F0final_sma_maxPos = row[3663]
			f_F0final_sma_stddev = row[3682]
			f_F0final_sma_minPos = row[3664]
			f_pcm_RMSenergy_sma_peakRangeAbs = row[4182]

			f_logHNR_sma_amean = row[3853]
			f_voicingFinalUnclipped_sma_maxPos =row[3702]
			f_mfcc_sma_one_amean	= row[5166]
			f_mfcc_sma_one_flatness = row[5167]


			m_F0final_sma_stddev = row[10056]
			m_F0final_sma_minPos = row[10038]
			m_pcm_RMSenergy_sma_maxPos = row[6437]
			m_pcm_RMSenergy_sma_minRangeRel = row[10561]

			m_logHNR_sma_amean= row[10227]
			m_voicingFinalUnclipped_sma_maxPos = row[10076]
			m_mfcc_sma_one_amean	=row[11540]
			m_mfcc_sma_one_flatness =row[11541]


			outcome = row[-1]
			cur_feature_values = [	f_F0final_sma_maxPos,
									f_F0final_sma_stddev,
									f_F0final_sma_minPos,
									f_pcm_RMSenergy_sma_peakRangeAbs,
									f_logHNR_sma_amean,							
									f_voicingFinalUnclipped_sma_maxPos,			
									f_mfcc_sma_one_amean,				
									f_mfcc_sma_one_flatness,


									m_F0final_sma_stddev,
									m_F0final_sma_minPos,
									m_pcm_RMSenergy_sma_maxPos,
									m_pcm_RMSenergy_sma_minRangeRel,
									m_logHNR_sma_amean,						
									m_voicingFinalUnclipped_sma_maxPos,			
									m_mfcc_sma_one_amean,		
									m_mfcc_sma_one_flatness,				

									outcome]
			all_feature_values.append(cur_feature_values)


	outputfile = 'ad_hoc_samples.csv'
	with open(outputfile, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=delim)
		lables = [					'f_F0final_sma_maxPos',
									'f_F0final_sma_stddev',
									'f_F0final_sma_minPos',
									'f_pcm_RMSenergy_sma_peakRangeAbs',
									'f_logHNR_sma_amean',
									'f_voicingFinalUnclipped_sma_maxPos',
									'f_mfcc_sma[1]_amean',
									'f_mfcc_sma[1]_flatness',

									'm_F0final_sma_stddev',
									'm_F0final_sma_minPos',
									'm_pcm_RMSenergy_sma_maxPos',
									'm_pcm_RMSenergy_sma_minRangeRel',
									'm_logHNR_sma_amean',
									'm_voicingFinalUnclipped_sma_maxPos',
									'm_mfcc_sma[1]_amean',
									'm_mfcc_sma[1]_flatness',

									'outcome']
		writer.writerow(lables)
		for row in all_feature_values:
			writer.writerow(row)


""" These utilities are for combining visual features with other types """

def convert_double_id_to_date_id(double_id):
	"""
	double_id ex: 2121107_1121107
	date_id ex: 0209_f01_m02
	"""

	session = double_id[1:5]
	if double_id[0] is '2':
		female = 'f' + double_id[5:7]
		male = 'm' + double_id[-2:]
	else: 
		male = 'm' + double_id[5:7]
		female = 'f' + double_id[-2:]

	return session + '_' + female + '_' + male


def combine_visual_network(visual_samples_file, network_samples_file, output_file):
	network_date_id_to_features_dict = dict()
	with open(network_samples_file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			try:
				date_id = convert_double_id_to_date_id(row[0])
				# leave out choice:
				features = (map(float,row[1:-1]))
				network_date_id_to_features_dict[date_id] = features
			except ValueError, e:
				print('error: {0}'.format(e))

	smile_motion_labels = []
	visual_date_id_to_features_dict = dict()
	with open(visual_samples_file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		smile_motion_labels = reader.next()[1:]
		for row in reader:
			try:
				date_id = row[0]
				features = (map(float,row[1:-1])) + [int(float(row[-1]))]
				visual_date_id_to_features_dict[date_id] = features
			except ValueError, e:
				print('error: {0}'.format(e))


	labels = ['date_id', 'network_features'] + smile_motion_labels

	with open(output_file, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(labels)
		for date_id, visual_features in visual_date_id_to_features_dict.iteritems():
			# w/ segments date_without_segment_id = date_id[:-2]
			date_without_segment_id = date_id # w/o segments
			if date_without_segment_id in network_date_id_to_features_dict:
				row = [date_id]
				row += network_date_id_to_features_dict[date_without_segment_id]
				row += visual_features
				writer.writerow(row)
			else:
				print(date_id)

def combine_motion_smile(motion_samples_file, smile_samples_file, output_file):
	labels = []

	smile_date_id_to_features_dict = dict()
	with open(smile_samples_file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		labels += reader.next()[1:]
		for row in reader:
			try:
				date_id = row[0]
				features = (map(float,row[1:-1]))
				smile_date_id_to_features_dict[date_id] = features
			except ValueError, e:
				print('error: {0}'.format(e))

	motion_date_id_to_features_dict = dict()
	with open(motion_samples_file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		labels += reader.next()
		for row in reader:
			try:
				date_id = row[0]
				features = (map(float,row[1:-1])) + [int(float(row[-1]))]
				motion_date_id_to_features_dict[date_id] = features
			except ValueError, e:
				print('error: {0}'.format(e))

	labels = [labels[2]] + labels[:2] + labels[3:]

	with open(output_file, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(labels)
		for date_id, motion_features in motion_date_id_to_features_dict.iteritems():
			# w/ segments date_without_segment_id = date_id[:-2]
			if date_id in smile_date_id_to_features_dict:
				row = [date_id]
				row += smile_date_id_to_features_dict[date_id]
				row += motion_features
				writer.writerow(row)
			else:
				print(date_id)

""" end visual feature combination functions """

""" this section for initial visual features calc utilities """


def load_coordinates_from_csv(f):
	values = []
	with open(f, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			values.append(row)
	# convert strings to floats
	numerical_values = [[float(string_num) for string_num in row] for row in values]
	# row[1] indicates if whole row is zeros
	return np.array([row for row in numerical_values if row[1] != 0])

	

def format_data_value(data, length):
	if len(data) == length:
		return data
	else:
		return '0{0}'.format(data)

def inverse(gender):
	"""
	The genders used in the file are either '1' == male or '2' == female. This method flips that value
	and is used for various reasons.

	:type gender: string (of 1 or 2)
	:param gender: represents a gender ('1' or '2')
	"""
	if gender == '1':
		return '2'
	else:
		return '1'

def get_gender(gender_id):
	if gender_id is '1':
		return 'm'
	else:
		return 'f'

def put_female_first(video_name):

	f_index = video_name.find('f')
	m_index = video_name.find('m')

	if f_index is -1 or m_index is -1:
		raise Exception('invalid video filename')

	if f_index < m_index:
		return video_name, 'f'
	else:
		session = video_name[:4]
		female = video_name[8:12]
		male = video_name[4:8]
		segment_number = video_name[12:]
		return session + female + male + segment_number, 'm'

def put_male_first(video_name):
	f_index = video_name.find('f')
	m_index = video_name.find('m')

	if f_index is -1 or m_index is -1:
		raise Exception('invalid video filename')

	if m_index < f_index:
		return video_name, 'm'
	else:
		session = video_name[:4]
		male = video_name[8:12]
		female = video_name[4:8]
		segment_number = video_name[12:]
		return session + male + female + segment_number, 'f'


def load_date_choices_dict(outcome_file):
	# ex key="0209_m01_f01"
	d = dict()
	with open(outcome_file, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		reader.next() # remove data labels
		for row in reader:
			session = format_data_value(row[6],4)
			decider = format_data_value(row[3],2)
			decider_gender_num = row[1][0]
			decider_gender= get_gender(decider_gender_num)
			dater = format_data_value(row[2],2)
			dater_gender = get_gender(inverse(decider_gender_num))
			key = '{0}_{1}{2}_{3}{4}'.format(session, decider_gender, decider, dater_gender, dater)
			choice = row[7]
			d[key] = choice
	return d


def get_output_filename(input_filename, output_directory):
	"""
	Determines the output file name given the input file name.

	:type input_filename: string
	:param input_filename: the input file for which the output file name needs to be determined

	:type output_directory: string
	:param output_directory: the directory where output should be stored
	"""

	filename = os.path.splitext(os.path.basename(input_filename))[0] + '.csv'
	return os.path.join(output_directory, filename)

def write_features_to_file(values, output_file, labels):

	with open(output_file, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(labels)
		for row in values:
			writer.writerow(row)

def get_video_name_from_filename(filename):
	return os.path.splitext(os.path.basename(filename))[0]

def write_video_features_to_file(output_file, values, output_dir='/outputs/motion/'):
	filename = output_dir + output_file + '.csv'
	if not os.path.isfile(filename):
		with open(filename, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			writer.writerow(['video_id','entropy_sum', 'mean_motion_template'])
			writer.writerow(values)

def aggregate_motion_files(motion_dir='motion'):

	files = glob.glob('{0}/*.csv'.format(motion_dir))
	video_features_dict = dict()

	for f in files:
		with open(f, 'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			reader.next()
			row = reader.next()
			video_features_dict[row[0]] = row[1:]
	return video_features_dict

def memberwise_add(x,y):
	return map(add,x,y)

def get_video_key_from_filename(filename):
	return os.path.splitext(os.path.basename(filename))[0]

""" end initial visual features calc utilities """


def load_csv_svm(inputfile):
	data_list = []
	labels = []
	with open(inputfile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		feature_labels = reader.next()
		for row in reader:
			try:
				if not ('3' in row and '4' in row):

					# just network
					#data_list.append(map(float, row[1:5] + [row[-1]]))

					# just visual
					#data_list.append(map(float, row[5:]))

					# not usual
					#data_list.append(map(float,row[0:]))

					# ususal
					data_list.append(map(float,row[1:]))
					
					# ind = 18
					# data_list.append(map(float, row[ind:ind + 1] + [row[-1]]))


					labels.append(row[0])
			except ValueError, e:
				print('error: {0}'.format(e))

	# permute in case it's ordered
	data_list = np.random.permutation(data_list)
	data_matrix = np.array(data_list)

	x = data_matrix[:,:-1]	# [every row, every col but the last]
	y = data_matrix[:,-1]	# [every row, only the last col]

	return x,y,labels


if __name__ == '__main__':
	#aggregate_samples('extracted_data/')
	create_ad_hoc()

