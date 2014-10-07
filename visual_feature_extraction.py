"""
filename: visual_feature_extraction.py
date: 6/15/2014
author: Blake

This code computes a variety of visual features:
 	(1) velocity, acceleration, and std dev of Body coordinates for each frame 
 	(2) Motion template image entropy and mean value 
 	(3) How much a person smiles during a date

Features are also calculated relative to individual norms in certain places. 
"""

import glob
import csv
import os
import sys
import cv2
import traceback
import numpy as np 
import math

from skimage.filter.rank import entropy
from skimage.morphology import disk

from object_detection import reduce_region
from file_utils import load_coordinates_from_csv, format_data_value, inverse, get_gender, put_female_first, put_male_first, load_date_choices_dict, get_output_filename, write_features_to_file, get_video_name_from_filename, get_video_key_from_filename, write_video_features_to_file, aggregate_motion_files, memberwise_add

np.set_printoptions(threshold=5000)
DEFAULT_THRESHOLD = 32

MOTION_LABELS = ['date_id', 'm_relative_entropy', 'm_relative_mean_motion','f_relative_entropy', 'f_relative_mean_motion', 'choice']
SMILE_LABELS = ['date_id', 'm_log_smile_count', 'f_log_smile_count']


def trimmed_std(values):
	"""
	Made a trimmed standard deviation
	"""
	mean = np.mean(values)
	std = np.std(values)
	lower_limit = mean - std*2
	upper_limit = mean + std*2
	limits = (lower_limit, upper_limit)
	return np.std(values[(values > lower_limit) & (values < upper_limit)])

def trimmed_mean(values):
	"""
	Trimmed mean as well
	"""
	mean = np.mean(values)
	std = np.std(values)
	lower_limit = mean - std*2
	upper_limit = mean + std*2
	limits = (lower_limit, upper_limit)
	return np.mean(values[(values > lower_limit) & (values < upper_limit)])

def get_acceleration_mean_and_std(velocities):
	"""
	Calculates the mean values and standard deviation of the values for "acceleration" - how quickly the point velocities change
	"""
	acceleration_between_consecutive_points = []
	for index in range(1,len(velocities)):
		prev_value = velocities[index-1]
		cur_value = velocities[index]
		acceleration = abs(cur_value - prev_value)
		acceleration_between_consecutive_points.append(acceleration)
	acceleration_between_consecutive_points = np.array(acceleration_between_consecutive_points)
	mean = trimmed_mean(acceleration_between_consecutive_points)
	std = trimmed_std(acceleration_between_consecutive_points)
	return mean, std

def get_velocities(values):
	"""
	returns the point velocity values for the object frame
	"""

	velocities = []
	for row_index in range(1,len(values)):
		prev_values = values[row_index-1]
		cur_values = values[row_index]
		x_diff = (cur_values[1] - prev_values[1])
		y_diff = (cur_values[2] - prev_values[2])
		velocities.append(math.sqrt(x_diff**2 + y_diff**2))
		
	return np.array(velocities)

def get_metric_diff(values, metric_func):
	"""
	determines the difference in the values based on a certain function over the course of the date
	"""
	first_third = len(values) / 3
	first_third_val = metric_func(values[:first_third])
	third_third_val = metric_func(values[first_third*2:])
	diff = third_third_val - first_third_val
	return diff

def create_video_features_dict(coordinates_dir='outputs/coordinates'):
	"""
	This function calculates the basic visual features I started with, for example the std dev of position, velocity, acceleration and functionals on those values of a persons body during a date.

	These features were unsurprisingly not particularly helpful, but it provided a nice starting point. 
	"""
	files = glob.glob('{0}/*.csv'.format(coordinates_dir))
	video_features_dict = dict()


	for f in files:
		print(f)

		coordinate_values = load_coordinates_from_csv(f)

		x_position_std = trimmed_std(np.array(coordinate_values[:,1]))
		y_position_std = trimmed_std(coordinate_values[:,2])
		print('x_pos: {0}'.format(x_position_std))
		print('y_pos: {0}'.format(y_position_std))

		diff_x_std_btwn_portions = get_metric_diff(np.array(coordinate_values[:,1]), trimmed_mean)
		diff_y_std_btwn_portions = get_metric_diff(coordinate_values[:,2], trimmed_mean)
		print('diff x std: {0}'.format(diff_x_std_btwn_portions))
		print('diff y std: {0}'.format(diff_y_std_btwn_portions))

		# velocity
		velocities = get_velocities(coordinate_values)
		vel_mean = trimmed_mean(velocities)
		vel_std = trimmed_std(velocities)
		print('vel_mean: {0}'.format(vel_mean))
		print('vel_std: {0}'.format(vel_std))

		# accel
		accel_mean, accel_std = get_acceleration_mean_and_std(velocities)
		print('accel_mean: {0}'.format(accel_mean))
		print('accel_std: {0}'.format(accel_std))

		diff_std_vel_btwn_portions = get_metric_diff(velocities, trimmed_std)
		print('vel_diff: {0}'.format(diff_std_vel_btwn_portions))

		# key
		key = get_video_key_from_filename(f)
		# add to dict
		feature_row = [x_position_std, y_position_std, diff_x_std_btwn_portions, diff_y_std_btwn_portions, vel_mean, vel_std, accel_mean, accel_std, diff_std_vel_btwn_portions]
		video_features_dict[key] = feature_row

	return video_features_dict

def create_date_features_dict(video_features_dict):
	"""
	once the video features are calculated, they need to be associated with a specific date. That's what this function does
	"""

	date_features_dict = dict()

	for video_name, features in video_features_dict.iteritems():
		date_key, gender = put_female_first(video_name)

		num_features = len(features)
		if date_key not in date_features_dict:
			date_features_dict[date_key] = list(range(1 + 2*num_features))
			date_features_dict[date_key][0] = date_key

		if gender == 'm':
			date_features_dict[date_key][1:num_features+1] = features
		else: 
			date_features_dict[date_key][1+num_features:] = features

	dict_without_unpaired_dates = dict()
	for date_key, features in  date_features_dict.iteritems():
		if not 1 in features: # and 2 in features and 3 in features):
			dict_without_unpaired_dates[date_key] = features
	return dict_without_unpaired_dates

def is_female_first(string):
	"""
	simple helper to tell if the date name provided represents a video of a male or female
	"""

	if string.find('f') < string.find('m'):
		return True
	else:
		return False

def aggregate_features_and_choices(date_features_dict, date_choices_dict):
	"""
	In order to associate the features with the out come or target (in this case the choices), the two must be associated. This function does that.
	"""

	m_values_to_write = []
	f_values_to_write = []

	for date_id, features in date_features_dict.iteritems():
		# w/ segments date_choice_key = date_id[:-2]
		date_choice_key = date_id # w/o segments 
		if date_choice_key in date_choices_dict:
			f_values_to_write.append(features + [date_choices_dict[date_choice_key]])
		male_choice_key = put_male_first(date_choice_key)[0]
		if male_choice_key in date_choices_dict:
			m_values_to_write.append(features + [date_choices_dict[male_choice_key]])

	return m_values_to_write, f_values_to_write


def get_entropy_and_mean_motion_temp(videofile):
	"""
	After messing around with detecting the full body coordinates, I moved to simply calculating aggregate measures of the amount of motion by individuals during a date. This is better because it accounts for factors like head movement or gestures. 
	It still didn't help much though because how much a person moves during a date doesn't seem to provide much information about how the date will turn out.
	"""

	# input video to use (called cam b/c I'm lazy, actually a cap "capture")
	cam = cv2.VideoCapture(videofile)
	# read the first frame to initialize
	if cam.isOpened():
		print('open')
	else:
		print('not open')
	ret, frame = cam.read()
	if not ret:
		sys.exit(1)

	# get height and width
	h, w = reduce_region(frame).shape[:2]
	# init the prev frame
	prev_frame = reduce_region(frame.copy())
	# init motion his to an array of zeros of size (h,w)
	motion_history = np.zeros((h, w), np.float32)
	sum_history = np.zeros((h, w), np.float32)

	hsv = np.zeros((h, w, 3), np.uint8)
	hsv[:,:,1] = 255

	# iterate through the frames
	frame_num = 0
	while True:

		# read in a frame
		ret, frame = cam.read()
		frame_num += 1
		if not ret:
			break

		if frame_num % 5 == 0:
			frame = reduce_region(frame)

			# calc the difference between this frame and the prev one
			frame_diff = cv2.absdiff(frame, prev_frame)

			# convert to gray
			gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

			# get the threshold from the user window
			thrs = DEFAULT_THRESHOLD

			# get the motion_mask value to use based upon the threshold and diff
			ret, motion_mask = cv2.threshold(gray_diff, thrs, 1, cv2.THRESH_BINARY)
			sum_history += motion_mask

			prev_frame = frame.copy()

	cam.release()
	cv2.destroyAllWindows()

	history_max = sum_history.max(axis=1).max()
	entropy_img = entropy(sum_history/history_max, disk(5))
	entropy_sum = entropy_img.sum(axis=1).sum()

	motion_template_mean = sum_history.mean(axis=1).mean()

	print('sum'),
	print(entropy_sum)
	print('mean'),
	print(motion_template_mean)

	return entropy_sum, motion_template_mean


def create_video_features_dict_motion_temp(video_input_directory):
	"""
	This is a helper method which calls the motion template method and associates its outcome value with a specific date
	"""

	files = glob.glob('{0}/*.avi'.format(video_input_directory)) + glob.glob('{0}/*.Avi'.format(video_input_directory))
	files = np.random.permutation(files)
	video_features_dict = dict()
	num_existing = 0
	base_existing_files = []
	for f in files:

		existing_files = glob.glob('/home/ec2-user/Dropbox/outputs/motion/*.csv')
		if num_existing != len(existing_files):
			num_existing = len(existing_files)
			base_existing_files = []
			for filename in existing_files:
				base_existing_files.append(get_video_name_from_filename(filename))

		videoname = get_video_name_from_filename(f)
		if videoname not in base_existing_files:
			print(f)

			try:
				entropy_sum, motion_template_mean = get_entropy_and_mean_motion_temp(f)
				# key
				key = get_video_key_from_filename(f)
				# add to dict
				feature_row = [entropy_sum, motion_template_mean]
				video_features_dict[key] = feature_row
				write_video_features_to_file(key, [key] + feature_row)
			except Exception as e:
				print('{0} is an invalid file'.format(videoname))
				print('!! filename: {0}\n'.format(f))
				exc_type, exc_value, exc_traceback = sys.exc_info()
				lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
				for line in lines:
					print(line)


	return video_features_dict

def create_video_features_dict_smile(smile_coordinates_input_directory):
	"""
	After realizing that how much people moved didn't matter, I switched to something that seemed to obviously be important - how much they smiled during the date. The actual code for extracting smiles is eleswhere and consists both of (a) extracting frames where people are smiling (b) developing the classifier which looks for smiles specifically trained on the dataset used here. 
	"""

	files = glob.glob('{0}/*.csv'.format(smile_coordinates_input_directory))
	video_features_dict = dict()

	for f in files:
		smile_count = 0
		key = get_video_key_from_filename(f)
		with open(f, 'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				if row[1] != '0':
					smile_count += 1
			print(smile_count)
			video_features_dict[key] = [math.log(smile_count)]

	return video_features_dict


def extract_relative_features(samples_file='outputs/features/smile_samples_f.csv', output_samples_file='outputs/features/relative_smile_samples_f.csv'):
	"""
	The use of movement features made me realize that the absolute amount of movement (or other quantities) was not that important. What might be more important was how much a person moved (or displayed other traits) relative to their own personal norm. This function computes this value by taking e.g., the average amount of movement and then subtracting that from the movement in each date individually. This give a relative measure of how much a person is moving (relative to their usual behavior).

	This type of feature worked well. While it's unrealistic for a lot of scenarios, for this one it actually is pretty plausible. 
	"""


	females = dict()
	males = dict()
	dates = dict()

	with open(samples_file, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		reader.next()
		for row in reader:

			if '3' in row or '2' in row or '1' in row:
				continue

			row = [row[0]] + map(float,row[1:-1]) + [int(row[-1])]
			dates[row[0]] = row[1:]

			f_id = row[0][:8]
			m_id = row[0][:5] + row[0][8:]

			f_features_start = 2			# 3
			f_features_end = 3			# 5
			m_features_start = 1			# 1
			m_features_end = 2			#3

			if f_id not in females:
				females[f_id] = [row[f_features_start:f_features_end]]
			else:
				females[f_id] += [row[f_features_start:f_features_end]]

			if m_id not in males:
				males[m_id] = [row[m_features_start:m_features_end]]
			else:
				males[m_id] += [row[m_features_start:m_features_end]]

	avg_values_by_female = dict()
	for f_id, values in females.iteritems():
		avg_values_by_female[f_id] = np.array(reduce(memberwise_add, values)) / len(values)

	avg_values_by_male = dict()
	for m_id, values in males.iteritems():
		avg_values_by_male[m_id] = np.array(reduce(memberwise_add, values)) / len(values)


	for date_id, values in dates.iteritems():
		f_id = date_id[:8]
		m_id = date_id[:5] + date_id[8:]

		try:
			f_values = avg_values_by_female[f_id]
			m_values = avg_values_by_male[m_id]
			avg_values = np.concatenate((m_values, f_values), axis=0)

			dates[date_id] = map(sub, values[:-1], avg_values) + [values[-1]]

		except KeyError as e:

			print('female key: {0}'.format(f_id))
			print('male key: {0}'.format(m_id))
			print(e)
			raise(e)
	labels = SMILE_LABELS
	values = [[key] + values for key, values in dates.iteritems()]
	write_features_to_file(values, output_samples_file, labels)




def visual_feature_extraction_driver(coordinates_dir='coordinates', m_feature_output_filename='/home/ec2-user/Dropbox/outputs/features/smile_samples_m.csv', f_feature_output_filename='/home/ec2-user/Dropbox/outputs/features/smile_samples_f.csv', outcome_file='/home/ec2-user/Dropbox/interactions.csv'):
	"""
	drives the whole process
	"""

	#video_features_dict = create_video_features_dict(coordinates_dir)
	video_features_dict = create_video_features_dict_motion_temp('/home/ec2-user/Dropbox/motion_copies')
	#video_features_dict = aggregate_motion_files()
	#video_features_dict = create_video_features_dict_smile(coordinates_dir)

	date_features_dict = create_date_features_dict(video_features_dict)
	date_choices_dict = load_date_choices_dict(outcome_file)
	m_values_to_write, f_values_to_write  = aggregate_features_and_choices(date_features_dict, date_choices_dict)
	labels = MOTION_LABELS
	write_features_to_file(m_values_to_write, m_feature_output_filename, labels)
	write_features_to_file(f_values_to_write, f_feature_output_filename, labels)


if __name__ == '__main__':
	visual_feature_extraction_driver()
	#extract_relative_features()


