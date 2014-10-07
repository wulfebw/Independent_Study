""" 
filename: LowLevelDescriptor.py
date: 3/20/2014
author: technically Blake, but really stackoverflow
"""

import sys
import re
import os
import wave
import csv
import numpy as np
import pylab
import traceback
import matplotlib.pyplot as plt
import subprocess

# text_grid constants
SKIP_LINES = 15

""" helper methods and Classes """

class Feature(object):
	def __init__(self, name):
		self.name = name
		self.values = []

def valid_filename(filename, extension):
	""" Makes sure the filename ends in the correct extension """

	ext = filename[-len(extension):]
	return ext == extension


class LowLevelDescriptor(object):
	""" 
	The feature class represents a single feature.
	It is purely for low level descriptor features. 
	"""
	

	def __init__(self, name, values=[]):
		""" Default constructor used if values already extracted and loaded """

		self.name = name
		self.values = np.array(values)


	@classmethod
	def load_from_text_grid(cls, name, input_file):
		""" load_from_text_grid used if feature has been extracted and saved as TextGrid """
		
		values = []
		try:
			with open(input_file, 'rb') as fin:
				for x in xrange(SKIP_LINES):
					fin.readline()
				for line in fin:
					start = re.search('=', line).start() + 1
					end = start + 8
					values.append(float(line[start:end]))
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
			print(lines)
			sys.exit()
		return cls(name, values)


	@classmethod
	def load_from_csv(cls, lld_name, input_file):
		""" 
		load_from_csv used if feature has been extracted and saved in csv 
		This method actually extracts everything from the file, but at the
		end only returns the desired feature. It's less efficient, but easier.
		It uses the Feature class as a way to keep values paired with names.
		"""

		feature_list = [] 
		try:
		# open file and read in
			with open(input_file, 'rb') as f:
				reader = csv.reader(f, delimiter=';')
				try:
					# create each feature first
					names = reader.next()
					for name in names:
						feature_list.append(Feature(name))
					# read in the values for each row and append each feature's value to its values
					for row in reader:
						for val, feature in zip(row, feature_list):
							feature.values.append(float(val))

				except csv.Error as e:
					exc_type, exc_value, exc_traceback = sys.exc_info()
					lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
					print(lines)
					sys.exit()
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
			for line in lines:
				print(line)
			sys.exit()

		# get this feature from the list
		# print(feature_list)
		index = names.index(lld_name)
		lld = feature_list[index]
		return cls(lld.name, lld.values)


	@staticmethod
	def extract_from_sound_file(input_file, output_file, config_file, openSMILE_path = '../../../../opensmile-2.0-rc1/opensmile/'):
		""" 
		extract_from_sound_file used if no values extracted yet from audio file.
		It has to be the correct type of wav file, sorry.
		
		automates the openSMILE extraction

		This method overwrites files.

		openSMILE_path should end with a forward slash 
		"""

		c = config_file
		i = input_file
		o = output_file
		p = openSMILE_path

		if not valid_filename(c, '.conf'):
			raise Exception("Invalid config: {0}".format(c))
		if not valid_filename(i, '.wav'):
			raise Exception("Invalid input file: {0}".format(i))
		if not (valid_filename(o, '.csv') or valid_filename(o, '.arff')):
			raise Exception("Invalid output file: {0}".format(o))

		call_str = 'SMILExtract -C {0} -I {1} -O {2}'.format(c, i, o)
		FNULL = open(os.devnull, 'w')	# add:  stdout=FNULL	when it works
		subprocess.call(call_str, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
		return o


	@staticmethod
	def extract_audio_from_video(input_file, transition_file, output_file):
		""" extract_audio_from_video extracts audio from video using ffmpeg """

		call_str = 'ffmpeg -i {0} -vn {1}'.format(input_file, transition_file)
		subprocess.call(call_str, shell=True)

		call_str = 'sox {0} {1}'.format(transition_file, output_file)
		subprocess.call(call_str, shell=True)



	def graph_values(self, output_dir, tag='', y_scale=1.5, img_extension='gif'):
		""" graph the LLD's values """

		plt.plot(self.values)
		plt.title(self.name)
		low = self.values.min() * y_scale
		high = self.values.max() * y_scale
		pylab.ylim([low,high])
		plt.savefig('{0}/{1}{2}.{3}'.format(output_dir, self.name, tag, img_extension))
		plt.clf()


	def remove_outliers(self, num_std=5):
		""" removes values num_std std deviations away from the mean """

		std = np.std(self.values)
		mean = np.mean(self.values)
		min_val = mean - num_std*std
		max_val = mean + num_std*std
		self.values = np.clip(self.values, min_val, max_val)
