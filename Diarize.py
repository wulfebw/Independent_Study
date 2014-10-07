""" 
filename: Diarize.py
date: 3/20/2014
author: Blake

This file contains code used for speaker diarization. This is definitely not the best way to segment 
audio by speaker, but it is the way I did it given constraints. Rather than do it this way,
it is much better to use one of the available spkr diarization libraries, which use actual
DSP methods.

Here are a few points about this method:

(1) We have available 2 audio files of the same conversation but recorded from different locations
(2) The differing locations means that each speaker will be louder in one
(3) There is a lot of background conversation in both files
(4) This background conversation should be of roughly equivalent volume in both recordings, 
	but the actual conversation should not
(5) So, by subtracting one recordings sound energy from the other, we should emliminate the noise 
	and keep the correct speech in each file. In practice it doesn't always work, but that's the logic.

Here are the steps:

(1) Extracting sound energy from both speed-date audio files
(2) Subtracting one energy file from the other
(3) In the resulting file, if the energy is above or below a certain threshold it is considered speech
(4) Components _not_ considered speech are then removed by applying a mask derived from the energy files
"""


import sys
import math
import traceback
import numpy as np
import os.path
import itertools
from itertools import izip_longest
from scipy.io import wavfile

from LowLevelDescriptor import LowLevelDescriptor as LLD
from file_utils import graph_values, get_args, remove_extension, get_file_name

# constants
openSMILE_DIR = '../../../opensmile-2.0-rc1/opensmile/'
FEATURE_NAME = 'pcm_loudness_sma'		# voiceProb_sma		pcm_loudness_sma	F0_sma
FEATURE_TYPE = 'pcm_loudness_sma'
SMOOTHING = 150
WINDOW_SIZE = 60
GRAPH = False

def get_positives_smooth(arr, threshold=0, smoothing=SMOOTHING):
	""" 
	In extracting periods where a speaker is talking, if you just take the times
	above the threshold you end up with very sporadic speech. This is an 
	extremely rudimentary method of smoothing out the speech. When it encounters
	speech, it assumes that the next SMOOTHING frames are also speech and 
	adds them to the extraction mask.

	:type arr: numpy.array 
    :param arr: an array of speech energy values 

    :type threshold: float
    :param threshold: the min value for a frame to be considered speech

    :type smoothing: int
    :param smoothing: smoothing factor - number of frames assumed speech after one is detected
	"""

	times = np.zeros(len(arr))
	i = 0
	while(i < len(arr)):
		if arr[i] > threshold:
			times[i:i + 1 + smoothing] = 1
			i += smoothing + 1
		else:
			i += 1

	return times

def get_lower_threshold(values):
	""" 
	Based on "values" determines where to set the threshold at which we assume
	the speaker is actually speaking. I tried setting the threshold a few different
	ways as shown in the commented out section.

	:type values: numpy.array
	:param values: the energy values
	"""

	# energy
	# values = np.maximum(values, np.median(values))
	# return np.median(values) + 3*np.std(values)

	# F0 
	# return np.median(values)

	# voicing
	return np.mean(values) # + .5*np.std(values)

def get_expanded_mask(original_mask, num_frames):
	""" 
	Expands the mask to the proper length by repeating 0s or 1s appropriately. This is needed
	because the energy values are extracted for windows defined by openSMILE. To apply the 
	mask to the soundfile, it needs to be expanded to the soundfile length.

	:type original_mask: numpy.array
	:param original_mask: the original mask values (0s or 1s denoting speech/nonspeech)

	:type num_frames: int
	:param num_frames: the number of frames to be expanded to.
	"""

	frames_per_value = float(num_frames) / float(len(original_mask))
	floor = math.floor(frames_per_value)
	diff = frames_per_value - floor

	# expand the mask to correct length
	expanded_mask = []
	diff_sum = 0
	repetitions = int(floor)
	for val in original_mask:
		diff_sum += diff
		if diff_sum > 1:
			expanded_mask.append(val)
			diff_sum = 0
		expanded_mask += [val] * repetitions

	# need to fix mask_len != num_frames, but for now:
	difference = num_frames - len(expanded_mask) 
	if difference > 0:
		zeros = [0] * difference
		expanded_mask += zeros
	elif difference == 0:
		pass # checking
	else:
		raise IndexError

	return expanded_mask

def extract_segments(input_file, output_file, mask):
	""" 
	Creates a new file with just the desired speakers segments (theoretically at least).

	:type input_file: string (filename)
	:param input_file: the file containing the original audio

	:type output_file: string (filename)
	:param output_file: the file to which the new sound should be written

	:type mask: numpy.array 
	:param mask: array containing 0s or 1s representing nonspeech/speech
	"""

	try:
		samp_rate, input_wav = wavfile.read(input_file)
		expanded_mask = get_expanded_mask(mask, len(input_wav))
		output_wav = []
		silence = np.int16(0)

		for index, frame in enumerate(input_wav):
			if expanded_mask[index] == 1.0:
				output_wav.append(frame)
			else:
				output_wav.append(silence)

		wavfile.write(output_file, samp_rate, np.array(output_wav))

	except Exception as e:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		print ''.join('!! ' + line for line in lines)  # Log it or whatever here
		sys.exit()


def window_max(values, window_len=WINDOW_SIZE):
	""" 
	When you take two audio files of roughly the same conversation and subtract them, it results in 
	sporadic speech. To combat this you can just level out each file's energy values over a certain
	range to the max in the range. This reduces the impact of misaligned files and misaligned timing.

	:type values: numpy.array
	:param values: list of energy values

	:type window_len: int
	:param window_len: size of window to take max from
	"""

	window_values = izip_longest(*(iter(values),) * window_len)
	new_values = []
	for window in window_values:
		max_val = max(window)
		for val in window:
			new_values.append(max_val)
	return np.array(new_values)
	
def audio_subtract(values_1, values_2):
	""" 
	Subtracts one files energy values from another. In this version, the max from each window is 
	used. In the code below, I subtract v_1 from v_2 and v_2 from v_1. This is not necessary, but
	I do it anyway because it makes the rest of the process simplier. 
	
	:type values_*: numpy.array
	:param values_*: list of energy values
	"""

	window_max_1 = window_max(values_1)
	window_max_2 = window_max(values_2)

	return window_max_1 - window_max_2

def convert_to_speaker_segments(input_file_1, input_file_2, output_file_1, output_file_2, config_file):
	""" 
	This is the method running the show. It essentially runs through the above methods calling them
	in order. 

	:type input_file_*: string
	:param input_file_*: name of the two audio files to be used as input

	:type output_file_*: string
	:type output_file_*: name of the output files

	:type config_file: string (filename)
	:param config_file: the openSMILE config file to use 
	"""
	
	# get the name of related files
	filename_1 = remove_extension(get_file_name(input_file_1))
	filename_2 = remove_extension(get_file_name(input_file_2))

	path = os.path.dirname(input_file_1)
	extraction_dir = '{0}/extraction'.format(path)

	# extract the energy and place in EXTRACTION_DIR for use in a bit
	extract_file_1 = '{0}/{1}.{2}.csv'.format(extraction_dir, filename_1, FEATURE_TYPE)
	extract_file_2 = '{0}/{1}.{2}.csv'.format(extraction_dir, filename_2, FEATURE_TYPE)
	if not os.path.isfile(extract_file_1):
		LLD.extract_from_sound_file(input_file_1, extract_file_1, config_file, openSMILE_DIR)
	if not os.path.isfile(extract_file_2):
		LLD.extract_from_sound_file(input_file_2, extract_file_2, config_file, openSMILE_DIR)

	# load them back in either with load_from_csv() or load_from_text_grid()
	# this is necessary because processing these files without an intermediate step 
	# leads to issues with only partially processed file sets
	feature_1 = LLD.load_from_csv(FEATURE_NAME, extract_file_1)
	feature_2 = LLD.load_from_csv(FEATURE_NAME, extract_file_2)

	# graph the feature values
	if GRAPH:
		feature_1.graph_values('graphs', tag='_1')
		feature_2.graph_values('graphs', tag='_2')

	# remove initial silence
	feature_1.remove_outliers()
	feature_2.remove_outliers()

	# truncate longer file
	diff = len(feature_1.values) - len(feature_2.values)
	if diff == 0:
		pass
	elif diff > 0:		# feature_1 is longer
		feature_1.values = feature_1.values[:-diff]
	else:				# feature_2 is longer, add to 1
		feature_2.values = feature_2.values[:diff]

	# subtract
	# it's possible to just take the inverse of one to get the other, but 
	# this is more obvious what's happening so I'll stick with it for now
	feature_1_minus_2 = audio_subtract(feature_1.values, feature_2.values)
	feature_2_minus_1 = audio_subtract(feature_2.values, feature_1.values)

	# graph scaled, normalized feature values along with the difference
	if GRAPH:
		feature_1.graph_values('graphs', tag='_scaled_norm_1')
		feature_2.graph_values('graphs', tag='_scaled_norm_2')
		graph_values('i1_minus_i2', feature_1_minus_2, 'graphs', tag='_1')
		graph_values('i2_minus_i1', feature_2_minus_1, 'graphs', tag='_2')

	# get the threshold at which we assume person is speaking
	threshold_1 = get_lower_threshold(feature_1_minus_2)
	threshold_2 = get_lower_threshold(feature_2_minus_1)

	# getting time when person is talking
	feature_1_positive_times = get_positives_smooth(feature_1_minus_2, threshold_1)
	feature_2_positive_times = get_positives_smooth(feature_2_minus_1, threshold_2)

	# actually extract the segments and save to file
 	extract_segments(input_file_1, output_file_1, feature_1_positive_times)
	extract_segments(input_file_2, output_file_2, feature_2_positive_times)
