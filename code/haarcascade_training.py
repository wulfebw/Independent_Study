"""
filename: haarcascade_training.py
date: 6/14/2014
author: Blake with a lot of help
"""

import cv
import os
import sys
import cv2
import glob
import struct
import shutil
import traceback
import subprocess
import numpy as np

"""
Notes on how I trained the haar cascade classifier

(1) extract a bunch of frames using the code in this file 
(2) labeled each frame as pos/neg manually and seperated using code in this file 
(3) used "imageClipper" to crop the pos images
	(a) imageClipper is next to openCV, go to bin directory 
	(b) call ./imageClipper directory_of_pos_images
	(c) crop according to help info
(4) use the cmd line utilities provided by openCV
	(a) 

opencv_createsamples:

	opencv_createsamples -info positive_metafile.dat -w 30 -h 45 -vec samples.vec

	OR

	opencv_createsamples -img -vec -bg -h -w -num -maxxangle -maxyangle -maxzangle 


opencv_traincascade

	opencv_traincascade -data trained_classifiers/ -vec samples.vec -w 30 -h 45 -bg negative_metafile.txt -minHitRate .99 -maxFalseAlarmRate 0.5 -numPos 3000 -numNeg 2000 -numStages 12 -precalcValBufSize 1024 -precalcIdxBufSize 1024
"""


"""
Common Utilities
"""

def get_video_name_from_filename(filename):
	return os.path.splitext(os.path.basename(filename))[0]

"""
Training the actual classifier
"""

def train_haar_classifier(pos_example_directory='imageClipper', negative_example_directory='positives', classifier_output_filename='conf/my_classifiers'):
	"""
	I ended up using a script from openCV instead, but I'll leave this here for reference
	"""

	pass

"""
Extracting Vec Files
"""
def extract_vec_files_from_positive_image_directory(positive_samples_directory='imageClipper', vec_directory='frames/vec', bg_metafile='bg_metafile.txt'):
	"""
	This function creates vector files full of positive samples used for training the classifier. 

	:type positive_samples_directory: string
	:param positive_samples_directory: the dir containing the positive samples manually extracted 

	:type vec_directory: string
	:param vec_directory: the output location for all the vector files

	:type bg_metafile: string
	:param bg_metafile: this is the background meta file required for createsamples script
	"""

	files = glob.glob('{0}/*.png'.format(positive_samples_directory))
	h = 45
	w = 30
	bg = bg_metafile
	num = 100
	max_angle = 0.5
	z_max_angle = max_angle / 2
	for f in files:
		img = f
		vec = '{0}/{1}.vec'.format(vec_directory, get_video_name_from_filename(f))
		call_str = 'opencv_createsamples -img {0} -vec {1} -bg {2} -h {3} -w {4} -num {5} -maxxangle {6} -maxyangle {7} -maxzangle {8} '.format(img, vec, bg, h, w, num, max_angle, max_angle, max_angle)
		FNULL = open(os.devnull, 'w')	# add:  stdout=FNULL	when it works
		subprocess.call(call_str, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

def collect_vec_files(vec_directory='frames/vec', vec_collection_file='outputs/frames/vec_metafile.dat'):
	"""
	writes a file with a list of vector file names
	"""

	files = glob.glob('{0}/*.vec'.format(vec_directory))
	with open(vec_collection_file, 'w') as outputfile:
		for f in files:
			outputfile.write(f)
			outputfile.write('\n')

def exception_response(e):
	"""
	I don't know if this is poor style. It seems like it would be, but not able to determine that based on searching online for information.
	"""
	exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		for line in lines:
			print(line):

def merge_vec_files(vec_directory='frames/vec', output_vec_file='frames/samples.vec'):
	"""
	Iterates throught the .vec files in a directory and combines them. 

	(1) Iterates through files getting a count of the total images in the .vec files
	(2) checks that the image sizes in all files are the same

	The format of a .vec file is:

	4 bytes denoting total images (int)
	4 bytes denoting size of images (int)
	2 bytes denoting min value (short)
	2 bytes denoting max value (short)

	ex: 	6400 0000 4605 0000 0000 0000

		hex		6400 0000  	4605 0000 		0000 	0000
			   	# images  	size of h * w	min		max
		dec	    100         1350			0 		0

	:type vec_directory: string
	:param vec_directory: Name of the directory containing .vec files to be combined

	:type output_vec_file: string
	:param output_vec_file: Name of aggregate .vec file for output

	"""


	files = glob.glob('{0}/*.vec'.format(vec_directory))

	# Check to make sure there are .vec files in the directory
	if len(files) <= 0:
		print('Vec files to be mereged could not be found from directory: {0}'.format(vec_directory))
		sys.exit(1)
	# Check to make sure there are more than one .vec files
	if len(files) == 1:
		print('Only 1 vec file was found in directory: {0}. Cannot merge a single file.'.format(vec_directory))
		sys.exit(1)


	# Get the value for the first image size
	prev_image_size = 0
	try:
		with open(files[0], 'rb') as vecfile:
			content = ''.join(vecfile.readlines())
			val = struct.unpack('<iihh', content[:12])
			prev_image_size = val[1]
	except IOError as e:
		print('An IO error occured while processing the file: {0}'.format(f))
		exception_response(e)


	# Get the total number of images
	total_num_images = 0
	for f in files:
		try:
			with open(f, 'rb') as vecfile:	
				content = ''.join(vecfile.readlines())
				val = struct.unpack('<iihh', content[:12])
				num_images = val[0]
				image_size = val[1]
				if image_size != prev_image_size:
					print('The image sizes in the .vec files differ. These values must be the same.')
					print('The image size of file {0}: {1}'.format(f, image_size))
					print('The image size of previous files: {0}'.format(prev_image_size))
					sys.exit(1)

				total_num_images += num_images
		except IOError as e:
			print('An IO error occured while processing the file: {0}'.format(f))
			exception_response(e)

	
	# Iterate through the .vec files, writing their data (not the header) to the output file
	# '<iihh' means 'little endian, int, int, short, short'
	header = struct.pack('<iihh', total_num_images, image_size, 0, 0)
	try:
		with open(output_vec_file, 'wb') as outputfile:
			outputfile.write(header)

			for f in files:
				with open(f, 'rb') as vecfile:
					content = ''.join(vecfile.readlines())
					data = content[12:]
					outputfile.write(data)
	except Exception as e:
		exception_response()

		
"""
Labeling Frames as Positive/Negative
"""

def sort_frames_into_pos_neg_samples(frames_directory='outputs/frames', positive_frames_directory='frames/positives', negative_frames_directory='frames/negatives'):
	"""
	In order to train the classifier you need labeled data. I didn't have that since I was training this on the dataset I was using so I needed to create labeled data. This function helps with the first step in this process. It iterates through frames accepting user input. the input labels a frame as positive or negative and stores it depends on that designation.
	"""

	files = glob.glob('{0}/*.jpg'.format(frames_directory))
	for f in files:
		print(f)
		image = cv2.imread(f)
		cv2.imshow('f', image)
		ch = cv2.waitKey(0)

		yes_ch = 63232	# up arrow key
		no_ch = 63233	# down arrow key

		while ch != yes_ch and ch != no_ch:
				ch = cv2.waitKey(33)
				print(ch)

		video_filename = get_video_name_from_filename(f)
		if ch == yes_ch:
			destination = '{0}/{1}.jpg'.format(positive_frames_directory, video_filename)
		else:
			destination = '{0}/{1}.jpg'.format(negative_frames_directory, video_filename)
		
		shutil.move(f, destination)
			

def collect_negative_samples(negative_frames_directory='frames/negatives', negative_meta_file='frames/negative_metafile.txt'):
	"""
	Writes the names of negative files to another file for reference in createsamples
	"""

	files = glob.glob('{0}/*.jpg'.format(negative_frames_directory))
	with open(negative_meta_file, 'w') as outputfile:
		for f in files:
			outputfile.write(f)
			outputfile.write('\n')

def extract_filename_and_object_coordinates_from_filename(filename):
	"""
	from a filename like: 

		'imageClipper/0209_f07_m02_11.jpg_0000_0155_0095_0035_0054.png'

	extracts two things:
	(1) original filename
		ex: 0209_f07_m02_11.jpg
	(2) object coordinates (list of strings)
		[0155, 0095, 0035, 0054]

	"""

	raw_text = os.path.splitext(os.path.basename(filename))[0]
	jpg_ending = raw_text.find('.jpg')+4
	original_filename = raw_text[:jpg_ending]
	coordinates = raw_text[jpg_ending+1:].split('_')
	return original_filename, coordinates[1:]



def collect_positive_samples(positive_samples_directory='positives/imageClipper', positive_meta_file='frames/positive_metafile.dat'):
	"""
	Resulting file used for creation of the *.vec master file
	"""

	files = glob.glob('{0}/*.png'.format(positive_samples_directory))
	with open(positive_meta_file, 'w') as outputfile:
		for f in files:
			original_filename, coords = extract_filename_and_object_coordinates_from_filename(f)
			outputfile.write('positives/{0} 1 {1} {2} {3} {4}'.format(original_filename, coords[0], coords[1], coords[2], coords[3]))
			outputfile.write('\n')



"""
Extracting Frames From Video Files 
"""

def write_frames_to_file(frames, filename, output_directory):
	"""
	Writes a frame to a file using openCVs imwrite
	"""

	for frame_num, frame in enumerate(frames):
		cur_frame_filename = '{0}/{1}_{2}.jpg'.format(output_directory, filename, frame_num)
		cv2.imwrite(cur_frame_filename, np.array(frame))
			

def get_frames_from_video(video_filename):
	"""
	Extracts the frames from the video.
	"""

	cap = cv2.VideoCapture(video_filename)
	max_frame_index = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	frames = []

	for index in xrange(250, int(max_frame_index), 500):
		cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, index)
		
		return_value, frame = cap.read()
		if not return_value:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frames.append(gray)

	return frames

def get_frames_from_videos_in_dir(video_directory='test_data', output_directory='outputs/frames'):
	"""
	This method calls the frame extraction method on all the videos in a directory or as listed.
	"""

	files = glob.glob('{0}/*.avi'.format(video_directory)) + glob.glob('{0}/*.Avi'.format(video_directory))
	
	for f in files:
		print(f)
		filename = get_video_name_from_filename(f)
		frames = get_frames_from_video(f)
		write_frames_to_file(frames, filename, output_directory)


if __name__ == '__main__':
	#get_frames_from_videos_in_dir()
	#sort_frames_into_pos_neg_samples()
	#collect_negative_samples()
	#collect_positive_samples()
	#extract_vec_files_from_positive_image_directory()
	merge_vec_files()