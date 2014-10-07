"""
filename: outcome_features.py
date: 5/20/2014
author: Blake

Contains outcome feature functions which use the Person class. 

Outcome features are features that use information from the results of the date (hence the outcome). These are not as interesting as the other features (audio, visual, network, etc), but they are easy to calculate and kind of fun to mess around with. For example, an outcome feature might be the ratio of people choosen positively by a person to the total number. This feature might be considered a measure of how selective a person is. Obviously this isn't something you could calculate in the real world, though it's possible to argue that measures reflecting the same attributes might be useful and that these are just proxies for determining that. 

"""


import csv

def invert(num):
	"""
	Need to sum the number of people denied in one of the functions below and this is a really easy way of doing it

	:type num: int
	:param num: the value for a person's date choice that needs to be inverted
	"""

	if num is 0:
		return 1
	else:
		return 0

def get_other_id(cur_id, other_id):
	"""
	Given the current date id and the other persons id, I want to get the date id for the other person

	:type cur_id: string
	:param cur_id: the id of the current date

	:type other_id: string
	:param other_id: the id of the other person in the data e.g., 01, 06, 11, etc
	"""

	gender = cur_id[0]
	if gender == '1':
		other_gender = '2'
	else:
		other_gender = '1'

	if len(other_id) == 1:
		other_id = '0' + other_id

	return other_gender + cur_id[1:-2] + other_id

def get_date_id(cur_id, other_id):
	"""
	given the current person id and the other person id, I want to get the date id. To do this, need to determine the ordering, which is based on female/male distinction

	:type cur_id: string
	:param cur_id: the id of the current person in the date

	:type other_id: string
	:param other_id: id of the other person in the date
	"""

	if cur_id[0] == '1':
		base = cur_id
		addition = other_id[-2:]
	else:
		base = other_id
		addition = cur_id[-2:]

	return base + addition

class Person(object):
	""" 
	Class for calculating all the outcome features 


	The members of this class contain information needed to calculate outcome features. The class is used to aggregate this information over a few parses of the outcome file content. The methods then calculate the features.
	"""

	def __init__(self, id, gender, session):
		""" id = id provided in experiment """
		self.id = id
		self.n_dates = 0
		# awareness
		self.choices = dict()
		self.n_correct_predictions = 0
		self.predictions = dict()
		# confidence
		self.n_affirmative_predictions = 0
		# decisiveness
		self.n_altered = 0
		# selectivity
		self.n_denied = 0
		# for passing these values to output file
		self.gender = gender
		self.session = session

	def get_awareness(self):
		"""
		Awareness is represented here as the ratio of correct predictions to the total number of dates - How aware someone is of how the date actually went.

		This measure was not particularly helpful in predicting a persons choice in a date. 
		"""

		return round(float(self.n_correct_predictions) / self.n_dates, 4)

	def get_confidence(self):
		"""
		Confidence is represented here by the ratio of positive predictions to total dates - how confident is someone that a date is going well regardless of the actual outcome.

		In predicting a guys choice this is helpful (unsurprisingly). For predicting a girls choice it doesn't help much
		"""
		return round(float(self.n_affirmative_predictions) / self.n_dates, 4)

	def get_decisiveness(self):
		"""
		Decisiveness is a measure of how likely someone was to change their mind when given a chance to alter their choice. 

		This was not helpful at all. I'm not sure, but I think it is because (a) people generally didn't change (b) the reason people did chance seemed to be because they had said yes to "too many" people.
		"""
		return round(1 - (float(self.n_altered) / self.n_dates), 4)

	def get_selectivity(self):
		"""
		Selectivity is the ratio of denied to dates. 

		This feature was helpful, but it seems too "close" to the actual outcome to the extent that it just feels like cheating. 
		"""
		return round(float(self.n_denied) / self.n_dates, 4)

def get_metrics(inputfile='interactions.csv', outputfile='outcome_features.csv', delim=','):
	"""
	This function calculates the outcome features: 

	(1) awareness (n_correct_predictions) / n_dates)
	(2) confidence (n_affirmative_predictions) / n_dates)
	(3) decisiveness (n_predictions_altered) / n_dates)
	(4) selectivity (wasn't actually used in the end)

	:type inputfile: string
	:param inputfile: the file containing the outcomes of the dates in a specified format

	:type outputfile: string
	:param outputfile: the file for outputing the feautres
	"""

	# initialize the people and outcome dictionaries
	people = dict()
	outcomes = dict()

	# read in the data and populate the dictionaries
	rows = []
	with open(inputfile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delim)
		reader.next()
		for row in reader:
			rows.append(row)

			person_id = row[1]
			other_person_id = get_other_id(row[1], row[2])
			date_id = get_date_id (person_id, other_person_id)
			gender = row[4]
			session = row[6]
			outcome = row[8]
			outcomes[date_id] = outcome

			if person_id in people:
				cur_person = people[person_id]
			else:
				cur_person = Person(person_id, gender, session)
				people[person_id] = cur_person
			cur_person.n_dates += 1


			cur_choice_f = int(row[7])
			cur_match = int(row[8])
			cur_choice_i = int(row[9])
			cur_prediction = int(row[10])


			# update the object members using the data read in
			# update awareness
			cur_person.choices[other_person_id] = cur_choice_f
			cur_person.predictions[other_person_id] = cur_prediction
			# update confidence
			cur_person.n_affirmative_predictions += cur_prediction
			# update decisiveness
			cur_person.n_altered += abs(cur_choice_i - cur_choice_f)
			# update selectivity
			cur_person.n_denied += invert(cur_choice_f)


	# count the number of correct predictions for each person
	for cur_person_id, cur_person in people.iteritems():
		for other_person_id, prediction in cur_person.predictions.iteritems():
			other_person_choice = people[other_person_id].choices[cur_person_id]
			if other_person_choice == prediction:
				cur_person.n_correct_predictions += 1

	# write the data to the output files
	genders = ['m', 'f']
	for gender in genders:
		cur_outputfile = outputfile[:-4] + '_' + gender + outputfile[-4:]
		with open(cur_outputfile, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=delim)
			writer.writerow(['id',
							 'awareness_m', 
							 'confidence_m', 
							 'decisiveness_m', 
							 'selectivity_m', 
							 'awareness_f', 
							 'confidence_f', 
							 'decisiveness_f', 
							 'selectivity_f', 
							 'outcome',
							 ])
			for person_id, person in people.iteritems():
				for other_person_id, _ in person.choices.iteritems():
					tot_id = person_id + '_' + other_person_id
					if gender == 'f':
						gender_num = '2'
					else:
						gender_num = '1'
					if person.gender == gender_num:
						other_person = people[other_person_id]
						outcome = person.choices[other_person_id]
						row = [	tot_id,
								person.get_awareness(),
								person.get_confidence(),
								person.get_decisiveness(),
								person.get_selectivity(),
								other_person.get_awareness(),
								other_person.get_confidence(),
								other_person.get_decisiveness(),
								other_person.get_selectivity(),
								outcome]
						writer.writerow(row)

if __name__ == '__main__':
	#get_metrics()

