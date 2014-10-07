"""
filename: graph_ssp.py
date: 4/21/2014
author: Blake

This file contains logic for extracting network features. The approach implemented here is described 
in this paper: http://cs.stanford.edu/~emmap1/ngpaper.pdf
"""


import csv
import networkx as nx 
import sys
import traceback
import numpy as np
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)
DRAWING = True

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

def format_id_num(id_num):
	""" 
	the id numbers are either one or two digits. If it is one digit, it needs to have a '0'
	prepended to it for later use.

	:type id_num: string
	:param id_num: the id number of an individual, either one or two digits
	"""

	if len(id_num) != 2:
		return '0' + id_num
	else:
		return id_num

def get_edge_dest(row):
	"""
	This method parses the second value ([1]) listed in a csv row and combines it with other info to
	determine who the other individual in a date was. 

	:type row: list of strings
	:param row: a row from the csv file. row[1] refers to the other individual in the speed-date
	"""

	gender = inverse(row[1][0])
	session = row[1][1:5]
	id_num = format_id_num(row[2])
	return gender + session + id_num


def load_graphs_from_csv(inputfile, delim=','):
	""" 
	This method loads session graphs from the csv file and returns a dictionary of 
	{session_name: session graph}. It does the following:

	(1) read in a list of nodes and list of sessions from the file
	(2) create the dictionary with the session names and empty graphs
	(3) read the file again, this time populating the graphs

	Could store all the rows in memory, but this was easier.

	:type inputfile: string
	:param inputfile: a csv file containing outcome information

	:type delim: string
	:param delim: delimiter used in the csv file
	"""

	session_graphs = dict()

	# read in nodes and sessions
	session_list = []
	node_list = []
	with open(inputfile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delim)
		reader.next()
		for row in reader:
			node = row[1]
			node_list.append(node)
			session = node[1:5]
			session_list.append(session)
			
	# create dictionary with the <key, value> = <session, graph>		
	graph_dict = dict()
	for session in session_list:
		if session in graph_dict:
			pass
		else:
			new_graph = nx.DiGraph(session=session)
			graph_dict[session] = new_graph

	# add nodes to the graphs
	for node in node_list:
		session = node[1:5]
		if session in graph_dict:
			gender = node[0]
			graph_dict[session].add_node(node, bipartite=gender, session=session)

	# read in data again to populate graphs
	# sure there's a more efficient way, but I'm in a hurry!
	with open(inputfile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delim)
		reader.next()
		for row in reader:
			if row[7] == '1':
				edge_origin = row[1]
				edge_dest = get_edge_dest(row) 
				session = row[1][1:5]
				graph = graph_dict[session]
				graph.add_edge(edge_origin, edge_dest)

	return graph_dict

""" Primary Functions """

"""
The next few functions are used to project the directed bipartite graph. There is a built in 
function in networkx called "generic_weighted_projected_graph", but I couldn't get it working 
and made some functions to do it. 

Note that this does not perform a true projection - it just does it for one node and leaves
the other nodes without connecting edges.
"""

def like_in_common_weight(DB, u, v):
	""" 
	This function returns the number of common edges between u and v in the graph "people of other
	gender _liked_ by node_x". The number of edges is the weight used in the resulting weighted, 
	undirected projection.

	:type DB (Directed Bipartite Graph): networkx.DiGraph
	:param DB (Directed Bipartite Graph): This is the DB graph representing date outcomes

	:type u, v: networkx.node 
	:param u, v: These are the nodes for which the number of common weights are determined

	"""

	u_edges = set(DB[u])
	v_edges = set(DB[v])
	common = u_edges.intersection(v_edges)
	return len(common)

def get_like_in_common_graph(DB, node, gender):
	"""
	This function performs the actual projection of the directed bipartite graph. The 
	result is an weighted, undirected graph that contains nodes of only one gender. The projected graph 
	represents being liking individuals of the other gender in common. 

	:type DB (Directed Bipartite Graph): networkx.DiGraph
	:param DB (Directed Bipartite Graph): This is the DB graph representing date outcomes

	:type node: networkx.node
	:param node: this is the node around

	:type gender: string
	:param gender: the gender of the current node 
	"""

	G = nx.Graph()
	G.add_node(node)
	nodes = set(n for n,d in DB.nodes(data=True) if d['bipartite']==gender)
	for cur_node in nodes:
		if cur_node is not node:
			G.add_node(node)
			edge_weight = like_in_common_weight(DB, node, cur_node)
			G.add_edge(cur_node, node, weight=edge_weight)
	return G

def like_by_in_common_weight(DB, u, v):
	""" 
	This function returns the number of common edges between u and v in the graph "people of other
	gender that like node_x". The number of edges is the weight used in the resulting weighted, 
	undirected projection.

	:type DB (Directed Bipartite Graph): networkx.DiGraph
	:param DB (Directed Bipartite Graph): This is the DB graph representing date outcomes

	:type u, v: networkx.node 
	:param u, v: These are the nodes for which the number of common weights are determined

	"""

	u_edges = set(DB.predecessors(u))
	v_edges = set(DB.predecessors(v))
	common = u_edges.intersection(v_edges)
	return len(common)

def get_liked_by_in_common_graph(DB, node, gender):
	""" 
	This is a redundant function that does the same thing as "get_like_in_common_graph".


	This function performs the actual projection of the directed bipartite graph. The 
	result is an weighted, undirected graph that contains nodes of only one gender. The projected graph 
	represents being liked by individuals of the other gender in common.

	:type DB (Directed Bipartite Graph): networkx.DiGraph
	:param DB (Directed Bipartite Graph): This is the DB graph representing date outcomes

	:type node: networkx.node
	:param node: this is the node around

	:type gender: string
	:param gender: the gender of the current node 
	"""

	G = nx.Graph()
	G.add_node(node)
	nodes = set(n for n,d in DB.nodes(data=True) if d['bipartite']==gender)
	for cur_node in nodes:
		if cur_node is not node:
			G.add_node(node)
			edge_weight = like_by_in_common_weight(DB, node, cur_node)
			G.add_edge(cur_node, node, weight=edge_weight)
	return G

def get_projected_graphs(graph, row):
	""" 
	This function calls the above functions to create the four projected graphs. 
	There are two of each type for each gender. This entire approach is not very 
	efficient. It would be better to only create 2 projected graphs for each session 
	(1 male and 1 female), rather than making one for each row. Once again this was
	easier and didn't add much time to the entire process (which I only ran once).

	:type graph (Directed Bipartite Graph): networkx.DiGraph
	:param graph (Directed Bipartite Graph): This is the DB graph representing date outcomes

	:type row: list of strings
	:param row: a row from the file representing a single date from a single individual
	"""

	this_node = row[1]
	other_node = get_edge_dest(row)

	graph_1 = get_like_in_common_graph(graph, this_node, '1')
	graph_2 = get_liked_by_in_common_graph(graph, this_node, '1')
	graph_3 = get_like_in_common_graph(graph, other_node, '2')
	graph_4 = get_liked_by_in_common_graph(graph, other_node, '2')

	return graph_1, graph_2, graph_3, graph_4

def get_weight(projection, edge_dest, edge_origin):
	"""
	This function gets the weight between two nodes for use in determining which weight to use
	as the feature. The reason this is a try statement is that some edges don't exist in 
	the actual file and it was easiest way to handle that. 

	:type projection: networkx.Graph
	:param projection: the projected from which the weight is retrieved

	:type edge_dest: networkx.node
	:param edge_dest: The current male/female node (which is the only node in this particular 
	projected graph with incoming edges)

	:type edge_origin: networkx.node
	:param edge_origin: node of one of the other participants of congruent geneder to the 
	participant for whom the features are being developed
	"""

	try:
		weight = projection[edge_origin][edge_dest]['weight']
	except Exception as e:
		weight = 0
	return weight

def get_value(orig_graph, edge_origin, edge_dest):
	"""
	Returns the value of having an edge or not having an edge between two members of opposite
	gender. +1 or -1

	:type orig_graph: networkx.DiGraph
	:param orig_graph: the directed bipartite graph

	:type edge_origin: networkx.node
	:param edge_origin: node of one of the other participants of congruent geneder to the 
	participant for whom the features are being developed

	:type edge_dest: networkx.node
	:param edge_dest: The current male/female node (which is the only node in this particular 
	projected graph with incoming edges)
	"""

	has_edge = orig_graph.has_edge(edge_origin, edge_dest)
	if has_edge:
		return 1
	else:
		return -1

def calculate_features(graph, projections, row):
	"""
	This one's ugly. The function cycles through all the projected graphs that have been 
	created calculating the sum of weights * values for a particular date. I essentially 
	split it into  two functions in one. One is for male paticipants and the other
	is for female paticipants (made it easier), but it's very poor style.

	:type graph: networkx.DiGraph
	:param graph: the original, directed, bipartite graph

	:type projections: list of networkx.Graph
	:param projections: the four projected graphs for each row

	:type row: list of strings
	:param row: the current speed-date as represented by a list information about it
	"""

	gender = row[4]
	features = []
	""" start male is primary """
	if gender == '1':
		cur_male_participant = row[1]
		males = set(n for n,d in graph.nodes(data=True) if d['bipartite']=='1')
		females = set(n for n,d in graph.nodes(data=True) if d['bipartite']=='2')

		# iterate through male projections calculating sum
		for projection in projections[:2]:
			total = 0
			for male in males:
				if male != cur_male_participant:
					weight = get_weight(projection, cur_male_participant, male)
					value = get_value(graph, male, get_edge_dest(row))		# look out for this
					total += weight * value
			features.append(total)

		cur_female_date = get_edge_dest(row)
		for projection in projections[2:]:
			total = 0
			for female in females:
				if female != cur_female_date:
					weight = get_weight(projection, cur_female_date, female)
					value = get_value(graph, female, cur_male_participant)		# look out for this
					total += weight * value
			features.append(total)
	""" end male is primary """
	""" start female is primary """
	elif gender == '2':
		cur_female_participant = row[1]
		males = set(n for n,d in graph.nodes(data=True) if d['bipartite']=='1')
		females = set(n for n,d in graph.nodes(data=True) if d['bipartite']=='2')

		# iterate through male projections calculating sum
		cur_male_date = get_edge_dest(row)
		for projection in projections[:2]:
			total = 0
			for male in males:
				if male != cur_male_date:
					weight = get_weight(projection, cur_male_date, male)
					value = get_value(graph, cur_female_participant, male)		# look out for this
					total += weight * value
			features.append(total)

		# iterate through female projections calculating sum
		for projection in projections[2:]:
			total = 0
			for female in females:
				if female != cur_female_participant:
					weight = get_weight(projection, cur_female_participant, female)
					value = get_value(graph, female, cur_male_date)		# look out for this
					total += weight * value
			features.append(total)
	""" end female is primary """
	else:
		pass
	return features

def get_new_row(old_row, feature_list):
	""" 
	Forms the string of the new row of the feature file to be created.

	:type old_row: list of strings
	:param old_row: the original row from the file

	:type feature_list: list of strings
	:param feature_list: a list of features to be added to the file
	"""

	return [old_row[1]] + [get_edge_dest(old_row)] + feature_list + [old_row[7]]

def extract_graph_features(inputfile='interactions.csv', delim=','):
	""" 
	This is the controlling function which calls the above ones. It iterates through the rows of the 
	file creating projected graphs and extracting features from them

	:type inputfile: string
	:param inputfile: the file containing the graph data

	:type delim: string
	:param delim: the delimiter in the csv file
	"""

	graph_dict = load_graphs_from_csv(inputfile)
	new_rows = []
	with open(inputfile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delim)
		reader.next()
		for row in reader:
			choice_f = row[7]
			# select appropriate graph
			session = row[1][1:5]
			B = graph_dict[session]

			# remove date edge
			edge_origin = row[1]
			edge_dest = get_edge_dest(row)

			if B.has_edge(edge_origin, edge_dest):
				B.remove_edge(edge_origin, edge_dest)
			if B.has_edge(edge_origin, edge_dest):
				B.remove_edge(edge_dest, edge_origin)

			if DRAWING:
				pos = dict()
				male = set(n for n,d in B.nodes(data=True) if d['bipartite']=='1')
				for index,node in enumerate(male):
					pos[node] = (3,index)
				female = set(n for n,d in B.nodes(data=True) if d['bipartite']=='2')
				for index,node in enumerate(female):
					pos[node] = (0,index)
				nx.draw_networkx(B, node_size=40, node_color='red', with_labels=True, arrows=True, pos=pos)
				plt.show()

			# get graph projectsions
			# 2 male graphs
				# a is likes in common with other guys (0)
				# b is being liked in common with other guys (1)
			# 2 female graphs
				# a is likes in common with other gals (2)
				# b is being liked in common with other gals (3)
			projections = get_projected_graphs(B, row)
			for projection in projections:
				pos = nx.spring_layout(projection)
				nx.draw_networkx(projection, node_size=40, node_color='red', with_labels=True, arrows=True, pos=pos)
				plt.show()

			# calculate actual features
			feature_list = calculate_features(B, projections, row)

			# add edge back in
			B.add_edge(edge_origin, edge_dest)
			B.add_edge(edge_dest, edge_origin)

			# add new row to list of samples to write
			new_rows.append(get_new_row(row, feature_list))

	outputfile = 'network_features.csv'
	with open(outputfile, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=delim)
		for row in new_rows:
			writer.writerow(row)	

if __name__ == '__main__':
	extract_graph_features()
