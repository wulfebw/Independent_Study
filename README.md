These scripts were used for a variety of tasks in my independent study. While this is primarily here for my own future reference, I'll give a brief description of the project and each of the files.

Project Overview:
-----------------
In social signal processing and computational behavioral analysis, researchers incorporate a variety of feature types in their models for predicting information about people. For example, an audio feature of speech such as pitch can tell you fairly accurately if someone is stressed [0], and a dyadic feature like mutual eye contact can help tell you if people are friends [1]. <b>The goal of this project is to use a large variety of feature types (audio, visual, dyadic, network, and others) to predict the outcome of an interaction between two people, and in doing so determine the impact these different feature types have on prediction accuracy of human decisions.</b> Audiovisual and outcome data was used for feature development. 

Concise Project Overview:
-------------------------
Q: What is this? 

A: A machine learning project focused on social signal processing.

Q: What does this code do? 

A: Mostly extracts features. 


Diarize.py:
-----------
In order to extract certain audio features, it's necessary to separate an audio file of people speaking into segments of only one person speaking. This task is called speaker diarization and consists of the two subtasks of (1) speech recognition and (2) speaker recognition. The data used for this research had a lot of human background noise, which prevented the open-source speaker diarization software libraries I tried from working. The audio data used was collected from two non-collocated microphones, however, so I wrote this script that takes advantage of this situation to <b>perform a crude speaker diarization.</b>

LowLevelDescriptor.py (LLD):
----------------------------
A LLD is a low level feature like pitch (f0). <b>This file contains a class used for extracting LLDs from audio data.</b> It uses a audio feature extraction library called openSMILE, which is great [2].

graph_ssp.py:
-------------
The specific data I used contained a great deal of network data due to the interaction of many different people in pairs. This data can be used quite effectively for predicting the outcome of interactions (in fact, it's the most effective type of feature.) <b>This file contains code for developing these features, which mostly consist of projecting the bipartite, undirected graph onto a weighted, directed graph, the weights of which are then aggregated through some reduce function</b>. This paper provided the basis for these features [3].

haarcascade_training.py:
------------------------
One of the visual features I used was the amount each person smiled during the interaction and related functionals. The standard approach to smile detection is to extract from each frame a level of smile intensity, but I couldn't find any open source software to accomplish this. Hopefully I can do this in the future, but for the time being I just used openCVs object detection-related functionality to perform the task. I trained a cascade classifier on the data (if you're interested, my other repo - mergevec - has more information on this). <b>The code in this file was used for training the cascade classifier</b> and amongst other things. 

outcome_features.py:
--------------------
Using the data covering the outcomes of the interactions, it is possible to make up some features. These sort of seem like cheating, but they're fun to come up with. This file contains the <b>code used for extracting a variety of outcome features.</b> 

svm_visual.py:
--------------
<b>The code in this file uses sklearn to perform a number of machine learning-related tasks</b> ranging from train-validation-test on the model to plotting learning curves. 

visual_feature_extraction.py:
-----------------------------
<b>Code for extracting visual features</b> like body coordinates for all frames of a video and motion template image entropy and mean value.

===================


[0] http://www.cs.dartmouth.edu/~campbell/ubicomp-2012.pdf

[1] http://www.aclweb.org/anthology/W13-4007

[2] http://opensmile.sourceforge.net/

[3] http://cs.stanford.edu/~emmap1/ngpaper.pdf
