Project Overview:
-----------------
In social signal processing and computational behavioral analysis, researchers incorporate a variety of feature types in their models for predicting information about people. For example, an audio feature of speech such as pitch can tell you fairly accurately if someone is stressed [0], and a dyadic feature like mutual eye contact can help tell you if people are friends [1]. <b>The goal of this project is to use a large variety of feature types (audio, visual, dyadic, network, and others) to predict the outcome of an interaction between two people, and in doing so determine the impact these different feature types have on prediction accuracy of human decisions.</b> Audiovisual data was used for feature development. 

Concise Project Overview:
-------------------------
Q: What is this? 

A: A machine learning project focused on social signal processing.

Q: What does this code do? 

A: Extracts features, performs classification, or serves related purposes.


LowLevelDescriptor.py (LLD):
----------------------------
A LLD is a low level feature like pitch (f0). <b>This file contains a class used for extracting LLDs from audio data.</b> It uses an audio feature extraction library called openSMILE, which is great [2].

graph_ssp.py:
-------------
The specific data I used contained a great deal of network data due to the interaction of many different people in pairs. This data can be used quite effectively for predicting the outcome of interactions (in fact, it was the most effective type of feature.) <b>This file contains code for developing these features, which mostly consist of projecting the bipartite, undirected graph onto a weighted, directed graph, the weights of which are then aggregated through some reduce function</b>. This paper provided the basis for these features [3].

![alt tag](https://github.com/wulfebw/Independent_Study/blob/master/media/bipartite.png)
The bipartite preference graph was used to create the following weighted projection. Summing and taking the average of the edge weights produced the network features used in predicting decisions.
![alt tag](https://github.com/wulfebw/Independent_Study/blob/master/media/projected.png)

haarcascade_training.py:
------------------------
One of the visual features I used was the amount each person smiled during the interaction and related functionals. The standard approach to smile detection is to extract from each frame a level of smile intensity, but I couldn't find any open source software to accomplish this. Hopefully I can do this in the future, but for the time being I just used openCVs object detection-related functionality to perform the task. I trained a cascade classifier on the data (if you're interested, my other repo - mergevec - has more information on this). <b>The code in this file was used for training the cascade classifier</b>. 

outcome_features.py:
--------------------
Using the data from the outcomes of the interactions, it is possible to make up some features. These sort of seem like cheating, but they're fun to come up with. This file contains the <b>code used for extracting a variety of outcome features.</b> 

Diarize.py:
-----------
In order to extract certain audio features, it's necessary to separate an audio file of people speaking into segments of only one person speaking. This task is called speaker diarization and consists of the two subtasks of (1) speech recognition and (2) speaker recognition. The data used for this research had a lot of human background noise, which prevented the open-source speaker diarization software libraries I tried from working. The audio data used was simultaneously collected from two non-collocated microphones, however, so I wrote this script that takes advantage of this situation to <b>perform a crude speaker diarization.</b>

![alt tag](https://github.com/wulfebw/Independent_Study/blob/master/media/i1_minus_i2_1.gif)

Subtracting speech intensity from multiple microphones and averaging the resulting values provided approximate speech segments for the different speakers.

svm_visual.py:
--------------
<b>The code in this file uses sklearn to perform a number of machine learning-related tasks</b> ranging from train-validation-test on the model to plotting learning curves. 

visual_feature_extraction.py:
-----------------------------
<b>Code for extracting visual features</b> like body coordinates for all frames of a video and motion template image entropy and mean value.

![alt tag](https://github.com/wulfebw/Independent_Study/blob/master/media/motion_template.png)

A motion template image from which functionals were extracted.

Further Information
-------------------
I wrote an in-depth paper covering the above research. Contact me by email if you're interested in reading it or want more information on the project. Parts of this system are omitted.

References
----------

[0] http://www.cs.dartmouth.edu/~campbell/ubicomp-2012.pdf

[1] http://www.aclweb.org/anthology/W13-4007

[2] http://opensmile.sourceforge.net/

[3] http://cs.stanford.edu/~emmap1/ngpaper.pdf
