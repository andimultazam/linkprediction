# Link Prediction

## Getting Started
twitter_t1, twitter_t10, twitter_t20, twitter_t50  is the 1%, 10%, 20%, 50% of our dataset
There's test and train folders inside each twitter_t folder

Please start running code with 1% of the dataset, i.e. twitter_t1 folder
twitter_t1/train/master.edges --- contains all the edges in the trained dataset
twitter_t1/train/nodeId.followers --- contains the nodeId - list of followers
twitter_t1/train/nodeId.followingss --- contains the nodeId - list of nodes which that node is following

src/DataCopier.py - extract some % of the original dataset.
src/TrainTestSplitter - split train/test data to 80/20.
src/Preprocess.py - do some simple preprocessing to get all edges and neighbors.
functions Utils.getNodeFollowings, Utils.getNodeFollowers, Utils.getNeighbors can be use to get the corresponding values.
Run each link prediction algorithm e.g. AdamicAdar.py, CNCF.py, etc.

## Introduction
The purpose of this project is to analyse current connection in social network and use multiple link prediction algorithms to recommend most likely connection to new social network accounts. We will use Twitter data set as the base to study social network connection. The performance of different algorithms will then be compared and evaluated against actual connection to select the most appropriate link prediction method in terms of accuracy and scalability. I will focus in analysing topology of social network in order to predict possible future connections. Some possible applications of link prediction are providing recommendations in e-commerce.

## Dataset
Source: `https://snap.stanford.edu/data/ego-Twitter.html`.
The Twitter dataset contains 81,306 nodes and 1,768,149 edges, where nodes represent Twitter account and edges represent Twitter circle of connections. Every node in dataset also contains a set of features that are different from each other ranging from 700 - 950 names, e.g. #Apple, #DoctorWho, #Avengers, etc. Given a list of nodes and it’s features, we would like to get an insight of how similar one node to another by their common features. We started with cleaning some features that has a similar form of name such as #ClimateChange and #Climatechange, or #Wisconsin and #Wisconsin:. We then list down all combination of pair nodes followed by their common features. We remove those pairs that have no common features to make sure that we are processing potential similar candidates.

## Algorithms
Link prediction based on similarity-based methods can be classified into three types based on the nature of similarity score or index calculation, as stated below: 

**Local Similarity Index Methods:** Measures similarities between 2 nodes based on local properties such as common neighbor and node degree.
**Overall Similarity Index Methods:** Measures similarity based on path and topology of the network.
**Node Guidance Capability Index Methods:** Extracts sub-graph containing important neighbors and measures similarity of 2 nodes within the sub-graph.

## Result
