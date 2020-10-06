from pyspark import SparkConf, SparkContext
import Utils
from NodeSuggestion import NodeSuggestion
import numpy as np
def get_neighbours(sc, masterTrainEdgeFile):
    #Get rdd of edges (nodeId, following-node-id)
    edges = Utils.getEdges(sc, masterTrainEdgeFile).cache()
    nodeFollowings = Utils.getNodeFollowings(edges)
    nodeFollowers = Utils.getNodeFollowers(edges)

    #Get rdd of (nodeId, neighbors-set)
    nodeNeighbors = nodeFollowers.union(nodeFollowings).groupByKey().mapValues(lambda v: set.union(*v))
    # nodeNeighbors.saveAsTextFile("data/result")
    # for node in nodeNeighbors.take(10):
    #     print("Node Id: {}, neighbors: {}".format(node[0], node[1]))

    #Get common neighbours
    test = nodeNeighbors.intersection(nodeNeighbors)

    #Find common neighbours degree
    
def run_algo(masterTrainEdgeFile):
    #Get rdd of edges (nodeId, following-node-id)
    edges = Utils.getEdges(sc, masterTrainEdgeFile).cache()
    nodeFollowings = Utils.getNodeFollowings(edges)
    nodeFollowers = Utils.getNodeFollowers(edges)
    
    #Get rdd of (nodeId, neighbors-set)
    nodeNeighbors = nodeFollowers.union(nodeFollowings).groupByKey().mapValues(lambda v: set.union(*v))
    for node in nodeNeighbors.take(10):
        print("Node Id: {}, neighbors: {}".format(node[0], node[1]))
    #Calculating Jaccard Similarity for each pair - round up similarity result to 5 decimal
    pairSimilarity  = nodeNeighbors.cartesian(nodeNeighbors).map(lambda l: ((l[0][0], l[1][0]), round(len(
        Utils.intersection(l[0][1], l[1][1])) / float(len(Utils.union(l[0][1], l[1][1]))), 5))).sortByKey()
    #Filter out pair that does not have any similarity and pair with same node
    pairSimilarity = pairSimilarity.filter(lambda x: x[0][0] != x[0][1] and x[1] > 0.0)
    nodeSuggestions = pairSimilarity.map(lambda x: (x[0][0], NodeSuggestion(x[0][1], x[1])))
    nodeSuggestions = Utils.removeExistingSuggestions(nodeSuggestions, edges)
    return nodeSuggestions

    
def setup_pyspark():
    conf = SparkConf() \
        .setAppName("Link Prediction Jaccard Algorithm")
    sc = SparkContext(conf=conf)
    return sc

def getNoOfCorrectPrediciton(masterTrainEdgeFile, masterTestEdgeFile, nodeSuggestions, outFile, K):
    topKSuggestions = Utils.getTopKSuggestions(nodeSuggestions, K).cache()

    # #########Step 5: Validate result###########
    edges = Utils.getEdges(sc, masterTrainEdgeFile)
    testEdges = Utils.getEdges(sc, masterTestEdgeFile)
    # Write into output file
    tp = Utils.validateResultCorrectPredictions(edges, testEdges, topKSuggestions, outFile, K)
    return tp

def evaluateNoOfCorrectPrediction(nodeSuggestions, K):
    print("Evaluate based on No. of correct predictions")
    tp = np.zeros(K)
    for i in range(K):
        print("K={}".format(str(i+1)))
        outFile = trainFolder + "/jaccard_k=" + str(i+1) + ".result"
        Utils.removeExistingDir(outFile)
        tp[i] = getNoOfCorrectPrediciton(masterTrainEdgeFile, masterTestEdgeFile, nodeSuggestions, outFile, (i+1))
    print(tp)
    return tp

if __name__ == "__main__":
    # Create SparkContext
    conf = SparkConf() \
        .setAppName("Link Prediction Jaccard Similarity Algorithm")
    sc = SparkContext(conf=conf)

    # Input parameters
    # src = sys.argv[1]
    #src = 'data/twitter_t1'
    src = '/home/kienguye/NUS/BigData/FinalProject/twitter_t1'
    K = 10

    trainFolder = src + "/train"
    testFolder = src + "/test"
    masterTrainEdgeFile = trainFolder + "/master.edges"
    masterTestEdgeFile = testFolder + "/master.edges"
    nodeSuggetions = run_algo(masterTrainEdgeFile).cache()
    # evaluateROC(nodeSuggetions, thresholds)
    evaluateNoOfCorrectPrediction(nodeSuggetions, K)
