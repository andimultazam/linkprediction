from pyspark import SparkConf, SparkContext
import Utils
from NodeSuggestion import NodeSuggestion
import numpy as np

def getNodeSuggestions(nodePair, bc):
    adamicAdar = nodePair.computeAdamicAdar(bc)
    result = []
    result.append((nodePair.getId1(), NodeSuggestion(nodePair.getId2(), adamicAdar)))
    result.append((nodePair.getId2(), NodeSuggestion(nodePair.getId1(), adamicAdar)))
    return result


def run_algo(masterTrainEdgeFile):
    ####### Step 1: get rdd of edges and node-neighbors ########
    # Get rdd of edges (nodeId, following-node-id)
    edges = Utils.getEdges(sc, masterTrainEdgeFile).cache()

    # Get rdd of (nodeId, neighbors-set)
    nodeNeighbors = Utils.getNeighbors(edges)
    # print("----------10 nodeNeighbors-----------------")
    # for node in nodeNeighbors.take(10):
    #     print("Node Id: {}, neighbors: {}".format(node[0], node[1]))

    ####### Step 2: Get rdd of NodePair(nodeId1, nodeId2, commonNeighbors-set) ########
    nodePairs = Utils.getCommonNeighbors(nodeNeighbors)
    # print("----------10 commonNeighbors-----------------")
    # for commonNeighbor in commonNeighbors.take(10):
    #     print(commonNeighbor)
    ####### Done Get rdd of (nodeId1-nodeId2, commonNeighbors-set) ########

    ##########Step 3: Compute suggestion for each node###############
    # Compute nodeDgree
    # nodeDegree is counted as both followers+following
    nodeDegrees = nodeNeighbors.map(lambda node: (node[0], len(node[1]))).collectAsMap()
    print(nodeDegrees)
    bc = sc.broadcast(nodeDegrees)

    nodeSuggestions = nodePairs.flatMap(lambda nodePair: getNodeSuggestions(nodePair, bc))
    nodeSuggestions = Utils.removeExistingSuggestions(nodeSuggestions, edges)
    for i in nodeSuggestions.take(10):
        print(i)
    return nodeSuggestions

def getFprTpr(masterTrainEdgeFile, masterTestEdgeFile, nodeSuggestions, outFile, threshold):
    topKSuggestions = Utils.getSuggestionsAboveThreshold(nodeSuggestions, threshold).cache()

    # #########Step 5: Validate result###########
    edges = Utils.getEdges(sc, masterTrainEdgeFile)
    testEdges = Utils.getEdges(sc, masterTestEdgeFile)
    # Write into output file
    fpr, tpr = Utils.validateResult(edges, testEdges, topKSuggestions, outFile)
    return fpr, tpr

def evaluateROC(nodeSuggetions, thresholds):
    print("Evaluate based on ROC")
    fpr = np.zeros(len(thresholds))
    tpr = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        print("Threshold={}".format(str(thresholds[i])))
        outFile = trainFolder + "/adamicAdar_threshold=" + str(thresholds[i]) + ".result"
        Utils.removeExistingDir(outFile)
        fpr[i], tpr[i] = getFprTpr(masterTrainEdgeFile, masterTestEdgeFile, nodeSuggetions, outFile, thresholds[i])
    return Utils.calculate_auc(fpr, tpr)

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
        outFile = trainFolder + "/adamicAdar_k=" + str(i+1) + ".result"
        Utils.removeExistingDir(outFile)
        tp[i] = getNoOfCorrectPrediciton(masterTrainEdgeFile, masterTestEdgeFile, nodeSuggestions, outFile, (i+1))
    print(tp)
    return tp


if __name__ == '__main__':
    # Create SparkContext
    conf = SparkConf() \
        .setAppName("Link Prediction Adamic Adar Algorithm")
    sc = SparkContext(conf=conf)

    # Input parameters
    # src = sys.argv[1]
    #src = '/home/grace/nus/big_data/project/linkprediction/data/twitter_t1'
    src = '/home/kienguye/NUS/BigData/FinalProject/twitter_t1'
    thresholds = [0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    K = 10

    trainFolder = src + "/train"
    testFolder = src + "/test"
    masterTrainEdgeFile = trainFolder + "/master.edges"
    masterTestEdgeFile = testFolder + "/master.edges"
    nodeSuggetions = run_algo(masterTrainEdgeFile).cache()
    #evaluateROC(nodeSuggetions, thresholds)
    evaluateNoOfCorrectPrediction(nodeSuggetions, K)
