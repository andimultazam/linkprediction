from pyspark import SparkConf, SparkContext, StorageLevel
import Utils
from Edge import Edge
from Path import Path
from NodeSuggestion import NodeSuggestion
import numpy as np
def addEdgeToPath(oldPath, edge):
    newNodeIds = oldPath.getNodeIds().copy()
    newNodeIds.append(edge.getId2())
    return Path(oldPath.getStartId(), edge.getId2(), newNodeIds)


def extractNextLevelPaths(previousLevelPaths, edgeWithStartIdAsKey):
    # rdd of (key, value) - (endId, path)
    previousLevelPathsWithEndIdAsKey = previousLevelPaths.map(lambda path: (path.getEndId(), path))

    # While joining
    # match[0] is the key nodeId
    # match[1][0] is the previous-level path
    # match[1][1] is the edge to be added to the path
    nextLevelPaths = previousLevelPathsWithEndIdAsKey.join(edgeWithStartIdAsKey) \
        .filter(lambda match: match[1][1].getId2() not in match[1][0].getNodeIds()) \
        .map(lambda match: addEdgeToPath(match[1][0], match[1][1]))

    return nextLevelPaths

#return rdd of ((startId, endId), (length, count))
def getPathCounts(paths, length1PathCounts, length):
    pathCounts =  paths.map(lambda path: ((path.getStartId(), path.getEndId(), length), 1))\
        .reduceByKey(lambda a, b: a+b)\
        .map(lambda nodePair: ((nodePair[0][0], nodePair[0][1]), (nodePair[0][2], nodePair[1])))
    return pathCounts.subtractByKey(length1PathCounts)

def getNodeSuggestion(pathCount, BETA):
    startId = pathCount[0][0]
    endId = pathCount[0][1]
    pathCounts = list(pathCount[1])
    sim = 0.0
    for count in pathCounts:
        length = count[0]
        lengthCount = count[1]
        sim = sim + pow(BETA, length) * lengthCount
    return (startId, NodeSuggestion(endId, sim))


def run_algo(masterTrainEdgeFile, BETA):
    ####### Step 1: get rdd of edges ########
    # Get rdd of edges (nodeId, following-node-id)
    edges = Utils.getEdges(sc, masterTrainEdgeFile).cache()

    ####### Step 2: Get all paths with length 2-6 ########
    # rdd of (startId, Edge)
    edgeWithStartIdAsKey = edges.map(lambda edge: (edge[0], Edge(edge[0], edge[1]))).cache()
    print("edgeWithStartIdAsKey")
    for edge in edgeWithStartIdAsKey.take(10):
        print(edge)

    # rdd of all Path of length 1
    length1Paths = edges.map(lambda edge: Path(edge[0], edge[1], [edge[0], edge[1]])).persist(
        StorageLevel.MEMORY_AND_DISK_SER)
    # print("length1Paths")
    # for path in length1Paths.take(10):
    #     print(path)
    # rdd of ((startId, endId), 1)
    length1PathCounts = edges.map(lambda edge: ((edge[0], edge[1]), 1)).persist(StorageLevel.MEMORY_AND_DISK_SER)

    edges.unpersist()

    length2Paths = extractNextLevelPaths(length1Paths, edgeWithStartIdAsKey).persist(StorageLevel.MEMORY_AND_DISK_SER)
    # length1Paths.unpersist()
    length2PathCounts = getPathCounts(length2Paths, length1PathCounts, 2).persist(StorageLevel.MEMORY_AND_DISK_SER)
    # print("length2Paths")
    # for path in length2Paths.take(10):
    #     print(path)

    length3Paths = extractNextLevelPaths(length2Paths, edgeWithStartIdAsKey).persist(StorageLevel.MEMORY_AND_DISK_SER)
    # length2Paths.unpersist()
    length3PathCounts = getPathCounts(length3Paths, length1PathCounts, 3).persist(StorageLevel.MEMORY_AND_DISK_SER)
    # print("length3Paths")
    # for path in length3Paths.take(10):
    #     print(path)

    # length4Paths = extractNextLevelPaths(length3Paths, edgeWithStartIdAsKey).persist(StorageLevel.MEMORY_AND_DISK_SER)
    # # length3Paths.unpersist()
    # length4PathCounts = getPathCounts(length4Paths, length1PathCounts, 4).persist(StorageLevel.MEMORY_AND_DISK_SER)
    # print("length4Paths")
    # for path in length4Paths.take(10):
    #     print(path)

    # length5Paths = extractNextLevelPaths(length4Paths, edgeWithStartIdAsKey).cache()
    # length4Paths.unpersist()
    # length5PathCounts = getPathCounts(length5Paths, length1PathCounts, 5)
    # print("length5Paths")
    # for path in length5Paths.take(10):
    #     print(path)

    # length6Paths = extractNextLevelPaths(length5Paths, edgeWithStartIdAsKey)
    # length5Paths.unpersist()
    # length6PathCounts = getPathCounts(length6Paths, length1PathCounts, 6)
    # print("length6Paths")
    # for path in length6Paths.take(10):
    #     print(path)

    ##########Step 3 Compute suggestion for each node##########
    # .union(length4PathCounts).union(length5PathCounts).union(length6PathCounts)
    allPathCounts = length2PathCounts.union(length3PathCounts)
    nodeSuggestions = allPathCounts.groupByKey() \
        .map(lambda pathCount: getNodeSuggestion(pathCount, BETA))

    # rdd of (key, value) - (nodeId, list of topK suggestionIds)
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
        outFile = trainFolder + "/katz_threshold=" + str(thresholds[i]) + ".result"
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
        outFile = trainFolder + "/katz_k=" + str(i+1) + ".result"
        Utils.removeExistingDir(outFile)
        tp[i] = getNoOfCorrectPrediciton(masterTrainEdgeFile, masterTestEdgeFile, nodeSuggestions, outFile, (i+1))
    print(tp)
    return tp

if __name__ == '__main__':
    # Create SparkContext
    conf = SparkConf() \
        .setAppName("Link Prediction Katz Algorithm")
    sc = SparkContext(conf=conf)

    # Input parameters
    # src = sys.argv[1]
    #src = '/home/grace/nus/big_data/project/linkprediction/data/twitter_t1'
    src = '/home/kienguye/NUS/BigData/FinalProject/twitter_t1'
    thresholds = [3.5, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]
    BETA = 0.8
    K = 10

    trainFolder = src + "/train"
    testFolder = src + "/test"
    masterTrainEdgeFile = trainFolder + "/master.edges"
    masterTestEdgeFile = testFolder + "/master.edges"
    nodeSuggetions = run_algo(masterTrainEdgeFile, BETA).cache()
    #evaluateROC(nodeSuggetions, thresholds)
    evaluateNoOfCorrectPrediction(nodeSuggetions, K)
