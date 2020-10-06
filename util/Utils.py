import os
from shutil import rmtree
import matplotlib.pyplot as plt
from NodePair import NodePair
from ThresholdResult import ThresholdResult
from Result import Result
from sklearn import metrics
import numpy as np


def removeExistingDir(dst):
    if os.path.exists(dst) and os.path.isdir(dst):
        print('Removing existing directory "{}"'.format(dst))
        rmtree(dst, ignore_errors=False)


# return rdd of (nodeId, nodeId2)
def getEdges(sc, masterEdgeFiles):
    edges = sc.textFile(masterEdgeFiles).map(lambda text: text.split()).map(lambda text: (text[0], text[1]))
    return edges

def getTotalNodeCount(edges):
    return edges.flatMap(lambda edge: list[edge[0], edge[1]]).distinct().count()

# return rdd of (nodeId, following-set)
def getNodeFollowings(edges):
    nodeFollowings = edges.groupByKey().mapValues(lambda iterable: set(iterable))
    # for node in nodeFollowings.take(10):
    #     print("Node Id: {}, following-set: {}".format(node[0], node[1]))
    return nodeFollowings


# return rdd of (nodeId, follower-set)
def getNodeFollowers(edges):
    nodeFollowers = edges.map(lambda edge: (edge[1], edge[0])).groupByKey().mapValues(lambda iterable: set(iterable))
    # for node in nodeFollowers.take(10):
    #     print("Node Id: {}, followers-set: {}".format(node[0], node[1]))
    return nodeFollowers


def getNeighbors(edges):
    edges.cache()
    nodeFollowings = getNodeFollowings(edges)
    nodeFollowers = getNodeFollowers(edges)
    nodeNeighbors = nodeFollowers.union(nodeFollowings).groupByKey().mapValues(lambda v: set.union(*v))
    return nodeNeighbors

#nodeNeighbor[0] is the nodeId
#nodeNeighbor[1] is set of neighbors
def nodeNeighborFlatMap(nodeNeighbor):
    #result is a list of (nodeId1, nodeId2) - nodeId
    #where (nodeId1, nodeId2) is a pair of nodeIds in the neighbors list
    result = []
    nodeId = nodeNeighbor[0]
    neighbors = list(nodeNeighbor[1])
    neighbors.sort()
    for i in range(len(neighbors)-1):
        for j in range(i+1, len(neighbors)):
            result.append(((neighbors[i], neighbors[j]), nodeId))
    return result

def getCommonNeighbors(nodeNeighbors):
    cnPairs = nodeNeighbors.flatMap(lambda nodeNeighbor: nodeNeighborFlatMap(nodeNeighbor))
    commonNeighbors = cnPairs.groupByKey().mapValues(lambda v: set(v)).map(
        lambda cn: NodePair(cn[0][0], cn[0][1], cn[1]))
    return commonNeighbors

def getNodePairWithCommonNeighbor(nodeId, followingNodes, commonNeighbor):
    pairs = []
    for followingNode in followingNodes:
        if followingNode != nodeId:
            #one pair means nodeId follows commonNeighbor, commonNeighbor follows followingNode
            pair = ((nodeId, followingNode), commonNeighbor)
            pairs.append(pair)
    return pairs

def getDirectedCommonNeighbors(edges):
    nodeFollowings = getNodeFollowings(edges)
    # while joining
    # match[0] is the common neighbor
    # match[1][0] is the start nodeId
    # match[1][1] is the list of followingNodesf
    commonNeighbors = edges.map(lambda edge: (edge[1], edge[0])).join(nodeFollowings) \
        .flatMap(lambda match: getNodePairWithCommonNeighbor(match[1][0], match[1][1], match[0])) \
        .groupByKey().mapValues(lambda v: set(v)) \
        .map(lambda nodePair: NodePair(nodePair[0][0], nodePair[0][1], nodePair[1]))
    return commonNeighbors


#input: nodeSuggestion is NodeSuggestion object
def getTopKSuggestionWithSim(nodeSuggestion, k):
    #sort the suggestion by similarity
    sortedSuggestion = sorted(nodeSuggestion, key = (lambda suggestion: suggestion.getSim()), reverse = True)
    noOfItems = min(k, len(sortedSuggestion))
    return sortedSuggestion[:noOfItems]

#input: nodeSuggestions is list of NodeSuggestions obje
def getSuggestionNodeIds(nodeSuggestions):
    result = []
    for nodeSuggestion in nodeSuggestions:
        result.append(nodeSuggestion.getSuggestionId())
    return result

#return rdd of (key, value) - (nodeId, list of topK suggestionIds)
def getTopKSuggestions(nodeSuggestions, K):
    #nodeSuggestions is rdd of (nodeId, NodeSuggestion)
    topKSuggestions = nodeSuggestions.groupByKey().mapValues(lambda suggestion: getTopKSuggestionWithSim(suggestion, K))\
        .mapValues(lambda nodeSuggestions: getSuggestionNodeIds(nodeSuggestions))
    return topKSuggestions

def getSuggestionsAboveThreshold(nodeSuggestions, threshold):
    #nodeSuggestions is rdd of (nodeId, NodeSuggestion)
    goodSuggestions = nodeSuggestions.filter(lambda suggestion: suggestion[1].getSim()>=threshold)\
            .map(lambda suggestion: (suggestion[0], suggestion[1].getSuggestionId()))\
             .groupByKey().mapValues(lambda suggestions: set(suggestions))
    return goodSuggestions


def removeExistingSuggestions(nodeSuggestions, edges):
    edgeWithKeys = edges.map(lambda edge: ((edge[0], edge[1]), 1))
    return nodeSuggestions.map(lambda nodeSuggestion: ((nodeSuggestion[0], nodeSuggestion[1].getSuggestionId()), nodeSuggestion[1]))\
        .subtractByKey(edgeWithKeys).map(lambda nodeSuggestion: (nodeSuggestion[0][0], nodeSuggestion[1]))

#return fpr, tpr
# def validateResult(trainEdges, testEdges, topKSuggestions, K, outFile):
#     trainNodeFollowings = getNodeFollowings(trainEdges)
#     testNodeFollowings = getNodeFollowings(testEdges)
#
#     print("traintestedges")
#     print(trainEdges.take(2))
#     print(testEdges.take(2))
#     print("testnodefollowings")
#     print(testNodeFollowings.take(2))
#     print("trainnodefollowings")
#     print(trainNodeFollowings.take(2))
#
#     print("topKsuggestions")
#     print(topKSuggestions.take(2))
#
#     # While joining
#     # tuple[0] is the nodeId
#     # tuple[1][0] is the list of nodes that this nodeId follows in the test dataset
#     # tuple[1][1] is the list of suggestionIds.
#     # After joing -- rdd of Result objects
#
#     #TODO- Check this again
#     totalNodeCount = trainEdges.union(testEdges).distinct().count()
#     print(totalNodeCount)
#     results = testNodeFollowings.join(trainNodeFollowings).mapValues(lambda x: (x[0], len(union(x[0], x[1]))))\
#             .join(topKSuggestions)\
#             .map(lambda tuple: Result(tuple[0], tuple[1][1], tuple[1][0][0], K, totalNodeCount, tuple[1][0][1])).cache()
#     # for i in results.take(10):
#     #     print(i)
#
#     # results = testNodeFollowings.join(topKSuggestions)\
#     #     .map(lambda tuple: Result(tuple[0], tuple[1][1], tuple[1][0], K)).cache()
#     # print("----------10 results-----------------")
#     # for i in results.take(10):
#     #     print(i)
#     #
#     # results.map(lambda result: result.toSaveString()).saveAsTextFile(outFile)
#     coordinates = results.map(lambda result: ([result.getTP()], [result.getFP()], [result.getTN()], [result.getFN()])) \
#         .reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))
#     tp, fp, tn, fn = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
#     print("True Positive: {}, False Positive: {}, True Negative: {}, False Negative: {}".format(tp, fp, tn, fn))
#     # FPR = FP/(FP+TN)
#     # TPR = TP/(TP+FN)
#     print(len(fp), len(tn))
#     fpr = [a / b for a, b in zip(fp, [x + y for x, y in zip(fp, tn)])]
#     tpr = [a / b for a, b in zip(tp, [x + y for x, y in zip(tp, fn)])]
#     y = np.array(tpr)
#     x = np.array(fpr)
#     print(fpr, tpr)
#     return x, y

#return fpr, tpr
def validateResult(trainEdges, testEdges, topKSuggestions, outFile):
    trainNodeFollowings = getNodeFollowings(trainEdges)
    testNodeFollowings = getNodeFollowings(testEdges)

    # While joining
    # tuple[0] is the nodeId
    # tuple[1][0][0] is the list of nodes that this nodeId follows in the test dataset
    # tuple[1][0][1] is the nodeFollowingCount
    # tuple[1][1] is the list of suggestionIds.
    # After joing -- rdd of Result objects
    totalNodeCount = trainEdges.distinct().count()
    print("totalNodeCount: {}".format(totalNodeCount))
    results = testNodeFollowings.join(trainNodeFollowings).mapValues(lambda x: (x[0], len(x[1])))\
            .join(topKSuggestions)\
            .map(lambda tuple: ThresholdResult(tuple[0], tuple[1][1], tuple[1][0][0], totalNodeCount, tuple[1][0][1])).cache()
    # for i in results.take(10):
    #     print(i)

    # results = testNodeFollowings.join(topKSuggestions)\
    #     .map(lambda tuple: Result(tuple[0], tuple[1][1], tuple[1][0], K)).cache()
    # print("----------10 results-----------------")
    # for i in results.take(10):
    #     print(i)
    #
    # results.map(lambda result: result.toSaveString()).saveAsTextFile(outFile)
    coordinates = results.map(lambda result: (1, (result.getTP(), result.getFP(), result.getTN(), result.getFN()))) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))

    coordinates.map(lambda coordinate: "True Positive: {}, False Positive: {}, True Negative: {}, False Negative: {}"
                    .format(coordinate[1][0], coordinate[1][1], coordinate[1][2], coordinate[1][3])).saveAsTextFile(outFile)

    coordinates = coordinates.map(lambda coordinate: coordinate[1]).collect()[0]
    print(coordinates)
    tp, fp, tn, fn = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    print("True Positive: {}, False Positive: {}, True Negative: {}, False Negative: {}".format(tp, fp, tn, fn))
    # FPR = FP/(FP+TN)
    # TPR = TP/(TP+FN)
    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    print(fpr, tpr)
    return fpr, tpr

def validateResultCorrectPredictions(trainEdges, testEdges, topKSuggestions, outFile, K):
    trainNodeFollowings = getNodeFollowings(trainEdges)
    testNodeFollowings = getNodeFollowings(testEdges)

    # While joining
    # tuple[0] is the nodeId
    # tuple[1][0][0] is the list of nodes that this nodeId follows in the test dataset
    # tuple[1][0][1] is the nodeFollowingCount
    # tuple[1][1] is the list of suggestionIds.
    # After joing -- rdd of Result objects
    totalNodeCount = trainEdges.distinct().count()
    print("totalNodeCount: {}".format(totalNodeCount))
    results = testNodeFollowings.join(trainNodeFollowings).mapValues(lambda x: (x[0], len(x[1])))\
            .join(topKSuggestions)\
            .map(lambda tuple: Result(tuple[0], tuple[1][1], tuple[1][0][0], K, totalNodeCount, tuple[1][0][1])).cache()
    # for i in results.take(10):
    #     print(i)

    # results = testNodeFollowings.join(topKSuggestions)\
    #     .map(lambda tuple: Result(tuple[0], tuple[1][1], tuple[1][0], K)).cache()
    # print("----------10 results-----------------")
    # for i in results.take(10):
    #     print(i)
    #
    # results.map(lambda result: result.toSaveString()).saveAsTextFile(outFile)
    coordinates = results.map(lambda result: (1, (result.getTP(), result.getFP(), result.getTN(), result.getFN()))) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))

    coordinates.map(lambda coordinate: "CorrectPrediction: {}"
                    .format(coordinate[1][0])).saveAsTextFile(outFile)

    coordinates = coordinates.map(lambda coordinate: coordinate[1]).collect()[0]
    print(coordinates)
    tp, fp, tn, fn = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    print("True Positive: {}, False Positive: {}, True Negative: {}, False Negative: {}".format(tp, fp, tn, fn))
    # FPR = FP/(FP+TN)
    # TPR = TP/(TP+FN)
    return tp


def calculate_auc(fpr, tpr):
    print("fpr: {}".format(fpr))
    print("tpr: {}".format(tpr))
    plt.plot(fpr, tpr)
    plt.savefig("/home/kienguye/mygraph.png")
    auc = metrics.auc(fpr, tpr)
    print(auc)

def union(set1, set2):
    if set1 is None:
        if set2 is None:
            return None
        else:
            return set2.union(set2)
    return set1.union(set2)


def intersection(set1, set2):
    if set1 is None or set2 is None:
        return None
    return set1.intersection(set2)

def find_jaccard_sim(set1, set2):
    similarity = len(intersection(set1, set2)) / len(union(set1, set2))
    return similarity

