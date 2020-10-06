import sys
import operator
from pyspark import SparkConf, SparkContext
import Utils
from Edge import Edge
from NodePair import NodePair
from CNGraph import CNGraph
from NodeSuggestion import NodeSuggestion
from Result import Result

def checkNodeInNeighborList(node, nodePair):
    if(node == nodePair.getId1() or node == nodePair.getId2() or node in nodePair.commonNeighbors):
        return True
    else:
        return False

def nodePairFlatMap(nodePair):
    list = []
    list.append((nodePair.getId1(), nodePair))
    list.append((nodePair.getId2(), nodePair))
    for commonNeighbor in nodePair.getCommonNeighbors():
        list.append((commonNeighbor, nodePair))
    return list

#Compute list of [nodeId, NodeSugession]. Each cnGraph object will generate at most 2 pairs (since the edge is directed)
def getNodeSuggestions(cnGraph, BETA):
    result = []
    nodePair = cnGraph.getNodePair()
    if Edge(nodePair.getId1(), nodePair.getId2()) not in cnGraph.getEdges():
        result.append((nodePair.getId1(), NodeSuggestion(nodePair.getId2(), cnGraph.calculateSimilarityBasedOnPaths(BETA, nodePair.getId1(), nodePair.getId2()))))
    if Edge(nodePair.getId2(), nodePair.getId1()) not in cnGraph.getEdges():
        result.append((nodePair.getId2(), NodeSuggestion(nodePair.getId1(), cnGraph.calculateSimilarityBasedOnPaths(BETA, nodePair.getId2(), nodePair.getId1()))))
    return result


#Create SparkContext
conf = SparkConf()\
    .setAppName("Link Prediction CNCF_Katz Algorithm")
sc = SparkContext(conf=conf)

#Input parameters
#src = sys.argv[1]
src = '/home/kienguye/NUS/BigData/FinalProject/twitter_t1'
K = 10
BETA=0.8

trainFolder = src + "/train"
testFolder = src + "/test"
masterTrainEdgeFile = trainFolder+"/master.edges"
masterTestEdgeFile = testFolder+"/master.edges"
outFile = trainFolder+"/nodePair_cncf_katz.result"
Utils.removeExistingDir(outFile)

####### Step 1: get rdd of edges and node-neighbors ########
#Get rdd of edges (nodeId, following-node-id)
edges = Utils.getEdges(sc, masterTrainEdgeFile).cache()

#Get rdd of (nodeId, neighbors-set)
nodeNeighbors = Utils.getNeighbors(edges)
# print("----------10 nodeNeighbors-----------------")
# for node in nodeNeighbors.take(10):
#     print("Node Id: {}, neighbors: {}".format(node[0], node[1]))

####### Step 2: Get rdd of NodePair(nodeId1, nodeId2, commonNeighbors-set) ########
commonNeighbors = Utils.getCommonNeighbors(nodeNeighbors)
# print("----------10 commonNeighbors-----------------")
# for commonNeighbor in commonNeighbors.take(10):
#     print(commonNeighbor)
####### Done Get rdd of (nodeId1-nodeId2, commonNeighbors-set) ########


############ Step 3: Compute cnGraphs (common-neighbors graphs)###############
#nodePair is a (key, value) of (node Id of one node in the common-neighbor, NodePair object)
nodePairs = commonNeighbors.flatMap(lambda nodePair: nodePairFlatMap(nodePair))
# print("----------10 nodePairs-----------------")
# for nodePair in nodePairs.take(10):
#     print(nodePair[0], nodePair[1])

#edgeRdd is a (key, value) of (nodeId1, Edge object)
edgeRdd = edges.map(lambda edge: (edge[0], Edge(edge[0], edge[1])))

#Joining the nodePairs with edgeRdd to find out all edges in the common-neighbor-subgraph
#While joining
    #match[0] is the nodeId key
    #match[1][0] is the NodePair object for the common-neighbor-subgraph
    #match[1][1] is the Edge object
#After join+mapping, matchRDD is (NodePair, 1 edge in the common-neighbor-subgraph)
matchRDD = nodePairs.join(edgeRdd)\
    .filter(lambda match: checkNodeInNeighborList(match[1][1].getId2(), match[1][0]))\
    .map(lambda match: match[1])


#create CNGraph object from result of join
cnGraphs = matchRDD.groupByKey().map(lambda entry: CNGraph(entry[0], list(entry[1])))

# print("----------10 cnGraphs-----------------")
# for cnGraph in cnGraphs.take(10):
#     print(cnGraph)
######### Done Compute cnGraphs ###############

##########Step 4: Compute suggestion for each node###############
nodeSuggestions = cnGraphs.flatMap(lambda cnGraph: getNodeSuggestions(cnGraph, BETA))

# print("----------10 nodeSuggestions-----------------")
# for i in nodeSuggestions.take(10):
#     print("NodeId: {}, {}".format(i[0], i[1]))

#rdd of (key, value) - (nodeId, list of topK suggestionIds)
# topKSuggestions = nodeSuggestions.groupByKey().mapValues(lambda suggestion: getTopKSuggestionWithSim(suggestion, K))\
#     .mapValues(lambda nodeSuggestions: getSuggestionNodeIds(nodeSuggestions))
topKSuggestions = Utils.getTopKSuggestions(nodeSuggestions, K)
# print("----------10 topKSuggestions-----------------")
# for i in topKSuggestions.take(10):
#     print(i)
##########Done Compute suggestion for each node###############

#########Step 5: Validate result###########
testEdges = Utils.getEdges(sc, masterTestEdgeFile)
Utils.validateResult(testEdges, topKSuggestions, K, outFile)