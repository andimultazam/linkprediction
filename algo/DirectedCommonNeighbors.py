from pyspark import SparkConf, SparkContext
import Utils
from NodeSuggestion import NodeSuggestion

def getNodeSuggestions(nodePair):
    result = []
    result.append((nodePair.getId1(), NodeSuggestion(nodePair.getId2(), len(nodePair.getCommonNeighbors()))))
    result.append((nodePair.getId2(), NodeSuggestion(nodePair.getId1(), len(nodePair.getCommonNeighbors()))))
    return result

#Create SparkContext
conf = SparkConf()\
    .setAppName("Link Prediction Directed-Common-Neighbors Algorithm")
sc = SparkContext(conf=conf)

#Input parameters
#src = sys.argv[1]
src = '/home/kienguye/NUS/BigData/FinalProject/twitter_t1'
K = 10

trainFolder = src + "/train"
testFolder = src + "/test"
masterTrainEdgeFile = trainFolder+"/master.edges"
masterTestEdgeFile = testFolder+"/master.edges"
outFile = trainFolder+"/nodePair_directedCommonNeighbors.result"
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
nodePairs = Utils.getDirectedCommonNeighbors(edges)
# print("----------10 commonNeighbors-----------------")
# for commonNeighbor in commonNeighbors.take(10):
#     print(commonNeighbor)
####### Done Get rdd of (nodeId1-nodeId2, commonNeighbors-set) ########


##########Step 3: Compute suggestion for each node###############
nodeSuggestions = nodePairs.map(lambda nodePair: (nodePair.getId1(), NodeSuggestion(nodePair.getId2(), len(nodePair.getCommonNeighbors()))))
nodeSuggestions = Utils.removeExistingSuggestions(nodeSuggestions, edges)
print("----------10 nodeSuggestions-----------------")
for i in nodeSuggestions.take(10):
    print("NodeId: {}, {}".format(i[0], i[1]))

#rdd of (key, value) - (nodeId, list of topK suggestionIds)
topKSuggestions = Utils.getTopKSuggestions(nodeSuggestions, K)

print("----------10 topKSuggestions-----------------")
for i in topKSuggestions.take(10):
    print(i)
##########Done Compute suggestion for each node###############

#########Step 3: Validate result###########
testEdges = Utils.getEdges(sc, masterTestEdgeFile)
Utils.validateResult(testEdges, topKSuggestions, K, outFile)