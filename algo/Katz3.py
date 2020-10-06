from pyspark import SparkConf, SparkContext, StorageLevel
import Utils
from Edge import Edge
from Path import Path
from NodeSuggestion import NodeSuggestion

def addNodeToPath(oldPath, nodeId):
    newNodeIds = oldPath.getNodeIds().copy()
    newNodeIds.append(nodeId)
    return Path(oldPath.getStartId(), nodeId, newNodeIds)

def flatMapNextLevelPaths(path, bc):
    paths = []
    nodeFollowings = bc.value.get(path.getEndId())
    if nodeFollowings is None:
        return paths
    nodesToAdd = set(nodeFollowings).difference(frozenset(path.getNodeIds()))
    for nodeId in nodesToAdd:
        newPath = addNodeToPath(path, nodeId)
        paths.append(newPath)
    return paths

    return paths
def extractNextLevelPaths(previousLevelPaths, bc):
    nextLevelPaths = previousLevelPaths \
        .flatMap(lambda path: flatMapNextLevelPaths(path, bc))
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

#Create SparkContext
conf = SparkConf()\
    .setAppName("Link Prediction Katz 2 Algorithm")
sc = SparkContext(conf=conf)

#Input parameters
#src = sys.argv[1]
src = '/home/kienguye/NUS/BigData/FinalProject/twitter_t1'
BETA=0.8
K = 10

trainFolder = src + "/train"
testFolder = src + "/test"
masterTrainEdgeFile = trainFolder+"/master.edges"
masterTestEdgeFile = testFolder+"/master.edges"
outFile = trainFolder+"/nodePair_katz2.result"
Utils.removeExistingDir(outFile)

####### Step 1: get rdd of edges ########
#Get rdd of edges (nodeId, following-node-id)
edges = Utils.getEdges(sc, masterTrainEdgeFile).cache()

# #########Step 2: Get top-3K suggestions base on CommonNeighbors############
# nodeNeighbors = Utils.getNeighbors(edges)
# nodePairs = Utils.getCommonNeighbors(nodeNeighbors)
# nodeSuggestions = nodePairs.flatMap(lambda nodePair: CommonNeighbors.getNodeSuggestions(nodePair))
#
# #rdd of (key, value) - (nodeId, list of top-3K suggestionIds)
# top3KSuggestions = Utils.getTopKSuggestions(nodeSuggestions, 3*K)
#
# ####### Step 3: for each nodeId, compute Katz similarity for 3K suggestions ########
# top3KSuggestions.map(lambda node: )
#
# ##########Done Compute Katz similairty###############

####### Step 2: Get all paths with length 2-6 ########
#rdd of (nodeId, set of followings)
nodeFollowings = Utils.getNodeFollowings(edges).collectAsMap()
bc = sc.broadcast(nodeFollowings)

#rdd of (startId, Edge)
edgeWithStartIdAsKey = edges.map(lambda edge: (edge[0], Edge(edge[0], edge[1]))).cache()
print("edgeWithStartIdAsKey")
for edge in edgeWithStartIdAsKey.take(10):
    print(edge)

#rdd of all Path of length 1
length1Paths = edges.map(lambda edge: Path(edge[0], edge[1], [edge[0], edge[1]])).persist(StorageLevel.MEMORY_AND_DISK_SER)
# print("length1Paths")
# for path in length1Paths.take(10):
#     print(path)
#rdd of ((startId, endId), 1)
length1PathCounts = edges.map(lambda edge: ((edge[0], edge[1]), 1)).persist(StorageLevel.MEMORY_AND_DISK_SER)
edges.unpersist()

length2Paths = extractNextLevelPaths(length1Paths, bc).persist(StorageLevel.MEMORY_AND_DISK_SER)
# length1Paths.unpersist()
length2PathCounts = getPathCounts(length2Paths, length1PathCounts, 2).persist(StorageLevel.MEMORY_AND_DISK_SER)
# print("length2Paths")
# for path in length2Paths.take(10):
#     print(path)


length3Paths = extractNextLevelPaths(length2Paths, bc).persist(StorageLevel.MEMORY_AND_DISK_SER)
# length2Paths.unpersist()
length3PathCounts = getPathCounts(length3Paths, length1PathCounts, 3).persist(StorageLevel.MEMORY_AND_DISK_SER)
# print("length3Paths")
# for path in length3Paths.take(10):
#     print(path)

length4Paths = extractNextLevelPaths(length3Paths, bc).persist(StorageLevel.MEMORY_AND_DISK_SER)
# length3Paths.unpersist()
length4PathCounts = getPathCounts(length4Paths, length1PathCounts, 4).persist(StorageLevel.MEMORY_AND_DISK_SER)
# print("length4Paths")
# for path in length4Paths.take(10):
#     print(path)

# length5Paths = extractNextLevelPaths(length4Paths, bc).cache()
# length4Paths.unpersist()
# length5PathCounts = getPathCounts(length5Paths, length1PathCounts, 5)
# print("length5Paths")
# for path in length5Paths.take(10):
#     print(path)

# length6Paths = extractNextLevelPaths(length5Paths, bc)
# length5Paths.unpersist()
# length6PathCounts = getPathCounts(length6Paths, length1PathCounts, 6)
# print("length6Paths")
# for path in length6Paths.take(10):
#     print(path)

##########Step 3 Compute suggestion for each node##########
# .union(length5PathCounts).union(length6PathCounts)
allPathCounts = length2PathCounts.union(length3PathCounts).union(length4PathCounts)
nodeSuggestions = allPathCounts.groupByKey()\
    .map(lambda pathCount: getNodeSuggestion(pathCount, BETA))

#rdd of (key, value) - (nodeId, list of topK suggestionIds)
topKSuggestions = Utils.getTopKSuggestions(nodeSuggestions, K).cache()

print("----------10 topKSuggestions-----------------")
for i in topKSuggestions.take(10):
    print(i)
##########Done Compute suggestion for each node###############

# #########Step 5: Validate result###########
testEdges = Utils.getEdges(sc, masterTestEdgeFile)
Utils.validateResult(testEdges, topKSuggestions, K, outFile)