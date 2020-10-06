from pyspark import SparkConf, SparkContext
import Utils


#'file:/home/kienguye/NUS/BigData/FinalProject/twitter_t1/train/64496469.edges/part-00001'
def getEgoNodeId(filename):
    parts = filename.split("/")
    for part in parts:
        if(part.endswith(".edges")):
            return part.split(".")[0]
    return ""

def getEdgeFromLine(edge):
    nodes = edge.split()
    if(len(nodes)>=2):
        return (nodes[0], nodes[1])
    return None

def getAllEdges(edgeFiles, masterEdgeFiles):
    lines = sc.wholeTextFiles(edgeFiles).map(lambda t: (getEgoNodeId(t[0]), t[1])).flatMapValues(lambda text: text.split("\n")).cache()
    # print("10 lines from edges files...")
    # for i in lines.take(10):
    #     print(i)
    normalEdges = lines.values().map(lambda edge: getEdgeFromLine(edge)).filter(lambda edge: edge != None)
    # print("10 normal edges from edges files...")
    # for i in normalEdges.take(10):
    #     print(i)
    #egoEdges = lines.flatMapValues(lambda edge: edge.split()).groupByKey().mapValues(lambda v: list(dict.fromkeys(v))).flatMapValues(lambda v: v).filter(lambda edge: edge != None)
    # print("10 ego edges from edges files...")
    # for i in egoEdges.take(10):
    #     print(i)
    #.union(egoEdges)
    edges = normalEdges.map(lambda t: "{} {}".format(t[0], t[1]))
    edges.saveAsTextFile(masterEdgeFiles)

#Create SparkContext
conf = SparkConf()\
    .setAppName("Link Prediction Preprocess Program")
sc = SparkContext(conf=conf)

#Input parameters
#src = sys.argv[1]
src = '/home/kienguye/NUS/BigData/FinalProject/twitter_t1'

trainFolder = src + "/train"
testFolder = src + "/test"
trainEdgeFiles = trainFolder+"/*.edges"
testEdgeFiles = testFolder+"/*.edges"
masterTrainEdgeFile = trainFolder+"/master.edges"
masterTestEdgeFile = testFolder+"/master.edges"

#Get master edges files for train and test data
Utils.removeExistingDir(masterTrainEdgeFile)
Utils.removeExistingDir(masterTestEdgeFile)
getAllEdges(trainEdgeFiles, masterTrainEdgeFile)
getAllEdges(testEdgeFiles, masterTestEdgeFile)

edges = Utils.getEdges(sc, masterTrainEdgeFile).cache()

#Save nodeid_followings to file
nodeFollowingsFile = trainFolder+"/nodeId.followings"
nodeFollowings = Utils.getNodeFollowings(edges)
Utils.removeExistingDir(nodeFollowingsFile)
nodeFollowings.saveAsTextFile(nodeFollowingsFile)


#Save nodeId_followers to file
nodeFollowersFile = trainFolder+"/nodeId.followers"
nodeFollowers = Utils.getNodeFollowers(edges)
Utils.removeExistingDir(nodeFollowersFile)
nodeFollowers.saveAsTextFile(nodeFollowersFile)