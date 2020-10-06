from pyspark import SparkConf, SparkContext
import Utils


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

def setup_pyspark():
    conf = SparkConf() \
        .setAppName("Link Prediction CNCF Algorithm")
    sc = SparkContext(conf=conf)
    return sc

if __name__ == "__main__":
    # Create SparkContext
    sc = setup_pyspark()

    # Input parameters
    # src = sys.argv[1]
    src = 'data/twitter_t1'
    trainFolder = src + "/train"
    testFolder = src + "/test"
    masterTrainEdgeFile = trainFolder + "/master.edges"

    #Find nearest neighbours
    get_neighbours(sc, masterTrainEdgeFile)
