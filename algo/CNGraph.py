from Edge import Edge
import math
class CNGraph:
    def __init__(self, nodePair, edges):
        self.nodePair = nodePair
        self.edges = edges

    def getNodePair(self):
        return self.nodePair

    def getEdges(self):
        return self.edges

    def getSim(self):
        return self.sim

    def __str__(self):
        return "({}, {}) - {} - {}".format(self.nodePair.getId1(), self.nodePair.getId2(), self.nodePair.getCommonNeighbors(), self.edges)

    def __repr__(self):
        return str(self)

    def printSum(self):
        return "({} - {}) has similarity of {}".format(self.nodePair.getId1(), self.nodePair.getId2(), self.sim)

    def calculateSimilarity(self, bc):
        degreeMap = bc.value
        sim = 0.0
        for v in self.nodePair.getCommonNeighbors():
            vCommonDegree = 0.0
            for edge in self.edges:
                if (v==edge.getId1() or v== edge.getId2()):
                    vCommonDegree = vCommonDegree + 1
            vDegree = degreeMap.get(v)
            sim = sim + vCommonDegree/math.log(vDegree)
        self.sim = sim
        return sim

    def calculateDirectedSimilarity(self, fromNode, toNode, bc):
        followingCountMap = bc.value
        sim = 0.0
        for v in self.nodePair.getCommonNeighbors():
            vCommonDegree = 0.0
            if Edge(fromNode, v) in self.edges and Edge(v, toNode) in self.edges:
                for edge in self.edges:
                    if (v == edge.getId1()):
                        vCommonDegree = vCommonDegree + 1
                vFollowingCount = followingCountMap.get(v)
                if(vFollowingCount>0):
                    sim = sim + vCommonDegree/vFollowingCount
        return sim

    def computeNextLevelPaths(self, previousLevelPath, nodeFollowings, length):
        nextLevelpaths = []
        for path in previousLevelPath:
            endId = path[length-1]
            followingNodeIds = nodeFollowings.get(endId)
            if followingNodeIds is not None:
                for nodeId in followingNodeIds:
                    if nodeId not in path:
                        newPath = path.copy()
                        newPath.append(nodeId)
                        nextLevelpaths.append(newPath)
        return nextLevelpaths

    def computePathCounts(self, paths, fromId, toId):
        count = 0
        for path in paths:
            if path[0] == fromId and path[len(path)-1] == toId:
                count = count+1
        return count

    def calculateSimilarityBasedOnPaths(self, BETA, fromId, toId):
        nodeFollowings = {}
        for edge in self.edges:
            if nodeFollowings.get(edge.getId1()) is None:
                followingList = [edge.getId2()]
                nodeFollowings[edge.getId1()]= followingList
            else:
                followingList = nodeFollowings.get(edge.getId1())
                followingList.append(edge.getId2())
                nodeFollowings[edge.getId1()] = followingList
        #Compute all paths of len 1-6
        #all paths are saved as list of nodeIds
        length1Paths = []
        for edge in self.edges:
            length1Paths.append([edge.getId1(), edge.getId2()])
        length2Paths = self.computeNextLevelPaths(length1Paths, nodeFollowings, 2)
        # length3Paths = self.computeNextLevelPaths(length2Paths, nodeFollowings, 3)
        # length4Paths = self.computeNextLevelPaths(length3Paths, nodeFollowings, 4)
        # length5Paths = self.computeNextLevelPaths(length4Paths, nodeFollowings, 5)
        # length6Paths = self.computeNextLevelPaths(length5Paths, nodeFollowings, 6)

        #Compute similarity
        sim = 0.0
        length2PathCounts = self.computePathCounts(length2Paths, fromId, toId)
        # length3PathCounts = self.computePathCounts(length3Paths, fromId, toId)
        # length4PathCounts = self.computePathCounts(length4Paths, fromId, toId)
        # length5PathCounts = self.computePathCounts(length5Paths, fromId, toId)
        # length6PathCounts = self.computePathCounts(length6Paths, fromId, toId)
        sim = sim + pow(BETA, 2)*length2PathCounts \
              # + pow(BETA, 3)*length3PathCounts \
              # + pow(BETA, 4)*length4PathCounts \
              # + pow(BETA, 5)*length5PathCounts + pow(BETA, 6)*length6PathCounts
        return sim

