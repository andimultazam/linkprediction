class Path:
    #nodeIds is a list of all nodeIds in the path
    def __init__(self, startId, endId, nodeIds):
        self.startId = startId
        self.endId = endId
        self.nodeIds = nodeIds

    def getStartId(self):
        return self.startId

    def getEndId(self):
        return self.endId

    def getNodeIds(self):
        return self.nodeIds

    def __str__(self):
        return "Path from {} to {}: {}".format(self.startId, self.endId, self.nodeIds)

    def __repr__(self):
        return str(self)