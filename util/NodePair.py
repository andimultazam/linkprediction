import math
class NodePair:
    def __init__(self, id1, id2, commonNeighbors):
        self.id1 = id1
        self.id2 = id2
        self.commonNeighbors = commonNeighbors

    def getId1(self):
        return self.id1

    def getId2(self):
        return self.id2

    def getCommonNeighbors(self):
        return self.commonNeighbors

    def computeAdamicAdar(self, bc):
        degreeMap = bc.value
        sim = 0.0
        for v in self.commonNeighbors:
            vDegree = degreeMap.get(v)
            sim = sim + 1/math.log(vDegree)
        return sim

    def __eq__(self, o: object) -> bool:
        if isinstance(o, NodePair):
            return ((self.id1 == o.id1) and (self.id2 == o.id2))
        return False

    def __hash__(self):
        return hash((self.id1, self.id2))

    def __str__(self):
        return "NodePair: ({}, {}) - {}".format(self.id1, self.id2, self.commonNeighbors)

    def __repr__(self):
        return str(self)