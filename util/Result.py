class Result:
    #testCount is the number of nodes that this node is following in the test dataset.
    def __init__(self, id, kSuggestions, testFollowings, k, totalNodeCount, nodeFollowingCount):
        self.id = id
        self.kSuggestions = kSuggestions
        self.testFollowings = testFollowings
        self.correctSuggestions = set(kSuggestions).intersection(testFollowings)
        self.noOfCorrectSuggestion = len(self.correctSuggestions)
        self.maxCorrectSuggestion = min(k, len(testFollowings))
        self.tp = len(self.correctSuggestions)
        self.fp = len(kSuggestions) - self.tp
        self.tn = totalNodeCount - nodeFollowingCount - self.fp
        # self.tn = k - self.fp
        #self.fn = min(k, len(testFollowings)) - self.tp
        self.fn = min(k, len(testFollowings)) - self.tp

    def getId(self):
        return self.id

    def getKSuggestions(self):
        return self.kSuggestions

    def getTestFollowings(self):
        return self.testFollowings

    def getCorrectSugestions(self):
        return self.correctSuggestions

    def getTP(self):
        return self.tp

    def getFP(self):
        return self.fp

    def getTN(self):
        return self.tn

    def getFN(self):
        return self.fn

    def __str__(self):
        return "Result: (NodeId: {}, tp: {}, fp: {}, tn: {}, fn: {}, correctSuggestions: {}, kSuggestions: {}"\
            .format(self.id, self.tp, self.fp, self.tn, self.fn, self.correctSuggestions, self.kSuggestions)


    def toSaveString(self):
        return "{},{},{},{},{}".format(self.id, self.tp, self.fp, self.tn, self.fn)

    def __repr__(self):
        return str(self)