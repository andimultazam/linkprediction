class ThresholdResult:
    def __init__(self, id, suggestions, testFollowings, totalNodeCount, nodeFollowingCount):
        self.id = id
        self.suggestions = suggestions
        self.testFollowings = testFollowings
        self.correctSuggestions = set(suggestions).intersection(testFollowings)
        self.tp = len(self.correctSuggestions)
        self.fp = len(suggestions) - self.tp
        #self.tn = totalNodeCount - nodeFollowingCount - self.fp
        self.tn = 100 - self.fp
        self.fn = len(testFollowings) - self.tp

    def getId(self):
        return self.id

    def getSuggestions(self):
        return self.suggestions

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
        return "Result: (NodeId: {}, tp: {}, fp: {}, tn: {}, fn: {}, correctSuggestions: {}, suggestions: {}"\
            .format(self.id, self.tp, self.fp, self.tn, self.fn, self.correctSuggestions, self.suggestions)


    def toSaveString(self):
        return "{},{},{},{},{}".format(self.id, self.tp, self.fp, self.tn, self.fn)

    def __repr__(self):
        return str(self)