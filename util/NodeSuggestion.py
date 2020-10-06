class NodeSuggestion:
    def __init__(self, suggestionId, sim):
        self.suggestionId = suggestionId
        self.sim = sim

    def getSuggestionId(self):
        return self.suggestionId

    def getSim(self):
        return self.sim

    def __str__(self):
        return "NodeSuggestion: ({} - {})".format(self.suggestionId, self.sim)

    def __repr__(self):
        return str(self)