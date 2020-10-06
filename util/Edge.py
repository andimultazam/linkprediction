class Edge:
    #id1 follows id2
    def __init__(self, id1, id2):
        self.id1 = id1
        self.id2 = id2

    def getId1(self):
        return self.id1

    def getId2(self):
        return self.id2

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Edge):
            return ((self.id1 == o.id1) and (self.id2 == o.id2))
        return False

    def __hash__(self):
        return hash((self.id1, self.id2))

    def __str__(self):
        return "Edge: ({} {})".format(self.id1, self.id2)

    def __repr__(self):
        return str(self)