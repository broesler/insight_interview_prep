#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: graph_search.py
#  Created: 07/12/2018, 15:44
#   Author: Bernie Roesler
#
"""
  Description: Graph search algorithms
"""
#==============================================================================

class GraphNode():
    def __init__(self, node_id):
        self.id = node_id
        self.next = []

    def __equiv__(self, b):
        return self.id == b.id

    def __repr__(self):
        ids = [n.id for n in self.next]
        return "<GraphNode: {{id: {}, next: {}}}>".format(self.id, ids)

def depth_first(a, b):
    """Given two nodes, traverse the list depth-first until we find a path."""
    # If we are at the target node, stop searching
    if a == b:
        # return [b]
        return True

    for n in a.next:
        # Search a level deeper until there are no more out-edges
        ret = depth_first(n, b)
        # If we've found a path to the target, return it
        if ret:
            # return [a] + ret
            return True

    # We have exhausted all paths without finding the target
    return False

def breadth_first(a, b):
    """Given two nodes, traverse the list breadth-first until we find a path."""
    # If we are at the target node, stop searching
    if a == b:
        return True

    for n in a.next:
        if n == b:
            return True
        return any([breadth_first(n, b) for n in a.next])

    # We have exhausted all paths without finding the target
    return False

# Build the graph
G = [GraphNode(i) for i in range(6)]
G[0].next = [G[x] for x in [1, 2, 4]]
G[1].next = [G[3]]
G[2].next = [G[4], G[3]]
G[4].next = [G[5]]

# Test code
p1 = depth_first(G[0], G[3]) # [1, 2, 4]
p2 = depth_first(G[0], G[4]) # [0, 2, 4]

p3 = breadth_first(G[0], G[4]) # [0, 4]
p4 = breadth_first(G[1], G[4]) # [0, 4]

print(p1, p2, p3, p4)
#==============================================================================
#==============================================================================
