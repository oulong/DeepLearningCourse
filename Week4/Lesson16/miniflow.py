import numpy as np

class Node(object):

    def __init__(self, inbound_nodes=[]):
        
        self.inbound_nodes = inbound_nodes

        self.outbound_nodes = []

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        self.value = None

    def forward(self):
        raise NotImplemented

class Input(Node):

    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        
        if value is not None:
            self.value = value

class Add(Node):

    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        self.value = 0
        for i in range(len(self.inbound_nodes)):
            self.value += self.inbound_nodes[i].value
        return self.value 

class Sigmoid(Node):

    def __init__(self, n):
        Node.__init__(self, [n])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value) 
        return self.value
    
class Linear(Node):

    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value
        self.value = sum(map(lambda x,y:x*y, inputs,weights)) + bias 
        return self.value
    
class LinearMatrix(Node):

    def __init__(self, X, W, B):
        Node.__init__(self, [X, W, B])

    def forward(self):

        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        B = self.inbound_nodes[2].value

        self.value = X.dot(W) + B 
        return self.value

class MSE(Node):

    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):

        y = self.inbound_nodes[0].value.reshape(-1,1)
        a = self.inbound_nodes[1].value.reshape(-1,1)

        #self.value =  sum(np.square(y - a)) / len(y)
        self.value = np.mean(np.square(y - a))

'''
Kahn's algorihtm
'''
def topological_sort(feed_dict):
    
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        G.setdefault(n, {'in':set(), 'out': set()})

        for it in n.outbound_nodes:
            G.setdefault(it, {'in':set(), 'out':set()})
            G[n]['out'].add(it)
            G[it]['in'].add(n)

            nodes.append(it)

    L = []
    S = set(input_nodes)
    while (len(S) > 0):
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if  len(G[m]['in']) == 0:
                S.add(m)
    #if G:
        #raise RuntimeError("graph has at least one cycle")
        
    return L

def forward_pass(output_node, sorted_nodes):

    for n in sorted_nodes:
        n.forward()

    return output_node.value

def forward_pass_no_ret(sorted_nodes):

    for n in sorted_nodes:
        n.forward()

