import numpy as np

class Node(object):

    def __init__(self, inbound_nodes=[]):
        
        self.inbound_nodes = inbound_nodes

        self.outbound_nodes = []

        self.gradients = {}

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        self.value = None

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Input(Node):

    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        pass

    def backward(self):
        
        self.gradients = {self: 0}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1
        
class Sigmoid(Node):

    def __init__(self, n):
        Node.__init__(self, [n])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value) 

    def backward(self):

        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += \
                    self.value * (1 - self.value) * grad_cost 
    
class Linear(Node):

    def __init__(self, X, W, B):
        Node.__init__(self, [X, W, B])

    def forward(self):

        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        B = self.inbound_nodes[2].value

        self.value = X.dot(W) + B 

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]

            self.gradients[self.inbound_nodes[0]] += \
                    np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += \
                    np.dot(self.inbound_nodes[0].value.T, grad_cost)

            self.gradients[self.inbound_nodes[2]] += \
                    np.sum(grad_cost, axis=0, keepdims=False)


class MSE(Node):

    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):

        y = self.inbound_nodes[0].value.reshape(-1,1)
        a = self.inbound_nodes[1].value.reshape(-1,1)

        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a

        #self.value =  sum(np.square(y - a)) / len(y)
        self.value = np.mean(np.square(self.diff))

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff 
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

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

def forward_and_backward(graph):

    for n in graph:
        n.forward()


    for n in graph[::-1]:
        n.backward()

