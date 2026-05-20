


from vis import render


class Value:
    def __init__(self : Value, data : float, _prev : tuple[Value]=(), _op : str='', label: str='', is_param=False):
        self.data = data
        self._prev = set(_prev)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self.is_param = is_param
        self._forward = lambda: None
        self._meta = {} # for vis

    def forward(self):
        for p in self.topological_order():
            p._forward()

    

    def topological_order(self) -> list['Value']:
        order = []
        visited = set()

        def dfs(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                dfs(child)
            order.append(node)

        dfs(self)
        return order

    def backward(self, reset = True):
        if reset:
            self._reset_grads()
            self.grad = 1.0

        self._backward()

        for child in self._prev:
            child.backward(False)
    
    def _reset_grads(self):
        self.grad = 0.0
        for child in self._prev:
            child._reset_grads()

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(x, y):
        out = Value(x.data + y.data, _prev=(x, y), _op='+')
        
        def backward():
            x.grad = x.grad + out.grad
            y.grad = y.grad + out.grad
        out._backward = backward

        def forward():
            out.data = x.data + y.data
        out._forward = forward

        return out
    
    def __neg__(x):

        out = Value(-x.data, _prev=(x,), _op='-')

        def backward():
            x.grad = x.grad - out.grad
        out._backward = backward
        
        def forward():
            out.data = -x.data
        out._forward = forward

        return out

    def __rmul__(self, y):
        return self * y

    def __mul__(x, y):
        if not isinstance(y, Value): y = Value(y)
        out = Value(x.data * y.data, _prev=(x, y), _op='*')

        def backward():
            x.grad = x.grad + out.grad * y.data
            y.grad = y.grad + out.grad * x.data
        out._backward = backward

        def forward():
            out.data = x.data * y.data
        out._forward = forward


        return out
    
    def square(x):
        out = Value(x.data ** 2, _prev=(x,), _op='^2')

        def backward():
            x.grad = x.grad + out.grad * 2 * x.data
        out._backward = backward

        def forward():
            out.data = x.data ** 2
        out._forward = forward

        return out
    
    def sqrt(x):
        out = Value(x.data ** 0.5, _prev=(x,), _op='sqrt')

        def backward():
            x.grad = x.grad + out.grad / (2 * x.data ** 0.5)
        out._backward = backward

        def forward():
            out.data = x.data ** 0.5
        out._forward = forward

        return out

    

    def print_vis(self, hide_without_labels=False):
        print(render(self, hide_without_labels))


    def update(self, h : float):
        for p in reversed(self.topological_order()):
            if p.is_param:
                p.data -= h * p.grad


    def relu(child):
        out = Value(max(0, child.data), _prev=(child,), _op='relu')

        def backward():
            if out.data > 0:
                child.grad = child.grad + out.grad
        out._backward = backward

        def forward():
            out.data = max(0, child.data)
        out._forward = forward

        return out


   
        

