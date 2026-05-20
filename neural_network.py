import random
from autograd import Value


class Neuron:

    def __init__(self, n_in: int):
        self.ws = [Value(random.uniform(-1.0, 1.0)) for n in range(n_in)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x: list[float]) -> Value:
        sum_in = sum((wi * xi for wi, xi in zip(self.ws, x)), self.b)
        relu = sum_in.relu()
        return relu
    
class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x: list[float]) -> list[Value]:
        return [n(x) for n in self.neurons]
    
class MLP:
    def __init__(self, n_in: int, layer_sizes: list[int]):
        all_layer_sizes = [n_in, *layer_sizes]
        self.layers = [Layer(all_layer_sizes[i], all_layer_sizes[i+1]) for i in range(len(all_layer_sizes) - 1)]

    def __call__(self, x:list[float]) -> list[float]:
        for l in self.layers:
            x = l(x)
        return x


    

    


        