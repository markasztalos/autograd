import random
import statistics

from autograd import Value
from neural_network import MLP, Layer, Neuron
from vis import vis_2d, vis_neuron

SEED = 42
random.seed(SEED)


def demonstrate_autograd():
    SEED = 42
    MAX = 1000
    f = lambda x: 3*x + 8
    xs = [x for x in range(1, MAX)]
    ys = [f(x) * random.uniform(.9, 1.1) for x in xs]

    x_mean = statistics.mean(xs)
    x_max = max([abs(x) for x in xs])
    learning_rate=0.01

    # option 1 rebuild always
    # w = Value(1.0, label='w', is_param=True)
    # b = Value(1.0, label='b', is_param=True)
    # for ep in range(150):
    #     for i, x in enumerate(xs):
    #         y = ys[i]
    #         x = Value((x-x_mean) / x_max, label='x')
    #         y = Value(y, label='y')
    #         model = w * x + b
    #         model.label = 'model'
    #         error = (model + (-y)).square()
    #         error.label = 'error'
    #         error.backward()
    #         error.update(learning_rate)
    #     print(f"epoh #{ep}: w={w.data/x_max} b={b.data-w.data*x_mean/x_max}")
    # error.print_vis()

    # option 2 reuse
    w = Value(1.0, label='w', is_param=True)
    b = Value(1.0, label='b', is_param=True)
    y = Value(1.0, label='y')
    x = Value(1.0, label='x')
    model = w * x + b
    model.label = 'model'
    error = (model + (-y)).square()
    error.label = 'error'
    
    for ep in range(150):
        for i, xv in enumerate(xs):
            y.data = ys[i]
            x.data = (xv-x_mean) / x_max
            error.forward()
            error.backward()
            error.update(learning_rate)
        # print(f"epoh #{ep}: w={w.data/x_max} b={b.data-w.data*x_mean/x_max}")

    # the solved linear modell
    w_model=w.data/x_max
    b_model=b.data-w.data*x_mean/x_max

    print(f"Model: y = {w_model:.2f} * x + {b_model:.2f}")

    print(vis_2d(xs, ys, w_model, b_model))


def demonstrate_neural_networks():
    x = [8, 2, 3]
    n = Neuron(3)
    # print(f"{n(x):.2f}")
    # l = Layer(3, 2)
    # print(f"{l(x)}")
    # nn = MLP(3, [4, 4, 1])
    # print(f"{nn(x)}")
    


def main():
    # demonstrate_autograd()
    demonstrate_neural_networks()


if __name__ == "__main__":
    main()
