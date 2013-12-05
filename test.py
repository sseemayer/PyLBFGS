import pylbfgs
import numpy as np

def evaluate(instance, x, n, step):

    # sphere function
    fx = np.sum(x * x)
    g = 2 * x

    return fx, g

np.set_printoptions(linewidth=200, formatter={'float': lambda x: "{0:.3f}".format(x)})
np.random.seed(42)

x = np.random.normal(size=(10,))

print("BEFORE: " + str(x))

param = pylbfgs.default_params()
param.epsilon = 1e-30
param.past = 3

code, fx, x = pylbfgs.lbfgs(x, evaluate)

print("Optimization exited with code {code} and function value {fx:.5f}".format(code=code, fx=fx))

print("AFTER: " + str(x))
