# pylbfgs
A simple Python binding for [libLBFGS](http://www.chokkan.org/software/liblbfgs).

	import pylbfgs
	import numpy as np

	def evaluate(instance, x, n, step):

	    # sphere function
	    fx = np.sum(x * x)
	    g = 2 * x

	    return fx, g

	x = np.random.normal(size=(10,))

	print("BEFORE: " + str(x))

	param = pylbfgs.default_params()
	param.epsilon = 1e-30
	param.past = 3

	pylbfgs.lbfgs(x, evaluate)

	print("AFTER: " + str(x))

## Installation

No setup.py yet, but you can simply add the `pylbfgs` directory into your project after compiling libLBFGS:

	1. clone this repo and `cd` into it
	2. `git submodule init && git submodule update`
	3. `cd pylbfgs/liblbfgs`
	4. `./autogen.sh && ./configure.sh && make`


## License
MIT
