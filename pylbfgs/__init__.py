import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct
import os.path

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

class LBFGSParameter(ct.Structure):
    """
    LBFGS parameters.

    See http://www.chokkan.org/software/liblbfgs/structlbfgs__parameter__t.html for documentation.
    """
    _fields_ = [
        ('m', ct.c_int),
        ('epsilon', ct.c_double),
        ('past', ct.c_int),
        ('delta', ct.c_double),
        ('max_iterations', ct.c_int),
        ('linesearch', ct.c_int),
        ('max_linesearch', ct.c_int),
        ('min_step', ct.c_double),
        ('max_step', ct.c_double),
        ('ftol', ct.c_double),
        ('wolfe', ct.c_double),
        ('gtol', ct.c_double),
        ('xtol', ct.c_double),
        ('orthantwise_c', ct.c_double),
        ('orthantwise_start', ct.c_int),
        ('orthantwise_end', ct.c_int)
    ]

evaluate_func = ct.CFUNCTYPE(
    ct.c_double,                # return value
    ct.c_voidp,                 # instance
    ct.POINTER(ct.c_double),    # *x
    ct.POINTER(ct.c_double),    # *g
    ct.c_int,                   # n
    ct.c_double                 # step
)

progress_func = ct.CFUNCTYPE(
    ct.c_int,                   # return value
    ct.c_voidp,                 # instance
    ct.POINTER(ct.c_double),    # *x
    ct.POINTER(ct.c_double),    # *g
    ct.c_double,                # fx
    ct.c_double,                # xnorm
    ct.c_double,                # gnorm
    ct.c_double,                # step
    ct.c_int,                   # n
    ct.c_int,                   # k
    ct.c_int                    # ls
)


liblbfgs = npct.load_library('liblbfgs', os.path.join(os.path.dirname(__file__), 'liblbfgs/lib/.libs'))
liblbfgs.lbfgs.restype = ct.c_int
liblbfgs.lbfgs.argtypes = [
    ct.c_int,                   # n
    array_1d_double,            # *x
    ct.POINTER(ct.c_double),    # *fx
    ct.c_voidp,                 # proc_evaluate
    ct.c_voidp,                 # proc_progress
    ct.c_voidp,                 # *instance
    ct.POINTER(LBFGSParameter)  # *param
]


liblbfgs.lbfgs_parameter_init.restype = ct.c_voidp
liblbfgs.lbfgs_parameter_init.argtypes = [
    ct.POINTER(LBFGSParameter)
]


PROGRESS_LINE   = "{k:4d} {ls:4d} {fx:.10e} {xnorm:.5e} {gnorm:.5e} {step:.3e}"

@progress_func
def progress(instance, x, g, fx, xnorm, gnorm, step, n, k, ls):
    """
    Default progress function.

    Displays, in order:
        - the current iteration
        - the number of line search steps in that iteration
        - the current function value
        - the L2 norm of all variables
        - the L2 norm of all gradients
        - the step size
    """

    ## if you want to access x, g, and instance, you can use these to convert:
    #x = npct.as_array(x, (n,))
    #g = npct.as_array(g, (n,))
    #instance = instance.contents

    print(PROGRESS_LINE.format(k=k, ls=ls, fx=fx, xnorm=xnorm, gnorm=gnorm, step=step))
    return 0

def default_params():
    """Initialize a LBFGSParameter object with default parameters."""
    param = LBFGSParameter()
    liblbfgs.lbfgs_parameter_init(param)
    return param

def lbfgs(x, evaluate, instance=None, param=default_params(), progress=progress):

    @evaluate_func
    def eval_wrapper(instance, x, g, n, step):
        """Wrapper function to handle converting to numpy data structures"""
        x = npct.as_array(x, (n,))
        g = npct.as_array(g, (n,))

        if instance:
            instance = instance.contents
        else:
            instance = None

        fx, gret = evaluate(instance, x, n, step)
        g[:] = gret
        return fx


    fx = ct.c_double()

    instance = ct.byref(instance) if instance != None else None
    code = liblbfgs.lbfgs(x.size, x, ct.byref(fx), eval_wrapper, progress, instance, param)

    return code, fx.value, x
