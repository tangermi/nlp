import scipy
import scipy.constants
import math


def constants():
    print("sciPy - pi = %.16f" % scipy.constants.pi)
    print("math - pi = %.16f" % math.pi)

    res = scipy.constants.physical_constants["alpha particle mass"]
    print(res)


if __name__ == '__main__':
    constants()
