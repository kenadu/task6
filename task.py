import numpy as np
import matplotlib.pylab as plt


def FD_1st_order(f, x, h=1e-4, fder=None, filename=None):
    '''compute 1st order derivative of f(x) using FFD, CFD, BFD. 
       tasks: 
       (1) output f'(x) in a tuple named (ffd, cfd, bfd), where ffd, cfd, bfd store the 
           f'(x) obtained by FFD, CFD, and BFD;
       (2) call FD_plot() to do the plotting:
           (2.1) when exact f'(x) is passed as input via fder, need to pass it on so that
                 a curve of exact dervative will be plotted in addition to FD curves
           (2.2) when filename is passed as input, need to pass it on so that the plot will 
                 be saved to a png file using a modification of this filename
    '''
    x = np.array(x)  # this makes sure elementwise operations such as x+h and x-h are valid

    ffd = (f(x + h) - f(x)) / h
    bfd = (f(x) - f(x - h)) / h
    cfd = (f(x + h) - f(x - h)) / (2 * h)

    FD_plot(x, h, ffd, cfd, bfd, fderivative=fder, FD_order='1st', filename=filename)
    return (ffd, cfd, bfd)


def FD_2nd_order(f, x, h=1e-4, fder2=None, filename=None):
    '''compute 2nd order derivative of f(x) using FFD, CFD, BFD. 
       tasks: 
       (1) output f''(x) in a tuple named (ffd, cfd, bfd), where ffd, cfd, bfd store the 
           f''(x) obtained by FFD, CFD, and BFD;
       (2) call FD_plot() to do the plotting:
           (2.1) when exact f''(x) is passed as input via fder2, need to pass it on so that
                 a curve of exact dervative will be plotted in addition to FD curves
           (2.2) when filename is passed as input, need to pass it on so that the plot will 
                 be saved to a png file using a modification of this filename          

    '''
    x = np.array(x)  # this makes sure elementwise operations such as x+h and x-h are valid

    ffd = (f(x + 2 * h) - 2 * f(x + h) + f(x)) / h ** 2
    bfd = (f(x - 2 * h) - 2 * f(x - h) + f(x)) / h ** 2
    cfd = (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2

    FD_plot(x, h, ffd, cfd, bfd, fderivative=fder2, FD_order='2nd', filename=filename)
    return (ffd, cfd, bfd)


def FD_3rd_order(f, x, h=1e-4, fder3=None, filename=None):
    """compute 3rd order derivative of f(x) using FFD, CFD, BFD. 
       tasks: 
       (1) output f'''(x) in a tuple named (ffd, cfd, bfd), where ffd, cfd, bfd store the 
           f'''(x) obtained by FFD, CFD, and BFD;
       (2) call FD_plot() to do the plotting:
           (2.1) when exact f'''(x) is passed as input via fder2, need to pass it on so that
                 a curve of exact dervative will be plotted in addition to FD curves
           (2.2) when filename is passed as input, need to pass it on so that the plot will 
                 be saved to a png file using a modification of this filename          
    """
    x = np.array(x)  # this makes sure elementwise operations such as x+h and x-h are valid

    ffd = (-f(x) + 3 * f(x + h) - 3 * f(x + 2 * h) + f(x + 3 * h)) / h ** 3
    bfd = (-f(x - 3 * h) + 3 * f(x - 2 * h) - 3 * f(x - h) + f(x)) / h ** 3
    cfd = (-f(x - 2 * h) + 2 * f(x - h) - 2 * f(x + h) + f(x + 2 * h)) / (2 * h ** 3)

    FD_plot(x, h, ffd, cfd, bfd, fderivative=fder3, FD_order='3rd', filename=filename)
    return (ffd, cfd, bfd)


def FD_plot(x, h, ffd, cfd, bfd, fderivative=None, FD_order=None, filename=None):
    if fderivative != None:
        fd = fderivative(x);
        maxfd = max(abs(fd))
        plt.plot(x, fd, 'r-', lw=0.8, label="exact " + FD_order + " deriv.")
        print('\nmax err between FDs and the exact {} order derivative when h={}:'.format(FD_order, h))
        print('\t  max(fd - ffd)={}'.format(max(abs(ffd - fd))))
        print('\t  max(fd - bfd)={}'.format(max(abs(bfd - fd))))
        print('\t  max(fd - cfd)={}'.format(max(abs(cfd - fd))))
        print('\t  exact max(fd)={}'.format(maxfd))
        if h <= 0.01:
            tol = 0.01 * max(maxfd, 1)
            if (max(abs(cfd - fd)) > 2 * tol or max(abs(bfd - fd)) > 10 * tol or max(abs(ffd - fd)) > 10 * tol):
                print('{0} Error in your {1} order FD code, need debugging {0}\n\n\n'
                      .format(10 * '*', FD_order))

    plt.plot(x, ffd, 'c-', label="forward FD")
    plt.plot(x, bfd, 'b--', label="backward FD")
    plt.plot(x, cfd, 'g:', lw=2, label="central FD")
    plt.legend(loc='best')
    plt.title(FD_order + ' order derivative:  h={:.3f}'.format(h))
    if filename != None: plt.savefig(filename + '_' + FD_order + '_' + str(h) + '.png')
    plt.show()


def FD_tests():
    f1 = lambda x: np.cos(x ** 2 - x)
    f1der = lambda x: -np.sin(x ** 2 - x) * (2 * x - 1)
    f1der2 = lambda x: -np.cos(x ** 2 - x) * (2 * x - 1) ** 2 - np.sin(x ** 2 - x) * 2
    f1der3 = lambda x: np.sin(x ** 2 - x) * (2 * x - 1) ** 3 - np.cos(x ** 2 - x) * (2 * x - 1) * 4 - np.cos(
        x ** 2 - x) * 2 * (2 * x - 1)
    x = np.arange(0, np.pi, 0.01)
    for h in [0.5, 0.05, 0.001]:
        FD_1st_order(f1, x, h, fder=f1der, filename='function1')
        FD_2nd_order(f1, x, h, fder2=f1der2, filename='function1')
        FD_3rd_order(f1, x, h, fder3=f1der3, filename='function1')

    f2 = lambda x: x / (1 + np.exp(-x))
    f2der = lambda x: 1 / (1 + np.exp(-x)) + (x * np.exp(-x)) / (1 + np.exp(-x)) ** 2
    f2der2 = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2 * (2 - x + 2 * f2(x) * np.exp(-x))
    # analytic 3rd derivative is quite complicated, you can use the following one
    f2der3 = lambda x: np.exp(x) * (np.exp(2 * x) * (x - 3) - 4 * np.exp(x) * x + x + 3) / (np.exp(x) + 1) ** 4
    x = np.arange(-15, 15, 0.01)
    for h in [0.5, 0.2, 1e-2]:
        FD_1st_order(f2, x, h, fder=f2der, filename='swish')
        FD_2nd_order(f2, x, h, fder2=f2der2, filename='swish')
        FD_3rd_order(f2, x, h, fder3=f2der3, filename='swish')


# ===============================================================
if __name__ == '__main__':
    FD_tests()
