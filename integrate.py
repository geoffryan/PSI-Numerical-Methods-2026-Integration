import numpy as np
import matplotlib.pyplot as plt


def riemann(a, b, n, f, *f_args, **f_kwargs):
    """
    Computes first-order approximation of then integral of f from a to b

    a:  float, lower integration limit
    b:  float, upper integration limit
    n:  int, number of sub-intervals to use
    f:  function  f(x, ...) which takes x (an array of floats [x0, x1, ...])
        and returns an array containing the function evaluated on each x:
        [f(x0), f(x1), ...]
    *f_args:    extra positional arguments for f
    **f_kwargs: extra keyword arguments for f

    The error should scale like 1 / n
    """

    # Make the array of sample points.
    # [a    a+h                b-h    b]
    # [x[0] x[1] x[2] x[3] ... x[n-1]  ]
    # [  0   | 1  | 2  |   ...  | n-1  ]
    xk = np.linspace(a, b, n, endpoint=False)

    # Evaluate f on x
    fk = f(xk, *f_args, **f_kwargs)

    # Compute the step size
    h = xk[1] - xk[0]

    # Return the Riemann approximation to the integral
    return h * fk.sum()


def trap(a, b, n, f, *f_args, **f_kwargs):
    """
    Computes second-order approximation of the integral of f from a to b

    a:  float, lower integration limit
    b:  float, upper integration limit
    n:  int, number of sub-intervals to use, will use n+1 function evals
    f:  function  f(x, ...) which takes x (an array of floats [x0, x1, ...])
        and returns an array containing the function evaluated on each x:
        [f(x0), f(x1), ...]
    *f_args:    extra positional arguments for f
    **f_kwargs: extra keyword arguments for f

    The error should scale like 1 / n^2
    """

    # Make the array of sample points.
    # [a    a+h  ...           b-h    b   ]
    # [x[0] x[1] x[2] x[3] ... x[n-1] x[n]]
    # [  0   | 1  | 2  |   ...  | n-1     ]
    xk = np.linspace(a, b, n+1)

    # Evaluate f on x
    fk = f(xk, *f_args, **f_kwargs)

    # Compute the step size (since we used linspace, there is a constant
    # separation between values in xk
    h = xk[1] - xk[0]

    # Return the Trapezoid Rule integration approximation
    return 0.5 * h * (fk[0] + fk[n] + 2 * fk[1:-1].sum())


def simp(a, b, n, f, *f_args, **f_kwargs):
    """
    Computes fourth-order approximation of the integral of f from a to b

    a:  float, lower integration limit
    b:  float, upper integration limit
    n:  int, number of samples to use, will use 2*n + 1 function evals
    f:  function  f(x, ...) which takes x (an array of floats [x0, x1, ...])
        and returns an array containing the function evaluated on each x:
        [f(x0), f(x1), ...]
    *f_args:    extra positional arguments for f
    **f_kwargs: extra keyword arguments for f

    The error should scale like 1 / n^4
    """

    # Make the array of sample points.
    # [a     a+h   a+2h  a+3h  a+4h ... b-2h    b-h     b    ]
    # [x[0]  x[1]  x[2]  x[3]  x[4] ... x[2n-2] x[2n-1] x[2n]]
    # [1     4     1|1   4     1 |  ...  |1     4       1    ]
    # [ parabola 0  | parabola 1 |  ...  |   parabola n      ]
    xk = np.linspace(a, b, 2*n+1)

    # Evaluate f
    fk = f(xk, *f_args, **f_kwargs)

    # Step size
    h = xk[1] - xk[0]

    # Simpson's rule
    return h * (fk[0] + fk[-1] + 4*fk[1:-1:2].sum() + 2*fk[2:-2:2].sum()) / 3.0


def f(x):

    return np.exp(x)


if __name__ == "__main__":

    # Correct integral result
    answer = np.e - 1

    # Number of sub-intervals to use
    N_sub = 2 ** np.arange(1, 13)

    # For each scheme, compute the total number of function evaluations (N*)
    # and compute the integral for each N_sub
    N1 = N_sub
    I1 = np.array([riemann(0, 1, n, f) for n in N_sub])

    N2 = N_sub + 1
    I2 = np.array([trap(0, 1, n, f) for n in N_sub])

    N4 = 2*N_sub + 1
    I4 = np.array([simp(0, 1, n, f) for n in N_sub])

    # Print the answers just because
    print(I1)
    print(I2)
    print(I4)

    # Make a convergence plot
    fig, ax = plt.subplots(1, 1)

    # Plot the absolute value of the error for each scheme so we can log-scale
    ax.plot(N1, np.fabs(I1-answer), ls='-', marker='.', label='riemann')
    ax.plot(N2, np.fabs(I2-answer), ls='-', marker='.', label='trapezoid')
    ax.plot(N4, np.fabs(I4-answer), ls='-', marker='.', label='simpsons')

    # Plot expected convergence rates for each method to help guide the eye
    ax.plot(N1, np.power(N1, -1.0), lw=2, alpha=0.5, color='grey', ls='--')
    ax.plot(N1, np.power(N1, -2.0), lw=2, alpha=0.5, color='grey', ls='--')
    ax.plot(N1, np.power(N1, -4.0), lw=2, alpha=0.5, color='grey', ls='--')

    # Add a legend to ID the schemes
    ax.legend()

    # Label & scale the axes
    ax.set(xscale='log', xlabel=r'Number of function evaluations $N$',
           yscale='log', ylabel=r'Error |approx - true|')

    # Save the plot
    figname = "convergence.png"
    print("Saving", figname)
    fig.savefig(figname)

    # Done!
    plt.show()
