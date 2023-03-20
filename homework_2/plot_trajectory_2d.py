import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import methods
import oracles

COLOR_RED = np.linspace(240, 166, 256) / 255.
COLOR_GREEN = np.linspace(244, 188, 256) / 255.
COLOR_BLUE = np.linspace(246, 203, 256) / 255.


def plot_levels(func, xrange=None, yrange=None, levels=None):
    """
    Plotting the contour lines of the function.

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> plot_levels(oracle.func)
    """
    if xrange is None:
        xrange = [-6, 6]
    if yrange is None:
        yrange = [-5, 5]
    if levels is None:
        levels = [0, 0.25, 1, 4, 9, 16, 25]
        
    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    colors = np.vstack([COLOR_RED, COLOR_GREEN, COLOR_BLUE]).T
    my_cmap = ListedColormap(colors)
    
    _ = plt.contourf(X, Y, Z, levels=levels, cmap=my_cmap)
    CS = plt.contour(X, Y, Z, levels=levels, colors='#ABBECC')
    plt.clabel(CS, inline=1, fontsize=8, colors='#AAAEBB') 
    plt.grid()              

        
def plot_trajectory(func, history, fit_axis=False, label=None, color='C1'):
    """
    Plotting the trajectory of a method. 
    Use after plot_levels(...).

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> [x_star, msg, history] = optimization.gradient_descent(oracle, np.array([3.0, 1.5], trace=True)
    >> plot_levels(oracle.func)
    >> plot_trajectory(oracle.func, history['x'])
    """
    x_values, y_values = zip(*history)
    plt.plot(x_values, y_values, '-o', linewidth=1.0, ms=5.0,
             alpha=1.0, c=color, label=label)
    
    # Tries to adapt axis-ranges for the trajectory:
    if fit_axis:
        xmax, ymax = np.max(x_values), np.max(y_values)
        COEF = 1.5
        xrange = [-xmax * COEF, xmax * COEF]
        yrange = [-ymax * COEF, ymax * COEF]
        plt.xlim(xrange)
        plt.ylim(yrange)


def a_calc(k0, k, phi=np.pi/3):
    s = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
        ])
    A = np.array([
        [k0, 0],
        [0, k]
        ])
    return np.dot(s, np.dot(A, s.T))


def save_plots(X, X0, b, xrange=None, yrange=None, levels=None, img=None):
    oracle = oracles.QuadraticOracle(X, b)

    gd_const = methods.GradientDescent(
        oracle, X0, tolerance=1e-10, 
        line_search_options={'method': 'Constant','c': 0.001}
        )
    gd_armijo = methods.GradientDescent(
        oracle, X0, tolerance=1e-10,
        line_search_options={
            'method': 'Armijo'}
            )
    gd_wolfe = methods.GradientDescent(
        oracle, X0, tolerance=1e-10
        )

    _, hist_const = gd_const.run()
    _, hist_armijo = gd_armijo.run()
    _, hist_wolfe = gd_wolfe.run()

    plot_levels(oracle.func, xrange, yrange, levels)
    # plt.savefig(f'figures/{img}.png')
    plot_trajectory(oracle.func, hist_const['x'], color='blue')
    plot_trajectory(oracle.func, hist_armijo['x'], color='orange')
    plot_trajectory(oracle.func, hist_wolfe['x'], color='red')
    plt.savefig(f'figures/{img}.png')


if __name__ == '__main__':
    X1 = a_calc(1, 50)
    print(X1)
    b1 = np.array([0, 0])
    X2 = a_calc(1, 5)
    print(X2)
    b2 = np.array([0, 0])
    X01 = np.array([2, 4])
    X02 = np.array([7, 8])

    # X3 = a_calc(50)
    # b3 = np.array([1, 2, 3])
    oracle1 = oracles.QuadraticOracle(X1, b1)
    oracle2 = oracles.QuadraticOracle(X2, b2)
    # oracle3 = oracles.QuadraticOracle(X3, b3)

    # делаем спуск разными методами
    save_plots(X1, X01, b1, img='fig1')
    save_plots(X2, X02, b2, img='fig2')

    X = np.array([[0.5, 0],
                  [0, -0.5]])
    y = np.array([0, 0])
    w_init = np.array([3, -3])
    xrange = np.array([-50, 50])
    yrange= np.array([-5, 5])
    levels = [0, 1, 2, 3, 4]
    save_plots(X, y, w_init, xrange=xrange, yrange=yrange, levels=levels, img='fig_3')

    X = np.array([[0.5, 0],
                  [0, -0.5]])
    y = np.array([0, 0])
    w_init = np.array([1, 0])
    xrange = np.array([-5, 5])
    yrange= np.array([-5, 5])
    levels = [0, 1, 2, 3, 4, 16, 64, 128]
    save_plots(X, y, w_init, xrange=xrange, yrange=yrange, levels=levels, img='fig_4')