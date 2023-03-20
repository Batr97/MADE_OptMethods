import numpy as np
import matplotlib.pyplot as plt
import scipy
import oracles
import methods


def random_func(k, n, b_min, b_max):
    A = np.random.uniform(low=1, high=k, size=(n-2))
    A = np.concatenate([[1, k], A])
    np.random.shuffle(A)
    # A = scipy.sparse.diags(A)
    A = np.diag(A)
    b = np.random.uniform(low=b_min, high=b_max, size=n) * k
    return A, b


def curve_generation(k, n, b_min, b_max, color, method):
    iterations = 5
    # dims = {}
    if method == 'Armijo':
        options = {'method': 'Armijo'}
    elif method == 'constant':
        options = {'method': 'Constant','c': 0.01}
    for i, dim in enumerate(n):
        print('dim', dim)
        x0 = np.zeros(dim)
        for iter in range(iterations):
            iter_lst = []
            for cond in k:
                A, b = random_func(cond, dim, b_min, b_max)
                oracle = oracles.QuadraticOracle(A, b)
                # gd_wolfe = methods.GradientDescent(oracle, x0, tolerance=1e-10)
                gd_armijo = methods.GradientDescent(oracle, x0, line_search_options=options)
                _, hist = gd_armijo.run(max_iter=2500)
                iter_lst.append(len(hist['func']))
            if iter == 0:
                plt.plot(k, iter_lst, label=f'dim={dim}', color=color[i])
            else:
                plt.plot(k, iter_lst, color=color[i])
    plt.xlabel("Conditional number")
    plt.ylabel("Iterations")
    plt.legend()
    plt.title(f'{method} k vs iterations')
    plt.savefig(f'figures/{method}_iter_vs_k_n.png') 



if __name__ == '__main__':
    # k = np.arange(10, 100, 10)
    k = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    n = np.array([10, 100, 1000, 3000])
    color = ["red", "black", "blue", "orange"]
    b_min, b_max = -7, 7
    curve_generation(k, n, b_min, b_max, color, method='Armijo')
    # curve_generation(k, n, b_min, b_max, color, method='constant')