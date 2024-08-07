import numpy as np
import matplotlib.pyplot as plt
import numpy.random


def save_plot(X, Y, min_x, min_y, plot_title, rows=2, cols=2, n_plots=4):
    fig, axs = plt.subplots(rows, cols, figsize=(10, 5))
    axs = axs.reshape(rows * cols)
    for i in range(n_plots):
        axs[i].scatter(X[:, i], Y, marker='.', label='trials', color='#1f77b440')
        axs[i].scatter(min_x[i], min_y, label='minimum', color='#ff7f0e')

        axs[i].annotate(f'({min_x[i]:.04f}, {min_y:.04f})', (min_x[i], min_y))
        axs[i].set_xlabel(f'$x_{i + 1}$')
        axs[i].set_ylabel(f'f(X)')
        axs[i].legend()
    fig.tight_layout()
    fig.title = plot_title
    plt.show()
    fig.savefig(plot_title.replace(' ', '_') + '.png')


def print_problem_solution(problem_title, min_x, min_y, gs):
    print(problem_title)
    print(f'min_x = {min_x}')
    print(f'min_y = {min_y}')
    for i, g in enumerate(gs, 1):
        print(f'g{i}(X) = {g(min_x):.04f}')
    print()


def problem1(N):
    rng = numpy.random.RandomState(828)

    def objective(X):
        x1, x2, x3, x4 = X

        r = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 - 1300 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3
        return r

    def g1(X):
        x1, x2, x3, x4 = X

        return -x1 + 0.0193 * x3

    def g2(X):
        x1, x2, x3, x4 = X

        return -x2 + 0.00954 * x3

    def g3(X):
        x1, x2, x3, x4 = X

        return -np.pi * x3 ** 2 * x4 - 1.3333 * np.pi * x3 ** 3 + 1296000

    def g4(X):
        x1, x2, x3, x4 = X

        return x4 - 240

    X = np.zeros([N, 4])
    for i in range(N):
        while True:
            x = np.array([
                rng.uniform(0.0625, 2),
                rng.uniform(0.0625, 2),
                rng.uniform(10, 100),
                rng.uniform(0, 200),
            ])
            if all([g1(x) <= 0, g2(x) <= 0, g3(x) <= 0, g4(x) <= 0]):
                X[i] = x
                break

    Y = np.apply_along_axis(objective, 1, X)

    min_index = np.argmin(Y)
    min_x = X[min_index]
    min_y = Y[min_index]

    save_plot(X, Y, min_x, min_y, "problem 1 output")
    print_problem_solution("Problem 1:", min_x, min_y, [g1, g2, g3, g4])


def problem2(N, i=0):
    rng = numpy.random.RandomState(828)

    P = 6000
    L = 14
    G = 12 * 10 ** 6
    E = 30 * 10 ** 6
    max_shear_stress = 13600
    max_normal_stress = 30000
    max_deflection = 0.25

    def get_normal_stress(X):
        x1, x2, x3, x4 = X

        normal_stress = (4 * P * L) / (x4 * x3 ** 2)
        return normal_stress

    def get_shear_stress(X):
        x1, x2, x3, x4 = X

        M = P * (L + x2 / 2)
        R = np.sqrt(x2 ** 2 / 4 + ((x1 + x3) / 2) ** 2)
        J = 2 * (x1 * x2 / np.sqrt(2) * (x2 ** 2 / 12 + ((x1 + x3) / 2) ** 2))

        t1 = P / (np.sqrt(2) * x1 * x2)
        t2 = M * R / J

        shear_stress = np.sqrt(t1 ** 2 + 2 * t1 * t2 * x2 / (2 * R) + t2 ** 2)

        return shear_stress

    def get_deflection(X):
        x1, x2, x3, x4 = X

        deflection = (4 * P * L) / (E * x3 ** 3 * x4)
        return deflection

    def get_buckling_load(X):
        x1, x2, x3, x4 = X

        buckling_load = 4.013 * np.sqrt(E * G * (x3 ** 2 * x4 ** 6 / 36)) * (
                1 - (x3 / (2 * L)) * np.sqrt(E / (4 * G))) / L ** 2
        return buckling_load

    def objective(X):
        x1, x2, x3, x4 = X

        cost = 1.1047 * x1 ** 2 * x2 + 0.04811 * x3 * x4 * (14 + x2)
        return cost

    def g1(X):
        return get_shear_stress(X) - max_shear_stress

    def g2(X):
        return get_normal_stress(X) - max_normal_stress

    def g3(X):
        x1, x2, x3, x4 = X

        return x1 - x4

    def g4(X):
        x1, x2, x3, x4 = X

        return 0.10471 * x1 ** 2 + 0.04811 * x3 * x4 * (14 + x2) - 5

    def g5(X):
        x1, x2, x3, x4 = X

        return 0.125 - x1

    def g6(X):
        return get_deflection(X) - max_deflection

    def g7(X):
        return P - get_buckling_load(X)

    X = np.zeros([N, 4])
    for i in range(N):
        while True:
            x = np.array([
                rng.uniform(0.1, 2),
                rng.uniform(0.1, 10),
                rng.uniform(0.1, 10),
                rng.uniform(0.1, 2),
            ])
            if all([g1(x) <= 0, g2(x) <= 0, g3(x) <= 0, g4(x) <= 0, g5(x) <= 0, g6(x) <= 0, g7(x) <= 0]):
                X[i] = x
                break

    Y = np.apply_along_axis(objective, 1, X)

    min_index = np.argmin(Y)
    min_x = X[min_index]
    min_y = Y[min_index]

    save_plot(X, Y, min_x, min_y, "problem 2 output")
    print_problem_solution("Problem 2:", min_x, min_y, [g1, g2, g3, g4, g5, g6, g7])


def problem3(N):
    rng = numpy.random.RandomState(828)

    def objective(X):
        x1, x2, x3 = X

        volume = (x3 + 2) * x2 * x1 ** 2
        return volume

    def g1(X):
        x1, x2, x3 = X

        return 1 - ((x2 ** 3 * x3) / (71785 * x1 ** 4))

    def g2(X):
        x1, x2, x3 = X

        return (4 * x2 ** 2 - x1 * x2) / (12566 * (x2 * x1 ** 3 - x1 ** 4)) + 1 / (5108 * x1 ** 2) - 1

    def g3(X):
        x1, x2, x3 = X

        return 1 - 140.45 * x1 / (x2 ** 2 * x3)

    def g4(X):
        x1, x2, x3 = X

        return ((x1 + x2) / 1.5) - 1

    X = np.zeros([N, 3])
    for i in range(N):
        while True:
            x = np.array([
                rng.uniform(0.05, 2),
                rng.uniform(0.25, 1.3),
                rng.uniform(2, 15),
            ])

            if all([g1(x) <= 0, g2(x) <= 0, g3(x) <= 0, g4(x) <= 0]):
                X[i] = x
                break

    Y = np.apply_along_axis(objective, 1, X)

    min_index = np.argmin(Y)
    min_x = X[min_index]
    min_y = Y[min_index]

    save_plot(X, Y, min_x, min_y, "problem 3 output", n_plots=3)
    print_problem_solution("Problem 3:", min_x, min_y, [g1, g2, g3, g4])


if __name__ == '__main__':
    problem1(1_000)
    problem2(1_000)
    problem3(1_000)

