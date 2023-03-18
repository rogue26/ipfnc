import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class IPF:
    def __init__(self, converged: float = 1E-6, max_iter: int = 100, marginals=None, n_solutions=1000):

        self.converged = converged
        self.max_iter = max_iter
        self.marginals = marginals
        self.n_solutions = n_solutions

        self.solutions = []

    def _optimize_ipf(self):
        for i in range(self.n_solutions):
            random_inner = np.random.random((3, 3, 3))
            self.solutions.append(self._solve_ipf(random_inner))

    def _solve_ipf(self, inner):
        i = 0
        max_overall_delta = np.inf
        # todo: https://stackoverflow.com/questions/59092561/how-to-use-iterator-in-while-loop-statement-in-python
        while max_overall_delta >= self.converged and i <= self.max_iter:
            for axis in range(inner.ndim):
                b_sum = np.broadcast_to(np.expand_dims(np.sum(inner, axis), axis=axis), inner.shape)
                b_marginal = np.broadcast_to(np.expand_dims(self.marginals[axis], axis=axis), inner.shape)

                inner = inner * b_marginal / b_sum

            deltas = [self.marginals[_] - np.sum(inner, _) for _ in range(inner.ndim)]
            max_deltas = [max(_.min(), _.max(), key=abs) for _ in deltas]
            max_overall_delta = max(max_deltas)

            i += 1

        return inner

    def _find_closest_solution(self, ref):
        sum_squared_residuals = [np.sum((solution - ref) ** 2) for solution in self.solutions]
        minimum_loc = np.where(sum_squared_residuals == min(sum_squared_residuals))[0][0]
        return self.solutions[minimum_loc],  minimum_loc

    def _cluster(self):
        """
        perform hierarchical clustering on the solutions to identify any patterns
        for use in optimize mode.
        """

    def _plot(self):
        """
        for use in optimize mode
        """
        fig, axes = plt.subplots(9, 3, sharex=True, sharey=True)

        flat_axes = axes.ravel()

        solution_of_maxima = np.empty((3, 3, 3))
        for (index, values), ax in zip(np.ndenumerate(np.ones((3, 3, 3))), flat_axes):
            # plot KDE
            sns.kdeplot(data=[solution[index] for solution in self.solutions], label=index, ax=ax)

            # get the x location of the maximum value
            data = ax.lines[0].get_xydata()
            maximum = data[np.where(data[:, 1] == max(data[:, 1]))]
            ax.axvline(maximum[0][0], color='red')
            solution_of_maxima[index] = maximum[0][0]

            ax.set(ylabel=None)
            ax.set_yticklabels([])
            ax.set_yticks([])

        closest, location = self._find_closest_solution(solution_of_maxima)

        for (index, values), ax in zip(np.ndenumerate(np.ones((3, 3, 3))), flat_axes):
            ax.axvline(closest[index], color='green')
