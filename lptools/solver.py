import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from functools import partial
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution


class Solver:
    def __init__(self, data=None):
        self.alpha = None
        self.sigma = None
        if data is not None:
            self.fit(data)

    def fit(self, values, from_df=True):
        if from_df:
            values = self.preproc(values)
        result = OLS(values, np.arange(len(values)))
        result = result.fit()
        mu, sigma = result.params[0], result.scale
        self.alpha = mu + sigma ** 2 / 2
        self.sigma = sigma

    def generator(self, t, N=150):
        """
        dp = \alpha p dt + \sigma p d W_t
        """
        paths = []
        for _ in range(N):
            path = [1.]
            for _ in range(t):
                p = path[-1] * (1 + self.alpha + self.sigma * np.random.randn())
                path.append(p)
            paths.append(path)
        return np.array(paths)

    def estimate_rv(self, rv, t):
        paths = self.generator(t)
        rv = np.vectorize(rv)
        return rv(paths)

    @staticmethod
    def plot_paths(paths):
        for path in paths:
            plt.plot(path)
        plt.show()

    def first_type_constrain(self, x, t, q):
        f = partial(self.IL, pl=x[0], pu=x[1])
        samples = self.estimate_rv(f, t)
        values = samples[:, -1]
        return np.quantile(values, q)

    def second_type_constrain(self, x, t, q):
        f = partial(self.IL, pl=x[0], pu=x[1])
        samples = self.estimate_rv(f, t)
        values = samples.max(axis=1)
        return np.quantile(values, q)

    def objective(self, x, t):
        samples = self.estimate_rv(lambda p: p, t)
        values = np.where((x[0] < samples) & (samples < x[1]), 1, 0)
        values = - values.mean(0).sum()
        # may not give the right direction for the iterative method cause of the discrete reward
        # use instead:
        # sp = np.sqrt(samples)
        # values = sp / (np.sqrt(x[1]) + 1e-10) + np.sqrt(x[0]) / sp
        # values = values.mean(axis=1).sum()
        return values

    def solve(self, t, delta1, q1, delta2, q2, verbose=True, maxiter=30, popsize=50):
        f1 = partial(self.first_type_constrain, t=t, q=q1)
        f2 = partial(self.second_type_constrain, t=t, q=q2)
        c1 = NonlinearConstraint(f1, -1, delta1)
        c2 = NonlinearConstraint(f2, -1, delta2)
        c3 = NonlinearConstraint(lambda x: x[1] - x[0], 0, np.inf)
        f = partial(self.objective, t=t)
        return differential_evolution(
            f, 2 * ((0., 2.),),
            constraints=[c1, c2, c3],
            maxiter=maxiter, popsize=popsize,
            polish=False, disp=verbose,
            workers=-1, updating='deferred'
        )

    @staticmethod
    def preproc(df):
        p = df.token0Price.values
        p = p / p[0]
        p = np.log(p)
        return p

    @staticmethod
    def IL(p, pl, pu):
        return 1 - 2 * np.sqrt(np.clip(p, pl, pu)) / (1 + p)