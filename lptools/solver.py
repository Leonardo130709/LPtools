import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from functools import partial
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
from scipy.stats import norm


class Solver:
    def __init__(self, data=None, dist=norm()):
        self.alpha = None
        self.sigma = None
        if data is not None:
            self.fit(data)
        self.N = 100
        self.dist = dist

    def fit(self, values, from_df=True):
        if from_df:
            values = self.preproc(values)
        result = OLS(values, np.arange(len(values)))
        result = result.fit()
        mu, var = result.params[0], result.scale
        self.alpha = mu + var / 2
        self.sigma = np.sqrt(var)

    def generator(self, t):
        """
        dp = \alpha p dt + \sigma p d W_t
        """
        paths = []
        for _ in range(self.N):
            path = [1.]
            for _ in range(t):
                p = path[-1] * (1 + self.alpha + self.sigma * self.dist.rvs())
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

    def first_type_constraint(self, x, t, q):
        f = partial(self.IL, pl=x[0], pu=x[1])
        samples = self.estimate_rv(f, t)
        values = samples[:, -1]
        return np.quantile(values, q)

    def second_type_constraint(self, x, t, q):
        f = partial(self.IL, pl=x[0], pu=x[1])
        samples = self.estimate_rv(f, t)
        values = samples.max(axis=1)
        return np.quantile(values, q)

    def objective(self, x, t):
        samples = self.estimate_rv(lambda p: p, t)
        sp = np.sqrt(samples)
        var = (sp / np.sqrt(x[1]) + np.sqrt(x[0]) / sp) / 2
        values = 1 / (1 - var) / sp
        values = np.where((x[0] < samples) & (samples < x[1]), values, 0)
        # values = sp / (np.sqrt(x[1]) + 1e-10) + np.sqrt(x[0]) / sp
        values = values.mean(axis=0).sum()
        return -values

    def solve(self, t, delta1, q1, delta2, q2, verbose=True, maxiter=30, popsize=50):
        f1 = partial(self.first_type_constraint, t=t, q=q1)
        f2 = partial(self.second_type_constraint, t=t, q=q2)
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
        p = df.token0Price
        p = np.log(p).diff().dropna().values
        return p

    @staticmethod
    def IL(p, pl, pu):
        return 1 - 2 * np.sqrt(np.clip(p, pl, pu)) / (1 + p)
