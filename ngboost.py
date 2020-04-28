import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor


class LogVarianceNorm:
    """Normal Distribution re-parameterize as follow.
        log_var = \ln \sigma^2
    """

    def score_and_fisher(self, t, m, log_var):
        """

        Args:
            t: target
            m: loc. the parameter to be estimated
            log_var: logarithm variance. the parameter to be estimated

        each gradient and fisher matrix as follow.

            gradients = [
                (m - t) / var,
                1 - (t - m) ** 2 / var
            ]

            fisher_matrix = [
                [1 / var, 0],
                [0, 1]
            ]

        where var is defined as var := exp(log_var)

        Returns:
            score and fisher matrix for each data.
            shape = (n_samples, 2), (n_samples, 2, 2)
        """
        var = np.exp(log_var)

        scores = np.zeros(shape=(len(t), 2))
        scores[:, 0] = (m - t) / var
        scores[:, 1] = 1 - (t - m) ** 2 / var

        fisher_matrix = np.zeros(shape=(len(t), 2, 2))
        fisher_matrix[:, 0, 0] = 1 / var
        fisher_matrix[:, 1, 1] = 1
        return scores, fisher_matrix

    def calculate_gradient(self, t, m, log_var, use_natural_gradient=True):
        grad, hess = self.score_and_fisher(t, m, log_var)
        if use_natural_gradient:
            return np.linalg.solve(hess, grad)
        return grad

    def score(self, t, m, log_var):
        loc, scale = self.to_loc_and_scale(m, log_var)
        return - norm.logpdf(t, loc=loc, scale=scale)

    def fit(self, y):
        loc, scale = norm.fit(y)
        log_var = 2 * np.log(scale)
        return loc, log_var

    def to_loc_and_scale(self, m, log_var):
        log_var = np.clip(log_var, -100, 100)
        var = np.exp(log_var)
        return m, var ** .5


class NormalNGBoost(object):
    def __init__(self, n_estimators=300, learning_rate=1e-2, subsample_bystep=.8, use_natural_grad=True):
        """
        NGBoost Regressor for Normal Distribution

        Args:
            n_estimators: total number of estimators
            learning_rate: boosting learning rate. (it's same as common gbdt library)
            subsample_bystep: subsample ratio in each boosting step.
            use_natural_grad: If true, use natural gradient. If False, use normal gradient (steepest descent)
        """
        self.distribution = LogVarianceNorm()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.use_natural_grad = use_natural_grad
        self.subsample_bystep = subsample_bystep

    def get_weak_learner(self):
        """
        get weak learner model class instance.
        [NOTE] too small `min_samples_leaf` and deep `max_depth` leads overfit for train
        so fix small depth and min_samples
        """
        return DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)

    def fit(self, X, y, **kwargs):
        """
        Fit NGBoost

        Args:
            X: input feature. shape = (n_samples, n_features) or (n_samples,)
            y: target. shape = (n_samples,)
            **kwargs:

        Returns:
            fitted NGBoost Model Instance (myself)
        """
        self.estimators_ = []  # store fitted models
        self.step_sizes_ = []  # store stepsize in each iteration

        N = len(X)
        X = X.reshape(N, -1)

        m, log_var = np.zeros(shape=(N,)), np.zeros(shape=(N,))

        # すべてのデータで平均と分散が共通と仮定して初期化
        self.init_m_, self.init_logvar_ = self.distribution.fit(y)
        m += self.init_m_
        log_var += self.init_logvar_

        for iteration in range(self.n_estimators):
            idx_use = np.random.permutation(int(N * self.subsample_bystep))

            # generate target gradient
            grads = self.distribution.calculate_gradient(y[idx_use], m[idx_use], log_var[idx_use],
                                                         use_natural_gradient=self.use_natural_grad)
            # mean estimator. fit mean gradient
            clf_mean = self.get_weak_learner()
            clf_mean.fit(X[idx_use], y=grads[:, 0])

            # log_var estimator. fit log_var gradient
            clf_var = self.get_weak_learner()
            clf_var.fit(X[idx_use], y=grads[:, 1])

            # generate batch direction (use in linear search)
            batch_directions = np.zeros(shape=(len(idx_use), 2))
            batch_directions[:, 0] = clf_mean.predict(X[idx_use])
            batch_directions[:, 1] = clf_var.predict(X[idx_use])
            self.estimators_.append(
                [clf_mean, clf_var]
            )

            # linear search and get best stepsize
            # [NOTE] サボってるので search 対象は .5, 1., 2. の決め打ち
            scores = []
            stepsize_choices = [.5, 1, 2]
            for stepsize in [.5, 1, 2]:
                d = batch_directions * stepsize
                score_i = self.distribution.score(y[idx_use],
                                                  m[idx_use] - d[:, 0],
                                                  log_var[idx_use] - d[:, 1]).mean()
                scores.append(score_i)

            best_idx = np.argmin(scores)
            best_stepsize = stepsize_choices[best_idx]

            stepsize_i = self.learning_rate * best_stepsize

            # Update distribution parameters using whole training data X
            whole_directions = np.array([
                clf_mean.predict(X),
                clf_var.predict(X)
            ]).T
            grad_parameters = whole_directions * stepsize_i
            m -= grad_parameters[:, 0]
            log_var -= grad_parameters[:, 1]

            self.step_sizes_.append(stepsize_i)

            if iteration % 50 == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean()
                print(
                    f'[iter: {iteration:03d}]\tscore: {scores[best_idx]:.4f}\tstepsize: {stepsize_i}\tnorm: {grad_norm:.4f}')
        return self

    def predict(self, X, n_estimators=None):
        mean = np.zeros_like(X) + self.init_m_
        log_var = np.zeros_like(X) + self.init_logvar_
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        for i, ((clf_mean, clf_var), stepsize) in enumerate(zip(self.estimators_, self.step_sizes_)):
            if n_estimators is not None and i >= n_estimators:
                break
            mean -= stepsize * clf_mean.predict(X)
            log_var -= stepsize * clf_var.predict(X)

        loc, scale = self.distribution.to_loc_and_scale(mean, log_var)
        return loc, scale


if __name__ == '__main__':
    def true_function(X):
        return np.sin(3 * X)


    def true_noise_scale(X):
        return np.abs(np.cos(X))


    n_samples = 200
    np.random.seed(71)
    X = np.random.uniform(-2, 1, n_samples)
    y = true_function(X) + np.random.normal(scale=true_noise_scale(X), size=n_samples)

    clf = NormalNGBoost(learning_rate=1e-2, n_estimators=500, use_natural_grad=False)
    clf.fit(X, y)

    xx = np.linspace(-2.3, 1.3, 300)
    output_dir = os.path.join('.', 'results')
    os.makedirs(output_dir, exist_ok=True)

    for i, n in enumerate(range(0, clf.n_estimators, 10)):
        loc, scale = clf.predict(xx, n_estimators=n)
        fig, axes = plt.subplots(figsize=(8, 8), nrows=2, sharex=True)
        ax = axes[0]
        ax.plot(xx, loc, '--', label='Predict (mean)', c='C1')
        ax.fill_between(xx, loc - scale, loc + scale, color='C1', label='One Sigma', alpha=.2)

        ax.scatter(X, y, label='Train', c='C0')
        ax.plot(xx, true_function(xx), c='C0')
        ax.set_ylim(-3, 3)
        ax.legend(loc=1)
        ax.set_title(f'n_estimator={n}')

        ax = axes[1]
        ax.set_ylabel('Uncertain Scale')
        ax.plot(xx, scale, label='Predict Scale', c='C0')
        ax.plot(xx, true_noise_scale(xx), '--', label='Ground Truth', c='C0')
        ax.set_ylim(-.1, np.max(true_noise_scale(xx)) + .1)
        ax.legend(loc=1)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{i:03d}.png'), dpi=120)
        plt.close(fig)
