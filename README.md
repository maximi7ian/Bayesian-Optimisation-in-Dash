# BayesianOptimisation
Many optimisation problems in machine learning require the optimisation of an objective function, f(x), that does not have an analytical expression and has unknown derivatives. This means evaluation of the function is restricted to sampling at a point x and getting a possibly noisy response.

If f is cheap to evaluate, we could sample at many points e.g. via grid search, random search or numeric gradient estimation. However, if function evaluation is expensive e.g. tuning hyperparameters of a neural network, drilling for oil or evaluating the effectiveness of a drug then it is important to minimise the number of samples drawn from the function f.

This is where Bayesian optimisation comes in handy! Bayesian optimization incorporates prior belief about f and updates the prior with samples drawn from f to get a posterior that better approximates f.

In Bayesian optimisation, a surrogate function is used to approximate the objective function (f) and a function named the acquisition function directs sampling of x to areas of most improvement. 

A popular surrogate function for Bayesian optimization are Gaussian processes(GPs). GPs define priors over functions and can be used to incorporate prior beliefs about the objective function (smoothness, …). It is also beneficial as it is computationally cheap to evaluate.

Acquisition functions trade off exploitation and exploration. Exploitation means sampling where the surrogate model predicts a high objective returns and exploration means sampling at locations where the prediction uncertainty is high. The goal is to maximize the acquisition function to determine the next sampling point. There are numerous popular acquisition functions including; maximum probability of improvement (MPI), expected improvement (EI) and upper confidence bound (UCB).

Bayesian optimisation also uses a kernel function which controls the shape of the function at specific points based on distance measures between actual data observations.

---
<!-- https://www.markdownguide.org/cheat-sheet/ -->
<!-- http://krasserm.github.io/2018/03/21/bayesian-optimization/ -->
<!-- http://philipperemy.github.io/visualization/ -->
<!-- https://machinelearningmastery.com/what-is-bayesian-optimization/ -->
<!-- https://distill.pub/2020/bayesian-optimization/ -->

<!-- https://scikit-learn.org/stable/modules/gaussian_process.html#matern-kernel -->