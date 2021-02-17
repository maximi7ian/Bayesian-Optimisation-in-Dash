# Imports
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Bayesian Opt funcitons and Plotting functions

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr).flatten()
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)


def create_plots(f, gpr, X, Y, X_sample, Y_sample, EI, X_next, bounds):
    '''
    Creates two plots:

    Left:
    True function and surrogate function (with CI)
    X samples and line indicating location of next sample

    Right:
    Aquisition function
    Location of next sample

    '''
    fig = make_subplots(rows=1, cols=2)
    
    mu, std = gpr.predict(X, return_std=True)

    upper = mu.ravel() + 1.9 * std.ravel()
    lower = mu.ravel() - 1.9 * std.ravel()

    x = X.ravel().tolist()
    fx = f(X).ravel().tolist()
    y = Y.ravel().tolist()
    x_sample = X_sample.ravel().tolist()
    y_sample = Y_sample.ravel().tolist()

    ei = EI.ravel().tolist()
    [[xnext]] = X_next.tolist()
    
    [bounds] = bounds.tolist()
    maxy = max(Y) + max(Y)*0.3
    [maxy] = maxy.tolist()
    miny = min(Y) - min(Y)*0.3
    [miny] = miny.tolist()

    # plot_approximation
    fig.add_trace(
        go.Scatter(x=x, y=upper,
                name='Confidence Interval',
                mode='lines',
                line=dict(
                    color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=x, y=lower,
                name='Confidence Interval',
                mode='lines',
                line=dict(
                    color='blue'),
                fill='tonexty',
                fillcolor='rgba(0,0,240,0.1)',
                showlegend=False),
        row=1, col=1
    )


    fig.add_trace(
        go.Scatter(x=x, y=y,
                mode='lines',
                name='True Function',
                line=dict(
                    color='white',
                    width=2,
                    dash='dash')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=x, y=mu.ravel().tolist(),
                mode='lines',
                name='Surrogate function',
                line=dict(
                    color='red',
                    width=2)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=x_sample, y=y_sample,
                mode='markers',
                marker_symbol = 'x',
                name='Next Sample',
                marker=dict(
                    color='black',
                    size=9),
                showlegend=False),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=[xnext, xnext], y=[miny, maxy],
                mode='lines',
                name='Next Sample',
                line=dict(
                    color='black',
                    dash='dash'),
                showlegend=False),
        row=1, col=1
    )


    # plot_acquisition
    fig.add_trace(
        go.Scatter(x=x, y=ei,
                mode='lines',
                name='Acquisition function',
                line=dict(
                    color='fuchsia',
                    width=2)),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=[xnext, xnext], y=[0, max(ei)+max(ei)*0.1],
                mode='lines',
                name='Next Sample',
                line=dict(
                    color='black',
                    dash='dash',
                    width=3
                )),
        row=1, col=2
    )


    fig.update_xaxes(range=[bounds[0]+bounds[0]*0.2, bounds[1]+bounds[1]*0.2], color = 'white', row=1, col=1)
    fig.update_xaxes(range=[bounds[0]+bounds[0]*0.2, bounds[1]+bounds[1]*0.2], color = 'white', row=1, col=2)
    fig.update_yaxes(range=[miny, maxy], color = 'white', row=1, col=1)
    fig.update_yaxes(range=[0, max(ei)+max(ei)*0.1], color = 'white', row=1, col=2)


    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_font=dict(
            color='white'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="right",
            x=0.8
        ),
        margin=dict(l=20, r=20, t=20, b=20)        
    )


    return fig



def plot_convergence(X_sample, Y_sample, n_init=1):
    '''
    Creates two plots:

    Left:
    Plot of distance between consecutive xs

    Right:
    Plot of best sampled y value so far


    '''
    fig = make_subplots(rows=1, cols=2)

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    
    xn= range(1, len(x)+1)
    yn= range(1, len(y)+1)
    
    x_neighbor_dist = [float(np.abs(a-b)) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    xn= list(range(1, len(x_neighbor_dist)+1))
    yn= list(range(1, len(y_max_watermark)+1))

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Scatter(x=xn, y=x_neighbor_dist,
                mode='markers+lines',
                name='Distance between consecutive x\'s',
                line=dict(
                    color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=yn, y=y_max_watermark,
                mode='markers+lines',
                name='Value of best selected sample',
                line=dict(
                    color='red')),
        row=1, col=2
    )

    fig.update_xaxes(title = dict(text = 'Iteration'), color = 'white', row=1, col=1)
    fig.update_xaxes(title = dict(text = 'Iteration'), color = 'white', row=1, col=2)
    fig.update_yaxes(title = dict(text = 'Distance'), color = 'white', row=1, col=1)
    fig.update_yaxes(title = dict(text = 'Best Y'), color = 'white', row=1, col=2)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_font=dict(
            color='white'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="right",
            x=0.7
        ),
        margin=dict(l=20, r=20, t=20, b=20)        
    )


    return fig

