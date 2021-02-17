import numpy as np

import matplotlib.pyplot as plt

def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()    
        
def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x)+1)
    
    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)
    
    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')

def plot_convergence(X_sample, Y_sample, n_init=1):
    fig = make_subplots(rows=1, cols=2)

    [x] = X_sample[n_init:].ravel()
    [y] = Y_sample[n_init:].ravel()
    r = range(1, len(x)+1)
    print(x, y)
    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    print(x_neighbor_dist)
    y_max_watermark = np.maximum.accumulate(y)
    print(y_max_watermark)
    


from plotly.subplots import make_subplots


def create_plots(gpr, X, Y, X_sample, Y_sample, EI, X_next):

    fig = make_subplots(rows=1, cols=2)

    mu, std = gpr.predict(X, return_std=True)

    upper = mu.ravel() + 1.96 * std
    lower = mu.ravel() - 1.96 * std

    x = X.ravel().tolist()
    fx = f(X).ravel().tolist()
    y = Y.ravel().tolist()
    ei = EI.ravel().tolist()
    [xnext] = X_next.tolist()


    # plot_approximation
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
        go.Scatter(x=x, y=upper,
                mode='lines',
                line=dict(
                    color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=x, y=lower,
                mode='lines',
                line=dict(
                    color='blue'),
                fill='tonexty'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=[xnext, xnext], y=[-5,5],
                mode='lines',
                line=dict(
                    color='black',
                    dash='dash'
                )),
        row=1, col=1
    )


    # plot_acquisition
    fig.add_trace(
        go.Scatter(x=x, y=ei,
                mode='lines',
                name='Acquisition function',
                line=dict(
                    color='red',
                    width=4)),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=[xnext, xnext], y=[-5,5],
                mode='lines',
                line=dict(
                    color='black',
                    dash='dash'
                )),
        row=1, col=2
    )


    return fig









    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_font=dict(
            color='white'
        ),
        xaxis=dict(
            color='white'
        ),
        yaxis=dict(
            color='white'
        )
    )

