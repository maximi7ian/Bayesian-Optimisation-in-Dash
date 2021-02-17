import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# np.exp(-(X - 2)**2) + np.exp(-(X - 6)**2/10) + 1/ (X**2 + 1)
# -2, 10, 0.025, 100

# X**2 * np.sin(5 * np.pi * X)**6.0
# 0, 1, 0.01, 45

# Extras
import numpy as np
# import plotly.express as px



from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor




from bayesian_opt_funcs import expected_improvement, propose_location, create_plots, plot_convergence

# Initialising variables
run_count = 1
plot_list = []
conv_plot = None

# Initialising plots


# Initialise the app
app = dash.Dash(__name__)



app.layout = html.Div(children=[
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(className='four columns div-user-controls', 
                                    children = [
                                        html.H1('Bayesian Optimisation'),
                                        html.H6('Formula'),
                                        dcc.Input(
                                            id='formula_text',
                                            type='text',
                                            value='-np.sin(3*X) - X**2 + 0.7*X',
                                            placeholder='-np.sin(3*X) - X**2 + 0.7*X',
                                            style={
                                                "marginBottom": "20px",
                                                "background-color": "#31302F",
                                                "color": "white",
                                                "width": "100%"}),

                                        html.Div([
                                            html.Div([
                                                html.H6('Lower Bound'),
                                                dcc.Input(
                                                    id='lower_bound',
                                                    type='number',
                                                    value=-1,
                                                    style={
                                                        # "margin-bottom": "20px",
                                                        "background-color": "#31302F",
                                                        "color": "white"})
                                            ], className="six columns"),

                                            html.Div([
                                                html.H6('Upper Bound'),
                                                dcc.Input(
                                                    id='upper_bound',
                                                    type='number',
                                                    value=2,
                                                    style={
                                                        # "margin-bottom": "20px",
                                                        "background-color": "#31302F",
                                                        "color": "white"}),
                                            ], className="six columns"),
                                        ], className="row",
                                        style={"margin-bottom": "20px"}),

                                        html.H6('Noise'),
                                        dcc.Input(
                                            id='noise',
                                            type='number',
                                            value=0.15,
                                            style={
                                                "margin-bottom": "20px",
                                                "background-color": "#31302F",
                                                "color": "white"}),

                                        html.H6('Number of Iterations'),
                                        dcc.Input(
                                            id='num_iter',
                                            type='number',
                                            value=30,
                                            style={
                                                "margin-bottom": "20px",
                                                "background-color": "#31302F",
                                                "color": "white"}),
                                        html.Br(),
                                        html.Button('Run', id='run'),
                                        # html.P('''Visualising time series with Plotly - Dash'''),
                                        # html.P('''Pick one or more stocks from the dropdown below.''')
                                    ]),  # Define the left element




                                  html.Div(className='eight columns div-for-charts bg-grey',
                                    children = [
                                        html.Div(className='eight rows div-for-iter',
                                            children = [
                                                html.Br(),
                                                dcc.Slider(
                                                    id='iter_slider',
                                                    min=1,
                                                    max=10,
                                                    marks={i: '{}'.format(i) for i in range(11)},
                                                    value=1,
                                                    
                                                ),

                                                dcc.Graph(id = 'init_plot',
                                                    style={
                                                        "width": "100%",
                                                        "padding-top": "0px",
                                                        "height": "45vh"
                                                    }),
                                                dcc.Graph(id = 'convergence_plot',
                                                    style={
                                                        "width": "100%",
                                                        "height": "45vh"
                                                    })
                                            ])
                                    ])  # Define the right element
                                  ])
                                ])





@app.callback(
        [Output('iter_slider', component_property='max'),
        Output('iter_slider', component_property='value'),
        Output('iter_slider', component_property='marks'),
        Output('init_plot', 'figure'),
        Output('convergence_plot', 'figure')],
        [Input('run', 'n_clicks'),
        Input('iter_slider', 'value')],
        [State('formula_text', 'value'),
        State('noise', 'value'),
        State('lower_bound', 'value'),
        State('upper_bound', 'value'),
        State('num_iter', 'value')]
        )
def update_b_to_default(n, slider_choice, form, noise, lower, upper, iterations):
    if not n:
        raise PreventUpdate

    global run_count, plot_list, conv_plot

    if n == run_count:

        marks = {i: '{}'.format(i) for i in range(iterations+1)}

        # Iterations
        ## Iterative plots
        bounds = np.array([[lower, upper]])
        noise = noise
        # Number of iterations
        n_iter = iterations
        
        def f(X, noise=noise):
            return eval(form + ' + noise * np.random.randn(*X.shape)')

        X_init = np.array([np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])])
        Y_init = f(X_init)


        ## General Plot
        # Dense grid of points within bounds
        X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
        # Noise-free objective function values at X 
        Y = f(X,0)

        ## GRP
        # Gaussian process with Mat??rn kernel as surrogate model
        m52 = Matern(length_scale=1.0, nu=2.5)
        gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**1.6)

        # Initialize samples
        X_sample = X_init
        Y_sample = Y_init


        plot_list=[]
        for i in range(n_iter):
            # Update Gaussian process with existing samples
            gpr.fit(X_sample, Y_sample)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)
            
            # Obtain next noisy sample from the objective function
            Y_next = f(X_next, noise)
            
            xx = create_plots(f, gpr, X, Y, X_sample, Y_sample, expected_improvement(X, X_sample, Y_sample, gpr), X_next, bounds)
            
            plot_list.append(xx)
            
            # Add sample to previous samples
            X_sample = np.vstack((X_sample, X_next))
            Y_sample = np.vstack((Y_sample, Y_next))

        
        conv_plot = plot_convergence(X_sample, Y_sample)
        run_count += 1



        return [iterations, 1, marks, plot_list[0], conv_plot]

    
    else:
        marks = {i: '{}'.format(i) for i in range(iterations+1)}
        return [iterations, slider_choice, marks, plot_list[slider_choice - 1], conv_plot]



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)




