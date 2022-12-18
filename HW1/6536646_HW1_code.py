# %% Import module

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import mean_squared_error
# %% Read data
df_train = pd.read_csv('train.dat')
df_test = pd.read_csv('test.dat')
# %% Problem 1 plot the data point
fig_data = go.Figure()
fig_data.add_scatter(x = df_train.loc[:, '# X'], y = df_train.loc[:, '     Y'], mode='markers', name='Train data')
fig_data.add_scatter(x = df_test.loc[:, '# X'], y = df_test.loc[:, '     Y'], mode='markers', name='Test data')
fig_data.update_layout(
    title_text = "Problem 1: Data points",
    xaxis_title = "X",
    yaxis_title = "Y",
    autosize=False,
    width = 800,
    height = 800,
    font_size =14
)
fig_data.show()
pio.write_image(fig_data, "prob1.png", width = 2*600, height=1*600, scale=1)
# %% Assign X and y
X_train = df_train.loc[:, '# X']; y_train = df_train.loc[:, '     Y']
X_test = df_test.loc[:, '# X']; y_test = df_test.loc[:, '     Y']
# %% Problem 2 fit polynomial order nth
models = {}
ytr_fitplot = {}
# create x from data 1000 points
xtr_plot = np.linspace(X_train.min(), X_train.max(), 1000)
# Create figure
fig_prob2 = go.Figure()
# Plot training data
fig_prob2.add_scatter(x = X_train, y = y_train, mode='markers', name='Train data')
for i in np.arange(1,11):
    # fit model order i
    models[i] = np.poly1d(np.polyfit(X_train, y = y_train, deg = i))
    # calculate y model for plot
    ytr_fitplot[i] = models[i](xtr_plot)
    # Plot model with training data
    fig_prob2.add_scatter(x = xtr_plot, y = ytr_fitplot[i], name = f"polynomial fit order {i}")
fig_prob2.update_layout(
    title_text = "Problem 2: Polynomial order 1-10",
    xaxis_title = "X",
    yaxis_title = "Y",
    autosize=False,
    width = 1000,
    height = 800,
    font_size =14
)
pio.write_image(fig_prob2, "prob2.png", width = 2*600, height=1*600, scale=1)
# %% Problem 3 determine MSE wrt training data
mse_train = {}
ytr_fit = {}
for i in np.arange(1,11):    
    # calculate y from fitted model
    ytr_fit[i] = models[i](X_train)
    # Calculate MSE
    mse_train[i] = mean_squared_error(y_train, ytr_fit[i])
fig_prob3 = go.Figure()
fig_prob3.add_scatter(x = [j for j in mse_train.keys()], y = [jj for jj in mse_train.values()], mode="markers+lines", name = "MSE")
fig_prob3.update_layout(
    title_text = "Problem 3: Mean-squared Error of The Model wrt. Training Data",
    xaxis_title = "n-th order",
    yaxis_title = "MSE",
    autosize=False,
    width = 800,
    height = 600,
    font_size =14
)
pio.write_image(fig_prob3, "prob3.png", width = 2*600, height=1*600, scale=1)
# %% Problem 4 determine MSE wrt test data
mse_test = {}
yte_fit = {}
for i in np.arange(1,11):    
    # calculate y from fitted model
    yte_fit[i] = models[i](X_test)
    # Calculate MSE
    mse_test[i] = mean_squared_error(y_test, yte_fit[i])
fig_prob4 = go.Figure()
fig_prob4.add_scatter(x = [j for j in mse_test.keys()], y = [jj for jj in mse_test.values()], mode="markers+lines", name = "MSE")
fig_prob4.update_layout(
    title_text = "Problem 4: Mean-squared Error of The Model wrt. Test Data",
    xaxis_title = "n-th order",
    yaxis_title = "MSE",
    autosize=False,
    width = 800,
    height = 600,
    font_size =14
)
pio.write_image(fig_prob4, "prob4.png", width = 2*600, height=1*600, scale=1)
# %% Prob 5 plot combine from prob 3 and 4 
fig_prob5 = go.Figure()
fig_prob5.add_scatter(x = [j for j in mse_train.keys()], y = [jj for jj in mse_train.values()], mode="markers+lines", name = "MSE training data")
fig_prob5.add_scatter(x = [j for j in mse_test.keys()], y = [jj for jj in mse_test.values()], mode="markers+lines", name = "MSE test data")
fig_prob5.update_layout(
    title_text = "Problem 5: Mean-squared Error of The Model wrt. Training and Test Data",
    xaxis_title = "n-th order",
    yaxis_title = "MSE",
    autosize=False,
    width = 800,
    height = 600,
    font_size =14
)
pio.write_image(fig_prob5, "prob5.png", width = 2*600, height=1*600, scale=1)
