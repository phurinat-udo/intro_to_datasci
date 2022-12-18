# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import optimize
# %%
raw = pd.read_csv('salary.dat')
raw.head()
raw.columns
# %%
for i in range(len(raw.loc[:, '# Company'])):
    raw.loc[i,'# Company'] = raw.loc[i,'# Company'].strip()
    
print(raw)
# %%
X = raw.loc[:, ' Experience']; y = raw.loc[:, ' Salary']
raw['# Company'].unique()
# %% Plot data
fig_data = px.scatter(raw, x=' Experience', y = ' Salary', color = '# Company')
fig_data.show()
# %% Create a model function
def linear_1v(beta, x):
    yp = beta[1] * x + beta[0]
    return yp

def rss_lin1v(beta, x, y):
    yp = linear_1v(beta, x)
    loss = np.sum((y - yp)**2)
    return loss
# %% 
beta_i = [0.0, 0.0]
# %%
