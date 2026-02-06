import os
import numpy as np
import pandas as pd
import pickle
from loco_methods import *
from ml_models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import *
from scipy.stats import *
import matplotlib.pyplot as plt
import vimpy

out = "Output_Data"
if not os.path.exists(out):
    os.mkdir(out)
output = "Output_Data/Case_Study"
if not os.path.exists(output):
    os.mkdir(output)
fig = "Figure"
if not os.path.exists(fig):
    os.mkdir(fig)
figure = "Figure/Case_Study"
if not os.path.exists(figure):
    os.mkdir(figure)

data_name = 'rosmap86'
tar = "y_reg"
n = 109
m = 17

train = pd.read_csv(f"{data_name}_train.csv")
test = pd.read_csv(f"{data_name}_test.csv")
y_train = train[tar]
X_train = train.drop(columns=[tar])
y_test = test[tar]
X_test = test.drop(columns=[tar])

### Test Mean Squared Errors

adamp_mse = [0]*50
mp_mse = [0]*50
split1_mse = [0]*50
split2_mse = [0]*50
cpi_mse = [0]*50
vim_mse = [0]*50

locoadamp = pd.read_csv(f"{output}/{data_name}_LOCO-AdaMP_RandomForest_n{n}_m{m}.csv")
locoadamp = locoadamp.sort_values(by='pval', ascending=True)
print("LOCO-AdaMP-------------------------------\n",locoadamp.head(10))
locomp = pd.read_csv(f"{output}/{data_name}_LOCO-MP_RandomForest_n{n}_m{m}.csv")
locomp = locomp.sort_values(by='pval', ascending=True)
print("LOCO-MP-------------------------------\n",locomp.head(10))
locosplit1 = pd.read_csv(f"{output}/{data_name}_LOCO-Split0.75_RandomForest.csv")
locosplit1 = locosplit1.sort_values(by='pval', ascending=True)
print("LOCO-Split0.5-------------------------------\n",locosplit1.head(10))
locosplit2 = pd.read_csv(f"{output}/{data_name}_LOCO-Split0.5_RandomForest.csv")
locosplit2 = locosplit2.sort_values(by='pval', ascending=True)
print("LOCO-Split0.75-------------------------------\n",locosplit2.head(10))
cpi = pd.read_csv(f"{output}/{data_name}_CPI_RandomForest.csv")
cpi = cpi.sort_values(by='pval', ascending=True)
print("CPI-------------------------------\n",cpi.head(10))
vim = pd.read_csv(f"{output}/{data_name}_VIM_RandomForest.csv")
vim = vim.sort_values(by='pval', ascending=True)
print("VIM-------------------------------\n",vim.head(10))

for i in range(50):
    
    adamp_fea = locoadamp['feature'][0:i+1]
    rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=42, bootstrap=False)
    rf.fit(X_train[adamp_fea],y_train)
    y_pred = rf.predict(X_test[adamp_fea])
    adamp_mse[i] = mean_squared_error(y_test, y_pred)
  
    mp_fea = locomp['feature'][0:i+1]
    rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=42, bootstrap=False)
    rf.fit(X_train[mp_fea],y_train)
    y_pred = rf.predict(X_test[mp_fea])
    mp_mse[i] = mean_squared_error(y_test, y_pred)
   
    split1_fea = locosplit1['feature'][0:i+1]
    rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=42, bootstrap=False)
    rf.fit(X_train[split1_fea],y_train)
    y_pred = rf.predict(X_test[split1_fea])
    split1_mse[i] = mean_squared_error(y_test, y_pred)

    split2_fea = locosplit2['feature'][0:i+1]
    rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=42, bootstrap=False)
    rf.fit(X_train[split2_fea],y_train)
    y_pred = rf.predict(X_test[split2_fea])
    split2_mse[i] = mean_squared_error(y_test, y_pred)

    cpi_fea = cpi['feature'][0:i+1]
    rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=42, bootstrap=False)
    rf.fit(X_train[cpi_fea],y_train)
    y_pred = rf.predict(X_test[cpi_fea])
    cpi_mse[i] = mean_squared_error(y_test, y_pred)

    vim_fea = vim['feature'][0:i+1]
    rf = RandomForestRegressor(n_estimators=200, max_features= 1/3, min_samples_leaf=5, random_state=42, bootstrap=False)
    rf.fit(X_train[vim_fea],y_train)
    y_pred = rf.predict(X_test[vim_fea])
    vim_mse[i] = mean_squared_error(y_test, y_pred)
    
plt.rcParams['axes.titlesize'] = 12.5
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5

color_list = ["#d44848", "#6eae6e","#f6a257","#4f91c1", "#9e7bbe", "#ed879c"]#red, green, orange, blue, purple, pink

x = np.arange(1, 51)
plt.figure(figsize=(4, 3))  
plt.plot(x, adamp_mse, marker='o', markersize=4, label='LOCO-AdaMP', color = color_list[0], zorder=10)
plt.plot(x, mp_mse, marker='o', markersize=4, label='LOCO-MP', color = color_list[1]) 
plt.plot(x, split1_mse, marker='o', markersize=4, label='LOCO-Split0.5', color = color_list[2])
plt.plot(x, split2_mse, marker='o', markersize=4, label='LOCO-Split0.75', color = color_list[3]) 
plt.plot(x, cpi_mse, marker='o', markersize=4, label='CPI', color = color_list[4])
plt.plot(x, vim_mse, marker='o', markersize=4, label='VIM', color = color_list[5])

plt.title("Random Forest")
plt.xlabel("Number of Features")
plt.ylabel("Test Mean Squared Error")
plt.grid(True, zorder=0, alpha=0.5, linestyle='--')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)
)
plt.savefig(f'{figure}/{data_name}_tmse.png', dpi=300, bbox_inches="tight")
plt.show()
plt.close()