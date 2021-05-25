import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn import neural_network as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


def readfile(dir):
    dataframe = pd.read_csv(dir, sep=',', header=0, index_col=0, dtype=str)
    return dataframe


def dataframe_to_numpy(dataframe):
    np_data = dataframe.to_numpy()
    return np_data


NAME = 'DSME_assembly.csv'
data = readfile(NAME)
# print(data)

np_data = dataframe_to_numpy(data)
# print(np_data)

# Data Preprocessing

# BLOCKTYPE encoding
temp_values_BLOCKTYPE = np_data[:, 3]
incoded_BLOCKTYPE = []
for i in range(len(temp_values_BLOCKTYPE)):
    if temp_values_BLOCKTYPE[i] == "BLK":
        incoded_BLOCKTYPE.append(0)
    elif temp_values_BLOCKTYPE[i] == "BRC":
        incoded_BLOCKTYPE.append(1)
for i in range(len(incoded_BLOCKTYPE)):
    np_data[i, 3] = incoded_BLOCKTYPE[i]

# SHIPTYPE encoding
temp_values_SHIPTYPE = np_data[:, 4]
incoded_SHIPTYPE = []
for i in range(len(temp_values_SHIPTYPE)):
    if temp_values_SHIPTYPE[i] == "CONT":
        incoded_SHIPTYPE.append(0)
    elif temp_values_SHIPTYPE[i] == "COT":
        incoded_SHIPTYPE.append(1)
    elif temp_values_SHIPTYPE[i] == "LNGC":
        incoded_SHIPTYPE.append(2)
for i in range(len(incoded_SHIPTYPE)):
    np_data[i, 4] = incoded_SHIPTYPE[i]

# 사내외 encoding
temp_values_INOUT = np_data[:, 12]
incoded_INOUT = []
for i in range(len(temp_values_INOUT)):
    if temp_values_INOUT[i] == "IN":
        incoded_INOUT.append(0)
    elif temp_values_INOUT[i] == "OUT":
        incoded_INOUT.append(1)
for i in range(len(incoded_INOUT)):
    np_data[i, 12] = incoded_INOUT[i]

# encoding OTHER
encoder = prep.LabelEncoder()
temp_values_PCGCODE = np_data[:, 9]
incoded_PCGCODE = encoder.fit_transform(temp_values_PCGCODE)
for i in range(len(incoded_PCGCODE)):
    np_data[i, 9] = incoded_PCGCODE[i]
temp_values_SHIP = np_data[:, 10]
incoded_SHIP = encoder.fit_transform(temp_values_SHIP)
for i in range(len(incoded_SHIP)):
    np_data[i, 10] = incoded_SHIP[i]
temp_values_SHOP = np_data[:, 11]
incoded_SHOP = encoder.fit_transform(temp_values_SHOP)
for i in range(len(incoded_SHOP)):
    np_data[i, 11] = incoded_SHOP[i]

# cast all data to type float64
np_data = np_data.astype('float64')
np.random.shuffle(np_data)

# remove all non-positive target data
temp = np_data[:,14]
idx = []
for i in range(len(temp)):
    if not temp[i] > 0.:
        idx.append(i)
np_data = np.delete(np_data, idx, axis=0)

# Remove Outliers (data outside 4 sigma)
max_deviations = 3
temp_lt_values = np_data[:,14]
mean_lt_value = np.mean(temp_lt_values)
print(mean_lt_value)
std_lt_value = np.std(temp_lt_values)
print(std_lt_value)
distance_from_mean_lt = abs(temp_lt_values - mean_lt_value)
invalid_data_lt_idx = []
for idx, distance in enumerate(distance_from_mean_lt):
    if distance >= (max_deviations * std_lt_value):
        invalid_data_lt_idx.append(idx)
temp_plt_values = np_data[:,13]
mean_plt_value = np.mean(temp_plt_values)
std_plt_value = np.std(temp_plt_values)
distance_from_mean_plt = abs(temp_plt_values - mean_plt_value)
invalid_data_plt_idx = []
for idx, distance in enumerate(distance_from_mean_plt):
    if distance >= (max_deviations * std_plt_value):
        invalid_data_plt_idx.append(idx)

set_lt = set(invalid_data_lt_idx)
set_plt = set(invalid_data_plt_idx)
plt_items_not_in_lt = list(set_plt - set_lt)
invalid_data_idx = invalid_data_lt_idx + plt_items_not_in_lt
print(invalid_data_idx)
np_data = np.delete(np_data, invalid_data_idx, axis=0)

# take first 200 items of each
np_x_train = np.empty((1, 15))
np_x_test = np.empty((1, 15))

count_cont = 0
count_cot = 0
count_lngc = 0

for i in range(np_data.shape[0]):
    if np_data[i, 4] == 0:
        if count_cont < 200:
            np_x_test = np.vstack((np_x_test, np_data[i, :]))
            count_cont += 1
        else:
            np_x_train = np.vstack((np_x_train, np_data[i, :]))
    elif np_data[i, 4] == 1:
        if count_cot < 200:
            np_x_test = np.vstack((np_x_test, np_data[i, :]))
            count_cot += 1
        else:
            np_x_train = np.vstack((np_x_train, np_data[i, :]))
    elif np_data[i, 4] == 2:
        if count_lngc < 200:
            np_x_test = np.vstack((np_x_test, np_data[i, :]))
            count_lngc += 1
        else:
            np_x_train = np.vstack((np_x_train, np_data[i, :]))

np_x_test = np.delete(np_x_test, 0, axis=0)
np_x_train = np.delete(np_x_train, 0, axis=0)
np_y_test = np_x_test[:, 13:15]
np_y_train = np_x_train[:, 13:15]
np_x_test = np.delete(np_x_test, 14, axis=1)
np_x_test = np.delete(np_x_test, 13, axis=1)
np_x_test = np.delete(np_x_test, 12, axis=1)
np_x_test = np.delete(np_x_test, 2, axis=1).astype('float64')
np_x_train = np.delete(np_x_train, 14, axis=1)
np_x_train = np.delete(np_x_train, 13, axis=1)
np_x_train = np.delete(np_x_train, 12, axis=1)
np_x_train = np.delete(np_x_train, 2, axis=1).astype('float64')
np_y_train_lt = np.delete(np_y_train, 0, axis=1).astype('float64')
np_y_test_lt = np.delete(np_y_test, 0, axis=1).astype('float64')
np_y_train_plt = np.delete(np_y_train, 1, axis=1).astype('float64')
np_y_test_plt = np.delete(np_y_test, 1, axis=1).astype('float64')

x_train_mins = np.amin(np_x_train, axis=0)
x_test_mins = np.amin(np_x_test, axis=0)
x_train_maxes = np.amax(np_x_train, axis=0)
x_test_maxes = np.amax(np_x_test, axis=0)

for i in range(np.size(np_x_train, 1)):
    if not x_train_maxes[i] == x_train_mins[i]:
        for j in range(np.size(np_x_train, 0)):
            np_x_train[j,i] = (np_x_train[j,i] - x_train_mins[i]) / (x_train_maxes[i] - x_train_mins[i])
for i in range(np.size(np_x_test, 1)):
    if not x_test_maxes[i] == x_test_mins[i]:
        for j in range(np.size(np_x_test, 0)):
            np_x_test[j,i] = (np_x_test[j,i] - x_test_mins[i]) / (x_test_maxes[i] - x_test_mins[i])

print(np_x_test)
print(np_x_test.shape)
print(np_x_train)
print(np_x_train.shape)

# print(np_x_test)

# Define Model
mlp_lt = nn.MLPRegressor(
    hidden_layer_sizes=[100],
    activation='relu',
    solver='sgd',
    alpha=0.0001,
    learning_rate='adaptive',
    learning_rate_init=0.002,
    max_iter=10000,
    shuffle=True,
    random_state=None,
    tol=1e-4,
    verbose=False,
    warm_start=False,
    momentum=0.7,
    nesterovs_momentum=True,
    early_stopping=False,
    n_iter_no_change=10,
    max_fun=1500
)

mlp_plt = nn.MLPRegressor(
    hidden_layer_sizes=[100],
    activation='relu',
    solver='sgd',
    alpha=0.0001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=100000,
    shuffle=True,
    random_state=None,
    tol=1e-4,
    verbose=False,
    warm_start=False,
    momentum=0.7,
    nesterovs_momentum=True,
    early_stopping=False,
    n_iter_no_change=10,
    max_fun=1500
)

# train model
mlp_lt.fit(np_x_train, np_y_train_lt)
mlp_plt.fit(np_x_train, np_y_train_plt)

# test model
np_predict_lt = mlp_lt.predict(np_x_test)
np_predict_plt = mlp_plt.predict(np_x_test)
# print(np_predict_lt.shape)
# print(np_predict_plt.shape)

# calculate loss
RMSE_lt = mean_squared_error(np_y_test_lt, np_predict_lt, squared=False)
RMSE_plt = mean_squared_error(np_y_test_plt, np_predict_plt, squared=False)
MAPE_lt = mean_absolute_percentage_error(np_y_test_lt, np_predict_lt)
MAPE_plt = mean_absolute_percentage_error(np_y_test_plt, np_predict_plt)

# print results
print("LT")
print("RMSE : ", RMSE_lt)
print("MAPE : ", MAPE_lt)
print("____________________________________")
print("PLT")
print("RMSE : ", RMSE_plt)
print("MAPE : ", MAPE_plt)
print("____________________________________")
print("\n")

concat_lt = np.concatenate((np_y_test_lt, np.array(np_predict_lt).reshape(-1, 1)), axis=1)
concat_plt = np.concatenate((np_y_test_plt, np.array(np_predict_plt).reshape(-1, 1)), axis=1)
sorted_lt = concat_lt[np.argsort(concat_lt[:, 0])]
sorted_plt = concat_plt[np.argsort(concat_plt[:, 0])]
np_predicted_lt = sorted_lt[:, 1]
np_predicted_plt = sorted_plt[:, 1]
np_y_test_lt = sorted_lt[:, 0]
np_y_test_plt = sorted_plt[:, 0]

# plot results
plt.figure(figsize=(10,10), dpi=200)
plt.subplot(2,1,1)
plt.scatter(range(len(np_predicted_lt)), np_y_test_lt, label='lt true')
plt.scatter(range(len(np_predicted_lt)), np_predicted_lt, label='lt predict')
plt.title('LT Test Results')
plt.xlabel('iterations')
plt.ylabel('LT')
plt.legend()
plt.subplot(2,1,2)
plt.scatter(range(len(np_predicted_plt)), np_y_test_plt, label='plt true')
plt.scatter(range(len(np_predicted_plt)), np_predicted_plt, label='plt predict')
plt.title('PLT Test Results')
plt.xlabel('iterations')
plt.ylabel('PLT')
plt.legend()
plt.show()