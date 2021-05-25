import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sklearn.preprocessing as prep
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

print("CUDA is available : ", torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
torch.backends.cudnn.benchmark = True
print(device)

params = {'batch_size': 4, 'shuffle': True}
EPOCHS = 2

BATCH_SIZE = 256
params = {'batch_size': BATCH_SIZE, 'shuffle': True}
params_test = {'batch_size': BATCH_SIZE, 'shuffle': False}
EPOCHS = 100000

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
max_deviations = 4
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

# numpy arrays to pytorch tensors
torch_x_train = torch.tensor(np_x_train, dtype=torch.float32, device=device)
torch_x_test = torch.tensor(np_x_test, dtype=torch.float32, device=device)
torch_y_train_lt = torch.tensor(np_y_train_lt, dtype=torch.float32, device=device)
torch_y_train_plt = torch.tensor(np_y_train_plt, dtype=torch.float32, device=device)
torch_y_test_lt = torch.tensor(np_y_test_lt, dtype=torch.float32, device=device)
torch_y_test_plt = torch.tensor(np_y_test_plt, dtype=torch.float32, device=device)

# create datasets
dataset_train_lt = torch.utils.data.TensorDataset(torch_x_train, torch_y_train_lt)
dataset_train_plt = torch.utils.data.TensorDataset(torch_x_train, torch_y_train_plt)
dataset_test_lt = torch.utils.data.TensorDataset(torch_x_test, torch_y_test_lt)
dataset_test_plt = torch.utils.data.TensorDataset(torch_x_test, torch_y_test_plt)

# create dataloaders for training
trainloader_lt = torch.utils.data.DataLoader(dataset_train_lt, **params)
testloader_lt = torch.utils.data.DataLoader(dataset_test_lt, **params_test)
trainloader_plt = torch.utils.data.DataLoader(dataset_train_plt, **params)
testloader_plt = torch.utils.data.DataLoader(dataset_test_plt, **params_test)

# model
input_size = 11
output_size = 1
learning_rate = 0.0005

lt_model = nn.Linear(input_size, output_size, bias=True).to(device)
loss_method_lt = nn.MSELoss()
optimizer_lt = torch.optim.SGD(lt_model.parameters(), lr=learning_rate, momentum=0.7)

plt_model = nn.Linear(input_size, output_size, True).to(device)
loss_method_plt = nn.MSELoss()
optimizer_plt = torch.optim.SGD(plt_model.parameters(), lr=learning_rate, momentum=0.7)

running_loss = 0.
for epoch in range(EPOCHS):
    torch_y_train_pred_lt = lt_model(torch_x_train)
    loss = loss_method_lt(torch_y_train_pred_lt, torch_y_train_lt).to(device)
    loss.backward()
    optimizer_lt.step()
    optimizer_lt.zero_grad()
    running_loss += loss.cpu().item()
    if epoch % 1000 == 999:
        print('[Epoch : %d] training loss: %.3f' %(epoch + 1, running_loss / 1000))
        running_loss = 0.

running_loss = 0.
for epoch in range(EPOCHS):
    torch_y_train_pred_plt = plt_model(torch_x_train)
    loss = loss_method_plt(torch_y_train_pred_plt, torch_y_train_plt).to(device)
    loss.backward()
    optimizer_plt.step()
    optimizer_plt.zero_grad()
    running_loss += loss.cpu().item()
    if epoch % 1000 == 999:
        print('[Epoch : %d] training loss: %.3f' %(epoch + 1, running_loss / 1000))
        running_loss = 0.
print("TRAINING DONE")

with torch.no_grad():
    torch_predict_lt = lt_model(torch_x_test)
    RSME_lt = np.sqrt(loss_method_lt(torch_predict_lt.cpu(), torch_y_test_lt.cpu()))
    print("LT RSME : ", RSME_lt)

np_predict_lt = torch_predict_lt.cpu().numpy()
MAPE_lt = mean_absolute_percentage_error(np_y_test_lt, np_predict_lt)
print("LT MAPE : ", MAPE_lt)

with torch.no_grad():
    torch_predict_plt = plt_model(torch_x_test)
    RSME_plt = np.sqrt(loss_method_plt(torch_predict_plt.cpu(), torch_y_test_plt.cpu()))
    print("PLT RSME : ", RSME_plt)

np_predict_plt = torch_predict_plt.cpu().numpy()
MAPE_plt = mean_absolute_percentage_error(np_y_test_plt, np_predict_plt)
print("PLT MAPE : ", MAPE_plt)

concat_lt = np.concatenate((np_y_test_lt, np.array(np_predict_lt).reshape(-1,1)), axis=1)
concat_plt = np.concatenate((np_y_test_plt, np.array(np_predict_plt).reshape(-1,1)), axis=1)
sorted_lt = concat_lt[np.argsort(concat_lt[:,0])]
sorted_plt = concat_plt[np.argsort(concat_plt[:,0])]
np_predicted_lt = sorted_lt[:,1]
np_predicted_plt = sorted_plt[:,1]
np_y_test_lt = sorted_lt[:,0]
np_y_test_plt = sorted_plt[:,0]

# plot results
plt.figure(figsize=(10,10), dpi=200)
plt.subplot(2,1,1)
plt.scatter(range(len(np_predict_lt)), np_y_test_lt, label='lt true')
plt.scatter(range(len(np_predict_lt)), np_predict_lt, label='lt predict')
plt.title('LT Test Results')
plt.xlabel('iterations')
plt.ylabel('LT')
plt.legend()
plt.subplot(2,1,2)
plt.scatter(range(len(np_predict_plt)), np_y_test_plt, label='plt true')
plt.scatter(range(len(np_predict_plt)), np_predict_plt, label='plt predict')
plt.title('PLT Test Results')
plt.xlabel('iterations')
plt.ylabel('PLT')
plt.legend()
plt.show()

print(lt_model.weight)
print(plt_model.weight)