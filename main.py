import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import matplotlib.pyplot as plt

print("CUDA is available : ", torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
torch.backends.cudnn.benchmark = True
print(device)

params = {'batch_size': 4, 'shuffle': True, 'num_workers' : 2}
EPOCHS = 1

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

temp_values_BLOCKTYPE = np_data[:, 3]
incoded_BLOCKTYPE = []
for i in range(len(temp_values_BLOCKTYPE)):
    if temp_values_BLOCKTYPE[i] == "BLK":
        incoded_BLOCKTYPE.append(0)
    elif temp_values_BLOCKTYPE[i] == "BRC":
        incoded_BLOCKTYPE.append(1)
for i in range(len(incoded_BLOCKTYPE)):
    np_data[i, 3] = incoded_BLOCKTYPE[i]

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

temp_values_INOUT = np_data[:, 12]
incoded_INOUT = []
for i in range(len(temp_values_INOUT)):
    if temp_values_INOUT[i] == "IN":
        incoded_INOUT.append(0)
    elif temp_values_INOUT[i] == "OUT":
        incoded_INOUT.append(1)
for i in range(len(incoded_INOUT)):
    np_data[i, 12] = incoded_INOUT[i]

encoder = LabelEncoder()
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
np_data = np_data.astype(np.float)

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
np_x_train = np.delete(np_x_train, 14, axis=1)
np_x_train = np.delete(np_x_train, 13, axis=1)
np_y_train_lt = np.delete(np_y_train, 0, axis=1)
np_y_test_lt = np.delete(np_y_test, 0, axis=1)
np_y_train_plt = np.delete(np_y_train, 1, axis=1)
np_y_test_plt = np.delete(np_y_test, 1, axis=1)

# print(np_x_train)
print("x_train shape:", np_x_train.shape)
# print(np_x_test)
print("x_test shape:", np_x_test.shape)
# print(np_y_train)
print("y_train shape:", np_y_train.shape)
# print(np_y_test)
print("y_test shape:", np_y_test.shape)

x_train_mins = np.min(np_x_train, axis=0)
x_test_mins = np.min(np_x_test, axis=0)
x_train_maxes = np.max(np_x_train, axis=0)
x_test_maxes = np.max(np_x_test, axis=0)

for i in range(np.size(np_x_train, 1)):
    if not x_train_maxes[i] == x_train_mins[i]:
        for j in range(np.size(np_x_train, 0)):
            np_x_train[j,i] = (np_x_train[j,i] - x_train_mins[i]) / (x_train_maxes[i] - x_train_mins[i])
for i in range(np.size(np_x_test, 1)):
    if not x_train_maxes[i] == x_train_mins[i]:
        for j in range(np.size(np_x_test, 0)):
            np_x_test[j,i] = (np_x_test[j,i] - x_test_mins[i]) / (x_test_maxes[i] - x_test_mins[i])

print(np_x_test)
print(np_x_test.shape)
print(np_x_train)
print(np_x_train.shape)

torch_x_train = torch.tensor(np_x_train, dtype=torch.float32)
# print(torch_x_train)
# print(torch_x_train.size())

torch_x_test = torch.tensor(np_x_test, dtype=torch.float32)
# print(torch_x_test)
# print(torch_x_test.size())

torch_y_train_lt = torch.tensor(np_y_train_lt, dtype=torch.float32)
# print(torch_y_train_lt)
# print(torch_y_train_lt.size())

torch_y_train_plt = torch.tensor(np_y_train_plt, dtype=torch.float32)
# print(torch_y_train_plt)
# print(torch_y_train_plt.size())

torch_y_test_lt = torch.tensor(np_y_test_lt, dtype=torch.float32)
# print(torch_y_test_lt)
# print(torch_y_test_lt.size())

torch_y_test_plt = torch.tensor(np_y_test_plt, dtype=torch.float32)
# print(torch_y_test_plt)
# print(torch_y_test_plt.size())

dataset_train_lt = torch.utils.data.TensorDataset(torch_x_train, torch_y_train_lt)
dataset_train_plt = torch.utils.data.TensorDataset(torch_x_train, torch_y_train_plt)
dataset_test_lt = torch.utils.data.TensorDataset(torch_x_test, torch_y_test_lt)
dataset_test_plt = torch.utils.data.TensorDataset(torch_x_test, torch_y_test_plt)

trainloader_lt = torch.utils.data.DataLoader(dataset_train_lt, **params)
testloader_lt = torch.utils.data.DataLoader(dataset_test_lt, **params)
trainloader_plt = torch.utils.data.DataLoader(dataset_train_plt, **params)
testloader_plt = torch.utils.data.DataLoader(dataset_test_plt, **params)


class Net_LT(nn.Module):
    def __init__(self):
        super(Net_LT, self).__init__()

        self.input = nn.Linear(13, 100)
        self.hidden1 = nn.Linear(100, 100)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.hidden1(self.input(x))
        x = x.view(-1, 100)
        x = self.fc(x)
        return x


def run():
    netlt = Net_LT()
    netlt = netlt.cuda()

    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(netlt.parameters(), lr=0.005, momentum=0.9)

    training_loss_history = []
    test_loss_history = []

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, loaded in enumerate(trainloader_lt, start=0):
            inputs, targets = loaded
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = netlt(inputs)
            loss = criterion(outputs, targets).to(device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))

            training_loss_history.append(running_loss / 200)

            with torch.no_grad():
                running_test_loss = 0.0
                for i, loaded_ in enumerate(testloader_lt, start=0):
                    test_inputs, test_targets = loaded_
                    test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                    test_outputs = netlt(test_inputs)
                    test_loss = criterion(test_outputs, test_targets).to(device)
                    running_test_loss += test_loss.item()
                test_loss_history.append(running_test_loss / i)
            running_loss = 0.0
    print("TRAINING DONE")

    # Calculate RMSE, MAPE
    MSE = 0.
    MAPE = 0.
    with torch.no_grad():
        for i, loaded in enumerate(testloader_lt, start=0):
            output, targets = loaded
            output, targets = output.to(device), targets.to(device)
            outputs = netlt(output)
            predicted = outputs.cpu()
            for j in range(4):
                if targets[j] != 0:
                    MSE += ((targets[j].cpu().item() - predicted[j, 0]) ** 2)
                    MAPE += np.abs(targets[j].cpu().item() - predicted[j, 0]) / targets[j].cpu().item()
    print("LT RMSE accuracy = ", np.sqrt(MSE / 600))
    print("LT MAPE accuracy = ", np.abs(MAPE) / 600)

if __name__ == 'main':
    torch.multiprocessing.freeze_support()
    run()


class Net_PLT(nn.Module):
    def __init__(self):
        super(Net_PLT, self).__init__()

        self.input = nn.Linear(13, 100)
        self.hidden1 = nn.Linear(100, 100)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.hidden1(self.input(x))
        x = x.view(-1, 100)
        x = self.fc(x)
        return x


def run1():
    netplt = Net_PLT()
    netplt = netplt.cuda()

    criterion = nn.L1Loss().to(device)
    optimizer = optim.SGD(netplt.parameters(), lr=0.005, momentum=0.9)

    training_loss_history = []
    test_loss_history = []

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, loaded in enumerate(trainloader_plt, start=0):
            inputs, targets = loaded
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = netplt(inputs)
            loss = criterion(outputs, targets).to(device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))

            training_loss_history.append(running_loss / 200)

            with torch.no_grad():
                running_test_loss = 0.0
                for i, loaded_ in enumerate(testloader_plt, start=0):
                    test_inputs, test_targets = loaded_
                    test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                    test_outputs = netplt(test_inputs)
                    test_loss = criterion(test_outputs, test_targets).to(device)
                    running_test_loss += test_loss.item()
                test_loss_history.append(running_test_loss / i)
            running_loss = 0.0
    print("TRAINING DONE")

    MSE = 0.
    MAPE = 0.
    with torch.no_grad():
        for i, loaded in enumerate(testloader_plt, start=0):
            output, targets = loaded
            output, targets = output.to(device), targets.to(device)
            outputs = netplt(output)
            predicted = outputs.cpu()
            for j in range(4):
                if targets[j] != 0:
                    MSE += ((targets[j].cpu().item() - predicted[j, 0]) ** 2)
                    MAPE += np.abs(targets[j].cpu().item() - predicted[j, 0]) / targets[j].cpu().item()
    print("PLT RMSE accuracy = ", np.sqrt(MSE / 600))
    print("PLT MAPE accuracy = ", np.abs(MAPE) / 600)


if __name__ == 'main':
    torch.multiprocessing.freeze_support()
    run1()





'''
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i in range(list(torch_x_train.size())[0]):
        inputs = torch_x_train[i,:]
        targets = torch_y_train_plt[i].reshape(1,1)
        optimizer.zero_grad()
        outputs = netplt(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))

        training_loss_history.append(running_loss / 200)

        with torch.no_grad():
            running_test_loss = 0.0
            for i in range(list(torch_x_test.size())[0]):
                test_inputs = torch_x_test[i,:]
                test_targets = torch_y_test_plt[i].reshape(1,1)
                test_outputs = netplt(test_inputs)
                test_loss = criterion(test_outputs, test_targets)
                running_test_loss += test_loss.item()
            test_loss_history.append(running_test_loss / i)
        running_loss = 0.0
print("TRAINING DONE")

# Calculate RMSE, MAPE
MSE = 0.
MAPE = 0.
with torch.no_grad():
    for i in range(list(torch_x_test.size())[0]):
        output = torch_x_test[i,:]
        targets = torch_y_test_plt[i,:]
        outputs = netplt(output)
        predicted = outputs.cpu().item()
        if targets[0] != 0:
            MSE += ((targets[0].cpu().item() - predicted) ** 2)
            MAPE += (targets[0].cpu().item() - predicted) / targets[0].cpu().item()
print("PLT RMSE accuracy = ", np.sqrt(MSE / 600))
print("PLT MAPE accuracy = ", np.abs(MAPE) / 600)
'''
