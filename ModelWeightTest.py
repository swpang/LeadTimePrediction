import pandas as pd
import numpy as np
import torch
import torch.nn as nn


def readfile(dir):
    dataframe = pd.read_csv(dir, sep=',', header=0, index_col=0, dtype='float32')
    return dataframe


def dataframe_to_numpy(dataframe):
    np_data = dataframe.to_numpy()
    return np_data


NAME = 'DiscussionTestData.csv'
data = readfile(NAME)
np_data = dataframe_to_numpy(data)


class NetLT(nn.Module):
    def __init__(self):
        super(NetLT, self).__init__()
        self.input = nn.Linear(11, 100, bias=True)
        self.hidden1 = nn.Linear(100, 100, bias=True)
        self.hidden2 = nn.Linear(100, 100, bias=True)
        self.fc = nn.Linear(100, 1, bias=True)

    def forward(self, x):
        x = self.input(x)
        x = torch.relu(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.fc(x)
        return x


class NetPLT(nn.Module):
    def __init__(self):
        super(NetPLT, self).__init__()
        self.input = nn.Linear(11, 100, bias=True)
        self.hidden1 = nn.Linear(100, 100, bias=True)
        self.hidden2 = nn.Linear(100, 100, bias=True)
        self.hidden3 = nn.Linear(100, 100, bias=True)
        self.fc = nn.Linear(100, 1, bias=True)

    def forward(self, x):
        x = self.input(x)
        x = torch.relu(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.hidden3(x)
        x = torch.relu(x)
        x = self.fc(x)
        return x


print('lt')
# load trained model
model_lt = NetLT()
model_lt.load_state_dict(torch.load('lt_model_hidden3.pt'))
model_lt.eval()

with torch.no_grad():
    for i in range(len(np_data[:,0])):
        np_temp_data = np.array(np_data[i,:])
        torch_temp_data = torch.tensor(np_temp_data, dtype=torch.float32)
        output = model_lt(torch_temp_data)
        print(output.numpy())

print('plt')
# load trained model
model_plt = NetPLT()
model_plt.load_state_dict(torch.load('plt_model_hidden3.pt'))
model_plt.eval()

with torch.no_grad():
    for i in range(len(np_data[:,0])):
        torch_temp_data = torch.tensor(np_data[i,:], dtype=torch.float32)
        output = model_plt(torch_temp_data)
        print(output.numpy())