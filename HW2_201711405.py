import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn as sk
import sklearn.preprocessing as prep
import sklearn.metrics as met
import matplotlib.pyplot as plt

# check and initialize CUDA
print("CUDA is available : ", torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
torch.backends.cudnn.benchmark = True

# give parameters
BATCH_SIZE = 256
params = {'batch_size': BATCH_SIZE, 'shuffle': True}
params_test = {'batch_size': 1, 'shuffle': False}
EPOCHS = 10000
LEARNING_RATE = 0.002
MOMENTUM = 0.7

def readfile(dir):
    dataframe = pd.read_csv(dir, sep=',', header=0, index_col=0, dtype=str)
    return dataframe


def dataframe_to_numpy(dataframe):
    np_data = dataframe.to_numpy()
    return np_data


def plot(model_name, mnp_predicted_lt, mnp_predicted_plt, mnp_y_test_lt, mnp_y_test_plt):
    concat_lt = np.concatenate((mnp_y_test_lt, np.array(mnp_predicted_lt).reshape(-1, 1)), axis=1)
    concat_plt = np.concatenate((mnp_y_test_plt, np.array(mnp_predicted_plt).reshape(-1, 1)), axis=1)
    sorted_lt = concat_lt[np.argsort(concat_lt[:, 0])]
    sorted_plt = concat_plt[np.argsort(concat_plt[:, 0])]
    mnp_predicted_lt = sorted_lt[:, 1]
    mnp_predicted_plt = sorted_plt[:, 1]
    mnp_y_test_lt = sorted_lt[:, 0]
    mnp_y_test_plt = sorted_plt[:, 0]

    # plot results
    plt.figure(figsize=(10, 10), dpi=200)
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(mnp_predicted_lt)), mnp_y_test_lt, label='lt true')
    plt.scatter(range(len(mnp_predicted_lt)), mnp_predicted_lt, label='lt predict')
    plt.title('LT Test Results :', model_name)
    plt.xlabel('iterations')
    plt.ylabel('LT')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(mnp_predicted_plt)), mnp_y_test_plt, label='plt true')
    plt.scatter(range(len(mnp_predicted_plt)), mnp_predicted_plt, label='plt predict')
    plt.title('PLT Test Results :', model_name)
    plt.xlabel('iterations')
    plt.ylabel('PLT')
    plt.legend()
    plt.show()

NAME = 'DSME_assembly.csv'
data = readfile(NAME)

np_data = dataframe_to_numpy(data)

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

# cast all data to type float32
np_data = np_data.astype('float32')
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
std_lt_value = np.std(temp_lt_values)
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
np_data = np.delete(np_data, invalid_data_idx, axis=0)

# take first 200 items of each ship type
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

# numpy arrays to pytorch tensors
torch_x_train = torch.tensor(np_x_train, dtype=torch.float32)
torch_x_test = torch.tensor(np_x_test, dtype=torch.float32)
torch_y_train_lt = torch.tensor(np_y_train_lt, dtype=torch.float32)
torch_y_train_plt = torch.tensor(np_y_train_plt, dtype=torch.float32)
torch_y_test_lt = torch.tensor(np_y_test_lt, dtype=torch.float32)
torch_y_test_plt = torch.tensor(np_y_test_plt, dtype=torch.float32)

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

# define Neural Network (2 hidden layer, 4 layers, hidden layer size 100)
class NetLT(nn.Module):
    def __init__(self):
        super(NetLT, self).__init__()
        self.input = nn.Linear(11, 100, bias=True)
        self.hidden1 = nn.Linear(100, 100, bias=True)
        self.hidden2 = nn.Linear(100, 100, bias=True)
        self.fc = nn.Linear(100, 1, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.input(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# initiate model
netlt = NetLT().to(device)
# define Loss function to CUDA (Mean Squared Error)
criterion = nn.MSELoss().to(device)
# define optimizer (Standard Gradient Descent)
optimizer = optim.SGD(netlt.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)

# arrays for graphing training loss
loss_history=[]
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, loaded in enumerate(trainloader_lt, start=0):
        train, targets = loaded
        train, targets = train.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = netlt(train)
        loss = criterion(outputs, targets).to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # calculate average loss every iteration
    if epoch % 10 == 9:
        print('[Epoch : %d] training loss: %.3f' % (epoch + 1, running_loss / 260))
        loss_history.append(running_loss / 260)
print("TRAINING DONE")

# plot loss graph
plt.plot(range(len(loss_history)), loss_history)
plt.title('Training Loss (MSE) History')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

# save trained model
torch.save(netlt.state_dict(), 'lt_model.pt')

# define Neural Network (3 hidden layer, 5 layers, hidden layer size 100)
class NetPLT(nn.Module):
    def __init__(self):
        super(NetPLT, self).__init__()
        self.input = nn.Linear(11, 100, bias=True)
        self.hidden1 = nn.Linear(100, 100, bias=True)
        self.hidden2 = nn.Linear(100, 100, bias=True)
        self.hidden3 = nn.Linear(100, 100, bias=True)
        self.fc = nn.Linear(100, 1, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.xavier_uniform_(self.hidden3.weight)
        nn.init.zeros_(self.hidden3.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.input(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.hidden3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# initiate model
netplt = NetPLT().to(device)
# define Loss function to CUDA (Mean Squared Error)
criterion_plt = nn.MSELoss().to(device)
# define optimizer (Standard Gradient Loss)
optimizer_plt = optim.SGD(netplt.parameters(), lr=0.0002, momentum=0.7, nesterov=True)

# arrays for graphing training loss
loss_history=[]
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, loaded in enumerate(trainloader_plt, start=0):
        train, targets = loaded
        train, targets = train.to(device), targets.to(device)
        optimizer_plt.zero_grad()
        outputs = netplt(train)
        loss = criterion_plt(outputs, targets).to(device)
        loss.backward()
        optimizer_plt.step()
        running_loss += loss.item()
    # calculate average loss every iteration
    if epoch % 10 == 9:
        print('[Epoch : %d] training loss: %.3f' %(epoch + 1, running_loss / 260))
        loss_history.append(running_loss / 260)
print("TRAINING DONE")

# plot loss graph
plt.plot(range(len(loss_history)), loss_history)
plt.title('Training Loss (MSE) History')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

# save trained model
torch.save(netplt.state_dict(), 'plt_model.pt')

# load trained model
model_lt = NetLT()
model_lt.load_state_dict(torch.load('lt_model.pt'))
model_lt.eval()
model_lt.to(device)

# Calculate RMSE, MAPE
np_predicted_lt = []
MSE = 0.
MAPE = 0.
with torch.no_grad():
    for i, loaded in enumerate(testloader_lt, start=0):
        test, targets = loaded
        test, targets = test.to(device), targets.to(device)
        predicted = model_lt(test)
        np_predicted_lt.append(predicted.item())
        MSE += (predicted.item() - targets) ** 2
        MAPE += np.abs(predicted.cpu().item() - targets.cpu()) / targets.cpu()
torch_MSE_lt = np.sqrt(MSE.cpu() / len(np_predicted_lt))
torch_MAPE_lt = 100 * MAPE / len(np_predicted_lt)


# load trained model
model_plt = NetPLT()
model_plt.load_state_dict(torch.load('plt_model.pt'))
model_plt.eval()
model_plt.to(device)

# Calculate RMSE, MAPE
np_predicted_plt=[]
MSE = 0.
MAPE = 0.
with torch.no_grad():
    for i, loaded in enumerate(testloader_plt, start=0):
        test, targets = loaded
        test, targets = test.to(device), targets.to(device)
        predicted = model_plt(test)
        np_predicted_plt.append(predicted.item())
        MSE += (predicted.item() - targets) ** 2
        MAPE += np.abs(predicted.cpu().item() - targets.cpu()) / targets.cpu()
torch_MSE_plt = np.sqrt(MSE.cpu() / len(np_predicted_plt))
torch_MAPE_plt = 100 * MAPE / len(np_predicted_plt)

print("DNN Regression")
print("____________________________________")
print("LT")
print("LT RMSE accuracy = %.3f" % torch_MSE_lt.item())
print("LT MAPE accuracy = %.3f" % torch_MAPE_lt.item())
print("___________________________________")
print("PLT")
print("PLT RMSE accuracy = %.3f" % torch_MSE_plt.item())
print("PLT MAPE accuracy = %.3f" % torch_MAPE_plt.item())

plot(model_name='DNN', mnp_predicted_lt=np_predicted_lt, mnp_predicted_plt=np_predicted_plt,
     mnp_y_test_plt=np_y_test_plt, mnp_y_test_lt=np_y_test_lt)

# LINEAR model
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
    RMSE_lt = np.sqrt(loss_method_lt(torch_predict_lt.cpu(), torch_y_test_lt.cpu()))
np_predict_lt = torch_predict_lt.cpu().numpy()
MAPE_lt = met.mean_absolute_percentage_error(np_y_test_lt, np_predict_lt)
with torch.no_grad():
    torch_predict_plt = plt_model(torch_x_test)
    RMSE_plt = np.sqrt(loss_method_plt(torch_predict_plt.cpu(), torch_y_test_plt.cpu()))
np_predict_plt = torch_predict_plt.cpu().numpy()
MAPE_plt = met.mean_absolute_percentage_error(np_y_test_plt, np_predict_plt)

plot(model_name='Linear', mnp_predicted_lt=np_predict_lt, mnp_predicted_plt=np_predict_plt,
     mnp_y_test_plt=np_y_test_plt, mnp_y_test_lt=np_y_test_lt)

print("Linear Regression")
print("____________________________________")
print("LT")
print("LT RMSE accuracy = %.3f" % RMSE_lt)
print("LT MAPE accuracy = %.3f" % MAPE_lt)
print("___________________________________")
print("PLT")
print("PLT RMSE accuracy = %.3f" % RMSE_plt)
print("PLT MAPE accuracy = %.3f" % MAPE_plt)

print('Linear model weights for LT : ', lt_model.weight)
print('Linear model weights for PLT : ', plt_model.weight)

# Define Model
mlp_lt = sk.neural_network.MLPRegressor(
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

mlp_plt = sk.neural_network.MLPRegressor(
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
print("TRAINING DONE")

# test model
np_predict_lt = mlp_lt.predict(np_x_test)
np_predict_plt = mlp_plt.predict(np_x_test)

# calculate loss
RMSE_lt = met.mean_squared_error(np_y_test_lt, np_predict_lt, squared=False)
RMSE_plt = met.mean_squared_error(np_y_test_plt, np_predict_plt, squared=False)
MAPE_lt = met.mean_absolute_percentage_error(np_y_test_lt, np_predict_lt)
MAPE_plt = met.mean_absolute_percentage_error(np_y_test_plt, np_predict_plt)

# print results
print("MLP Regression")
print("____________________________________")
print("LT")
print("LT RMSE accuracy = %.3f" % RMSE_lt)
print("LT MAPE accuracy = %.3f" % MAPE_lt)
print("___________________________________")
print("PLT")
print("PLT RMSE accuracy = %.3f" % RMSE_plt)
print("PLT MAPE accuracy = %.3f" % MAPE_plt)

#plot
plot(model_name='MLP', mnp_predicted_lt=np_predict_lt, mnp_predicted_plt=np_predict_plt,
     mnp_y_test_plt=np_y_test_plt, mnp_y_test_lt=np_y_test_lt)

