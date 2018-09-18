# research-pytorch
## Load Data
### Numpy-Torch
```
a = np.array([[1,2], [3,4]])
b = torch.from_numpy(a)      # convert numpy array to torch tensor
c = b.numpy()                # convert torch tensor to numpy array
```

## Models
https://github.com/MorvanZhou/PyTorch-Tutorial

### Linear Regression Model
```
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__();
        self.linear = torch.nn.Linear(input_size, output_size)  
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(input_size, output_size);

# Loss function
loss_func = torch.nn.MSELoss();
```
### Logistic Regression
```
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size, num_classes)

# Loss function
loss_func = torch.nn.CrossEntropyLoss();
```
### Neural Network Model (1 hidden layer)
```
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__();
        self.fully_connected1 = torch.nn.Linear(input_size, hidden_size);
        self.relu = torch.nn.ReLU();
        self.fully_connected2 = torch.nn.Linear(hidden_size, num_classes);
    
    def forward(self, x):
        out = self.fully_connected1(x);
        out = self.relu(out);
        out = self.fully_connected2(out);
        return out

net = Net(input_size, hidden_size, num_classes)

# Loss function
loss_func = torch.nn.CrossEntropyLoss();
```

### Convolutional Neural Network Model (2 conv layer)
```
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__();
        # in_channel: input height
        # out_channel: n filter
        # kernel size: filter size
        # stride: filter movement/step
        # padding:
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5, 
                stride=1,
                padding=2, 
            ), # output shape (16, 28, 28)
            nn.ReLU(), # activation
            nn.MaxPool2d(kernel_size=2), # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=5, 
                stride=1, 
                padding=2
            ),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size =2),   # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization

cnn = CNN()
```
### Recurrent Neural Network Model (Many-to-One)
```
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial states 
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))  
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  
        return out

rnn = RNN(input_size, hidden_size, num_layers, num_classes)
```

## Optimizer
```
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) ;
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate);
```

## GPU
```
model.cuda() 
```

## Execution
```
# Forward + Backward + Optimize
optimizer.zero_grad()
outputs = model(x)
loss = loss_func(outputs, y)
loss.backward()
optimizer.step()
```

## Save and Load
### Save and load the entire model.
```
torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')
```

### Save and load only the model parameters(recommended).
```
torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('params.pkl'))
```

### Problem
- unable to converge for regression

