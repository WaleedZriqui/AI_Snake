import torch
import torch.nn as nn
import torch.optim as optim  # optimizer is used to decrease the rates of error during training the neural networks.
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module): # inheretace class torch.nn.Module which is a base class for all neural network module.
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # call class Module constructor
        self.linear1 = nn.Linear(input_size, hidden_size) # create a single layer of input_size as inputs and hidden_size as output
        self.linear2 = nn.Linear(hidden_size, output_size) # so on

    def forward(self, x): # x will be the input
        x = F.relu(self.linear1(x)) # applay activation function 
        x = self.linear2(x) # enter x to second layer 
        return x 

    def save(self, file_name='model.pth'): # if we have a new high score 
        model_folder_path = './model' # new folder in directory
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma): # lr : learning rate 
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # make optimizer 
        self.criterion = nn.MSELoss() # loss function [loss = (Qnew - Q)^2]

    def train_step(self, state, action, reward, next_state, done): # all coming input may be single value, or tuple oe list so we should convert it to tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1: # we only have one dimension
            # (1, x) # make all in one dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # convert single value to tuple 

        # Start Bellman algorithm
        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        
        
        self.optimizer.zero_grad() # to empty the gradient
        loss = self.criterion(target, pred) # pred = prediction
        loss.backward()

        self.optimizer.step()



