import numpy as np
from agents.dynamics.dynamics_model import *
from datetime import datetime
import os
import torch as th
import torch.nn as nn
import torch.optim as optim

def root_mean_squared_error(y_true, y_pred):
    return th.sqrt(th.mean((y_true - y_pred) ** 2))


class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0,checkpoints_path="dynamicsModelCheckpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.checkpoints_path = checkpoints_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        th.save(model.state_dict(), self.checkpoints_path)
        self.val_loss_min = val_loss

class fwDiffNNModel(DynamicsModel):
    def __init__(
            self,
            state_size,
            action_size,
            model_rw=False,
            data_size=50000,
            hidden_units=[32,32],
            lr=5e-5,
            ):

        super(fwDiffNNModel, self).__init__(state_size, action_size, pred_rew=model_rw, data_buffer_size=data_size)
        logs = "logs/fwNN_" + datetime.now().strftime("%Y%m%d-%H%M%S")

        self.prev_best = np.inf
        layers = []
        input_size = self._input_size
        for units in hidden_units:
            layers.append(nn.Linear(input_size, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            input_size = units
        layers.append(nn.Linear(input_size, self._output_size))
        self.device = ('cuda:3' if th.cuda.is_available() else 'cpu')
        self.model = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.MSELoss()  # Mean Square Error
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self._decay =False,# 0.99
        

    def train(self, num_epochs=10):
        hist = {"tloss":[], "vloss":[]}
        # if self._decay:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr']=param_group['lr']*self._decay


        inp, out, val_inp, val_out = self.get_data(is_delta=True) #Gets batch data
        inp = inp.clone().detach().to(th.float32)
        out = out.clone().detach().to(th.float32)
        val_inp = val_inp.clone().detach().to(th.float32)

        val_out = val_out.clone().detach().to(th.float32)

        train_dataset = th.utils.data.TensorDataset(inp, out)#creates a pytorch dataset
        val_dataset = th.utils.data.TensorDataset(val_inp, val_out)#creates a pytorch dataset

        print("train_dataset",len(train_dataset))

        train_loader = th.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True )#creates a pytorch data loader
        val_loader = th.utils.data.DataLoader(val_dataset, batch_size=32)#creates a pytorch data loader
        
        early_stopping = EarlyStopping(patience=100, verbose=True) #Creates an instance of the Early stopping class
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for input, output in train_loader:
                if input.shape[0]==1: continue
                input = input.to(self.device)
                output = output.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(input)
                loss = self.criterion(predictions, output)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation
            val_loss = 0.0
            self.model.eval()
            with th.no_grad():
                for val_inp, val_out in val_loader:
                   
                    val_inp, val_out = val_inp.to(self.device), val_out.to(self.device)
                    val_predictions = self.model(val_inp)
                    val_loss += self.criterion(val_predictions, val_out).item()

            # Log training and validation loss
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            hist["vloss"].append(val_loss)
            hist["tloss"].append(train_loss)


            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        return hist
        
        

    def predict(self, data, num_samples=1):
        data = th.tensor(data,dtype=th.float32).to(self.device)
        self.model.eval()
        with th.no_grad():
            pred = self.model(data)
            next_state = self.get_new_states(data, pred)
           
            return next_state.unsqueeze(0).to('cpu')

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        th.save(self.model.state_dict(), os.path.join(save_dir, f"model.pth"))

    def load_model(self, save_dir):
        model_path = os.path.join(save_dir, "model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(th.load(model_path))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"No model found at {model_path}")

    def set_model_to(self, model_to_copy):
        self.model.load_state_dict(model_to_copy.model.state_dict())