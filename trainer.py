import numpy as np
from sympy import Mul
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
import tqdm

class ClassBacterialTrainer():
    def __init__(self, model:nn.Module, train_dataloader:DataLoader, test_dataloader:DataLoader, optimizer:torch.optim.Optimizer,device=torch.device('cpu')):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.device=device
        self.model.to(device)
        
        self.loss = nn.CrossEntropyLoss()

    def train_loop(self, report_freq = 10):
        self.model.train()
        batch_COs = []
        batch_accs = []
        batch_f1s = []
        for (X, y) in tqdm.tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            X:torch.Tensor = X.to(device=self.device)
            y:torch.Tensor = y.to(device=self.device)

            preds:torch.Tensor = self.model(X)
            probs = preds.softmax(1)
            loss:torch.Tensor = self.loss(preds,y)
            loss.backward()
            self.optimizer.step()
            
            #report
            batch_CO = loss.item()
            batch_COs.append(batch_CO)
            
            most_probable = probs.argmax(1)
            batch_acc = ((most_probable==y).sum())/y.size(0) #correct / samples
            batch_accs.append(batch_acc.item())

            batch_f1 = multiclass_f1_score(probs,y,num_classes=24)
            batch_f1s.append(batch_f1.item())

        mean_epoch_CO = np.array(batch_COs).mean()
        mean_epoch_acc = np.array(batch_accs).mean()
        mean_epoch_f1 = np.array(batch_f1s).mean()
        
        return mean_epoch_CO, mean_epoch_acc, mean_epoch_f1
    
    def test_loop(self):
        self.model.eval()
        batch_COs = []
        batch_accs = []
        batch_f1s = []
        with torch.no_grad():
            for (X, y) in tqdm.tqdm(self.train_dataloader):
                X:torch.Tensor = X.to(device=self.device)
                y:torch.Tensor = y.to(device=self.device)

                preds:torch.Tensor = self.model(X)
                probs = preds.softmax(1)
                loss:torch.Tensor = self.loss(preds,y)
                
                #report
                batch_CO = loss.item()
                batch_COs.append(batch_CO)
                
                most_probable = probs.argmax(1)
                batch_acc = ((most_probable==y).sum())/y.size(0) #correct / samples
                batch_accs.append(batch_acc.item())

                batch_f1 = multiclass_f1_score(probs,y,num_classes=24)
                batch_f1s.append(batch_f1.item())

        mean_epoch_CO = np.array(batch_COs).mean()
        mean_epoch_acc = np.array(batch_accs).mean()
        mean_epoch_f1 = np.array(batch_f1s).mean()
        
        return mean_epoch_CO, mean_epoch_acc, mean_epoch_f1
    
    def full_epoch_loop(self,epochs):
        train_results = []
        test_results = []
        for epoch in range(epochs):
            print(f'Train Epoch: {epoch+1}')
            train_epoch_CO, train_epoch_acc, train_epoch_f1 = self.train_loop()
            print(f'Train CO: {train_epoch_CO}')
            print(f'Train Accuracy: {train_epoch_acc}')
            print(f'Train F1: {train_epoch_f1}')
            train_results.append((train_epoch_CO,train_epoch_acc,train_epoch_f1))
            
            print(f'    Test Epoch: {epoch+1}')
            test_epoch_CO, test_epoch_acc, test_epoch_f1 = self.test_loop()
            print(f'    Test CO: {test_epoch_CO}')
            print(f'    Test Accuracy: {test_epoch_acc}')
            print(f'    Test F1: {test_epoch_f1}')
            test_results.append((test_epoch_CO,test_epoch_acc,test_epoch_f1))
        
        return train_results, test_results
    