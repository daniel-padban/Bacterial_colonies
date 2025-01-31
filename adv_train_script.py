import json
import torch
import wandb
from torch.utils.data import Subset
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pandas as pd
import datetime

from adv_model import AdvBacterialClassiferCNN
from adv_trainer import AdvClassBacterialTrainer
from adv_class_dataset import AdvClassBacterialDataset

current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
print(current_datetime)

random_state = 100
torch.manual_seed(random_state)
torch.mps.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
dev = torch.device(('cuda' if torch.cuda.is_available() 
        else 
            'mps' if torch.backends.mps.is_available() 
        else 
           'cpu'))
print(f'Device: {dev}')

with open('config.json','r') as f:
    config = json.load(f)

print(f'Config:\n\n{config}')


# -------------------- Model --------------------
HW = config['HW']
conv1 = config['conv1']
k1 = config['k1']
s1 = config['s1']
K_pool1 =  config['K_pool1']
conv2 = config['conv2']
k2 = config['k2']
s2 = config['s2']
K_pool2 = config['K_pool2']
conv3= config['conv3']
k3 = config['k3']
s3 = config['s3']
K_pool3 = config['K_pool3']
fc1 = config['fc1']
fce = config['fce']

out_dim = config['out_dim']

model = AdvBacterialClassiferCNN(
    HW,
    conv1,
    k1,
    s1,
    K_pool1,
    conv2,
    k2,
    s2,
    K_pool2,
    conv3,
    k3,
    s3,
    K_pool3,
    fc1,
    fce,
    9,
    out_dim
)
model.to(dev)
#-------------------- General Dataset --------------------
species_samples = pd.read_csv('bacteria_species.csv')
with open('species_info.json') as si:
    info_dict = json.load(si)

main_dataset = AdvClassBacterialDataset(species_samples=species_samples,species_info=info_dict,r_state=random_state,transform=None)


#-------------------- Train Dataset --------------------
train_transform = v2.Compose([
    v2.RandAugment(3,5),
    v2.PILToTensor(),
    v2.ToDtype(torch.float),
    v2.Resize([HW,HW]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = config['batch_size']
train_idx_slice = slice(0,215)
train_indices = range(*train_idx_slice.indices(len(main_dataset)))
train_dataset = Subset(main_dataset,train_indices)
train_dataset.dataset.transform = train_transform
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=False)


#-------------------- Test Dataset --------------------
test_transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float),
    v2.Resize([HW,HW]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_idx_slice = slice(215,None)
test_indices = range(*test_idx_slice.indices(len(main_dataset)))
test_dataset = Subset(main_dataset,test_indices)
test_dataset.dataset.transform = test_transform
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=False,)


#-------------------- Training --------------------
lr = config['lr']
w_decay = config['w_decay']
optimizer = AdamW(params=model.parameters(),lr=lr,weight_decay=w_decay)
epochs = config['epochs']

trainer = AdvClassBacterialTrainer(model,train_loader,test_loader,optimizer,device=dev)
train_results, test_results  = trainer.full_epoch_loop(epochs)
model_pth = f'models/adv_model--{current_datetime}'
torch.save(model.state_dict(),model_pth)

run = wandb.init('DP-Team', 'Bacterial',config=config,name=f'adv_model--{current_datetime}')

run.save(model_pth)
for e,result_tup in enumerate(train_results):
    log_dict = {
        'Epoch':e+1,
        'Train_CE':result_tup[0],
        'Train_acc':result_tup[1],
        'Train_F1':result_tup[2],
    }
    run.log(log_dict)

for e,result_tup in enumerate(train_results):
    log_dict = {
        'Epoch':e+1,
        'Train_CE':result_tup[0],
        'Train_acc':result_tup[1],
        'Train_F1':result_tup[2],
    }
    run.log(log_dict)
run.finish(0)
