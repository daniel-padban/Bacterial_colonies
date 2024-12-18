import json
import torch
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torch.utils.data import DataLoader
import datetime

from model import BacterialClassiferCNN
from trainer import ClassBacterialTrainer
from class_dataset import ClassBacterialDataset

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

out_dim = config['out_dim']

model = BacterialClassiferCNN(
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
    out_dim
)
model.to(dev)

batch_size = config['batch_size']
train_transform = v2.Compose([
    v2.RandAugment(3,5),
    v2.PILToTensor(),
    v2.ToDtype(torch.float),
    v2.Resize([HW,HW]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_idx_slice = slice(0,215)
img_dir = 'bac_images'
labels_path = 'bacteria_species.csv'
train_dataset = ClassBacterialDataset(
    img_dir,
    labels_path,
    'image_name',
    'label_name',
    train_idx_slice,
    r_state=random_state,
    transform=train_transform
)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)

test_transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float),
    v2.Resize([HW,HW],),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_idx_slice = slice(215,None)
test_dataset = ClassBacterialDataset(
    img_dir,
    labels_path,
    'image_name',
    'label_name',
    test_idx_slice,
    r_state=random_state,
    transform=test_transform
)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True,)

lr = config['lr']
w_decay = config['w_decay']
optimizer = AdamW(params=model.parameters(),lr=lr,weight_decay=w_decay)

epochs = config['epochs']

trainer = ClassBacterialTrainer(model,train_loader,test_loader,optimizer,device=dev)

trainer.full_epoch_loop(epochs)

current_datetime = datetime.datetime.now().strptime('%Y-%m-%d %H:%M')
torch.save(model.state_dict(),f'models/model--{current_datetime}')