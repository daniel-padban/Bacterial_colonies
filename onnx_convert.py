import torch
import torchvision.transforms.v2 as v2
from pandas import read_csv
from json import load
from class_dataset import AdvClassBacterialDataset

model_pth = 'models/adv_model--2025-01-31 18_21.pth'
model:torch.nn.Module = torch.load(model_pth,weights_only=False)
model.eval()

species_samples = read_csv('bacteria_species.csv')
with open('data/species_info.json') as si:
    info_dict = load(si)

HW = 100
train_transform = v2.Compose([
    v2.RandAugment(3,5),
    v2.PILToTensor(),
    v2.ToDtype(torch.float),
    v2.Resize([HW,HW]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = AdvClassBacterialDataset(species_samples,info_dict,transform=train_transform)
dummy_tup = dataset.__getitem__(0)
dummy_im = dummy_tup[0].unsqueeze(0)
dummy_e = dummy_tup[2].unsqueeze(0)

torch.onnx.export(model,
                (dummy_im,dummy_e),
                f'{model_pth}.onnx',
                input_names=['input'],
                output_names=['output'],
                dynamo=True,
                training=False)