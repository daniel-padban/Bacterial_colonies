import torch
from adv_class_dataset import AdvClassBacterialDataset
from pandas import read_csv
from json import load

model_pth = 'models/adv_model--2025-01-31 16_56.pth'
model:torch.nn.Module = torch.load(model_pth)
model.eval()

species_samples = read_csv('bacteria_species.csv')
with open('species_info.json') as si:
    info_dict = load(si)

dataset = AdvClassBacterialDataset(species_samples,info_dict)
dummy_input = AdvClassBacterialDataset().__getitem__(0)[0]

torch.onnx.export(model,
                dummy_input,
                f'{model_pth}.onnx',
                input_names=['input'],
                output_names=['output'],
                dynamo=True,
                training=False)