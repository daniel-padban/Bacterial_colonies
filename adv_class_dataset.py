import json
from numpy import r_
import torch
import string
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as v2

class AdvClassBacterialDataset(Dataset):
    '''
        edwde

        :param all_data: Requires a list with tuples (img, label), will override loading by paths
        '''

    def __init__(self,species_samples:pd.DataFrame,species_info:dict,r_state:int=None,transform:v2.Transform=None):
        super().__init__()

        self.species_samples = species_samples
        self.species_info = species_info
        self.r_state = r_state
        self.transform = transform
        self.OH_info = self._prep_info()
        self._shuffle_data()

    def _prep_info(self):
        info_df = pd.DataFrame.from_dict(self.species_info,orient='index')
        info_df.index.name = 'species_id'
        info_df.reset_index(inplace=True)
        OH_encoded_df = pd.get_dummies(info_df,columns=['gram','culturing','agar'],prefix=['gram','culturing','agar'])
        OH_encoded_df.set_index('species_id',inplace=True)
        return OH_encoded_df
    
    def _shuffle_data(self):
        if self.r_state:
            self.species_samples = self.species_samples.sample(frac=1,random_state=self.r_state)
        else:
            self.species_samples = self.species_samples.sample(frac=1)

    def __len__(self):
        return len(self.species_samples)
    
    def __getitem__(self, idx):
        img_pth = self.species_samples['image_name'].iloc[idx]
        species_id:str = self.species_samples['label_name'].iloc[idx]
        species_meta = self.OH_info.loc[species_id]

        x = Image.open(f'bac_images/{img_pth}')
        if self.transform:
            x = self.transform(x)
        y = int(species_id.strip(string.ascii_letters))-1
        
        e = torch.tensor(species_meta)
        
        return x, y, e # x = image ------ y = species digit ----- e = embedding (species info)


if __name__ == '__main__':
    with open('species_info.json') as si:
        info_dict  = json.load(si)
    species_df = pd.read_csv('bacteria_species.csv')

    dataset = AdvClassBacterialDataset(species_df,info_dict,100,None,)
    x,y,e = dataset.__getitem__(0)
    dataset.__len__()
    x.show()
    print(e.size(0))