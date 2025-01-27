import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ClassBacterialDataset(Dataset):
    '''
        edwde

        :param all_data: Requires a list with tuples (img, label), will override loading by paths
        '''
    def __init__(self, img_dir:str, csv_labels_path:str,img_path_col:str,img_label_col:str,idx_slice:slice, shuffle:bool=True,r_state:int=42,all_data:list = None, transform=None,):
        super().__init__() 
        
        self.transform = transform
        self.img_dir = img_dir
        self.labels = pd.read_csv(csv_labels_path)
        self.img_path_col = img_path_col
        self.img_label_col = img_label_col
        self.idx_slice = idx_slice #which indices to use
        if all_data is not None:
            self.img_label_data = self._pick_data(all_data,shuffle,r_state)
        else:
            self.img_label_data, self.all_data = self._load_all_data(shuffle,r_state)
        
    def _load_all_data(self,shuffle:bool=True,r_state:int=None):
        '''
        returns a list of tuples: (img, label) from source (directory)
        '''
        img_list = []
        label_list = []
        for img_path in self.labels['image_name']:
            img = Image.open(self.img_dir + '/' + img_path)
            label_df = self.labels.loc[self.labels[self.img_path_col] == img_path]
            label = label_df[self.img_label_col].values[0]
            img_list.append(img)
            label_list.append(label)
        all_data = list(zip(img_list,label_list))

        if shuffle and r_state:
            shuffled_data = pd.DataFrame(all_data).sample(frac=1,random_state=r_state)
            data_to_use = shuffled_data[self.idx_slice]
        else:
            data_to_use = all_data[self.idx_slice]
        return data_to_use, all_data
    
    def _pick_data(self, all_data:list,shuffle:bool=True,r_state:int=None):
        '''
        Selects indices of already loaded data
        '''
        if shuffle and r_state:
            shuffled_data = pd.DataFrame(all_data).sample(frac=1,random_state=r_state)
            data_to_use = shuffled_data.iloc(0)[self.idx_slice]
        else:
            data_to_use = all_data[self.idx_slice]
        
        return data_to_use

    def __len__(self):
        data_len = len(self.img_label_data)
        return data_len

    def __getitem__(self, idx):
        X = self.img_label_data.iloc()[idx][0]
        y = self.img_label_data.iloc()[idx][1]
        y = int(y[2:]) - 1
        if self.transform is not None:
            X = self.transform(X)

        return X, y
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    idx_slice = slice(0,100)
    bac_set = ClassBacterialDataset(
        img_dir='bac_images',
        csv_labels_path='bacteria_species.csv',
        img_path_col='image_name',
        img_label_col='label_name',
        idx_slice=idx_slice,
        transform=None,
    )

    test_img,test_label = bac_set.__getitem__(0)
    print(test_label)
    plt.imshow(np.asarray(test_img))
    plt.show()
