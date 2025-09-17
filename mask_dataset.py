import os 
from torch.utils.data import Dataset
import cv2 as cv
from preprocess import prepare_data

class face_mask_dataset(Dataset):
    def __init__(self,dataset_path,dataset_type='Train', transforms=False):
        super(face_mask_dataset,self).__init__()
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.dataset_type = dataset_type
        self.dataTemp = prepare_data(self.dataset_path,self.dataset_type)
        self.data = self.dataTemp[0]
        self.labels = self.dataTemp[1]
    def load_image(self,index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        label = self.labels[index]
        img_path = os.path.join(self.dataset_path,self.dataset_type)
        if label == 0:
            img_path = os.path.join(img_path,'Mask')
        else:
            img_path = os.path.join(img_path,'Non Mask')
        img_read = cv.imread(os.path.join(img_path, self.data[index]))
        img = cv.cvtColor(img_read, cv.COLOR_BGR2RGB)
        if self.transforms:
            return self.transforms(img),label
        else:
            return img,label