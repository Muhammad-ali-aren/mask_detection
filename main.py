import time
import torch
import torch.nn as nn
import torchvision.transforms as T
from mask_dataset import face_mask_dataset
from torch.utils.data import DataLoader
from faceNet import FaceNet
from train_test import train_model
from plot import plot_peformance_curves
import os
import yaml

with open('config.yaml','r') as file:
    config = yaml.safe_load(file)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

torch.manual_seed(42)
EPOCHS = 5
LR = 0.001
Batch_size = 16

test_transform = T.Compose([
    T.ToTensor(),
    T.Resize(size=(200,200))
])
test_dataset = face_mask_dataset(config['dataset_path'],dataset_type='Test',transforms=test_transform)
test_loader = DataLoader(test_dataset,batch_size=len(test_dataset.data))

train_transform = T.Compose([
    T.ToTensor(),
    T.Resize(size=(200,200)),
    T.RandomHorizontalFlip(p=0.5),
])
Batch_size=16
train_dataset = face_mask_dataset(config['dataset_path'],dataset_type='Train',transforms=train_transform)
train_loader = DataLoader(train_dataset,batch_size=Batch_size,shuffle=True,drop_last=True)

model = FaceNet(in_channels=3,hidden_size=16,output_shape=1)
loss_fun = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=LR)

if __name__ == "__main__":
    start_time = time.time()
    model_results,model = train_model(model,train_loader,test_loader,optimizer,loss_fun,EPOCHS,device)
    total_time = time.time() - start_time

    print(f' Total Training time {total_time:.2f} seconds')

    model_path = os.path.join(config['dataset_path'],'faceNet.pt')
    torch.save(model.state_dict(),model_path)
    print('model state saved successfully..')
    print(' ')
    print('ploting accuracy and loss curves')
    plot_peformance_curves(model_results,EPOCHS)


