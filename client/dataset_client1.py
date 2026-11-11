import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import torch
from glob import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        label = int((self.img_ids[idx]).split(",")[1])   
        
        if label == 2:
            label_new = 0
        else:
            label_new = 1

        img_id = (self.img_ids[idx]).split(",")[0]        
        img = cv2.imdecode(np.fromfile(os.path.join(self.img_dir, str(label), img_id), dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
            # print(img.shape)
            # print(mask.shape)
            # print("ending")

        img = img.astype('float32') ##/ 255
        img = img.transpose(2, 0, 1)


        return torch.from_numpy(img).to(torch.float32), label_new  
    
if __name__ == "__main__":
    trainloader, validationloader = prepare_dataset_client1("isic2019", 4, "FL_divide_4clients.json", 16)