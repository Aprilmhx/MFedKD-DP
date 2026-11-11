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
        label_new = label-1

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
    trainloaders, valloaders, testloaders=prepare_dataset("isic2019", 4, "FL_divide_4clients.json", 1)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True)
    #
    for image, target in valloaders[0]:
        print("test")

    #for i in range(len(train_img_ids)):
        # print(i)
    #    image, label= train_dataset[i]

    #    image = np.transpose(image, (1,2,0))

    #    print(image.min())
    #    print(image.max())
        #print(mask.min())
        #print(mask.max())
        ##visualize(image=image)  #(32, 32, 3) (32, 32, 1)