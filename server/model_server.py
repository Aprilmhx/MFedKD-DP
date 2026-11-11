import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import roc_auc_score
from cnn5 import cnn5

SUPPORTED_ARCHS = ("cnn5", "resnet18")
class Net(nn.Module):
    """Single model, selectable between CNN5 and ResNet18."""
 
    def __init__(self, num_classes: int, arch: str = "cnn5") -> None:
        super(Net, self).__init__()
        arch = arch.lower()
        if arch not in SUPPORTED_ARCHS:
            raise ValueError(f"Unknown arch '{arch}', choose from {SUPPORTED_ARCHS}")
        self.arch = arch
        if arch == "cnn5":
            self.model = cnn5(num_classes=8)
        else:  # resnet18
            self.model = models.resnet18(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 2)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)      

def get_score(hits, counts, pflag=False):
    acc = (hits[0]+hits[1]+hits[2]+hits[3]+hits[4]+hits[5]+hits[6]+hits[7])/(counts[0]+counts[1]+counts[2]+counts[3]+counts[4]+counts[5]+counts[6]+counts[7])

    if pflag:
        print("*************Metrics******************")               
        print("Melanoma: {}, Melanocytic nevus: {}, Basal cell carcinoma: {}, Actinic keratosis: {}, Benign keratosis: {}, Dermatofibroma: {}, Vascular lesion: {}, Squamous cell carcinoma: {}".format(hits[0]/counts[0], hits[1]/counts[1], hits[2]/counts[2], hits[3]/counts[3], hits[4]/counts[4], hits[5]/counts[5], hits[6]/counts[6], hits[7]/counts[7]))
        print("Accuracy: {}".format(acc))        
    return acc    
    
def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    print("model 1")
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    
    for epoch in range(epochs):
        losses = []
        class_hits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #
        class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7] #
        ##auc_pred = np.array([])
        ##auc_label = np.array([])
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images) 
            y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()
            ##if auc_pred.size == 0:
            ##    auc_pred = y_pred
            ##else:
            ##    auc_pred = np.vstack((auc_pred, y_pred))
            y_label = labels.cpu().numpy()
            ##auc_label = np.append(auc_label, y_label)
           
            loss = criterion(outputs, labels)   
            losses.append(loss.item())
            _, preds = torch.max(outputs.data, 1)
            
            for idx in range(preds.shape[0]):
                class_counts[labels[idx].item()] += 1.0  
                if preds[idx].item() == labels[idx].item():
                    class_hits[labels[idx].item()] += 1.0  

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()                   

        ##train_auc = roc_auc_score(auc_label, auc_pred, multi_class="ovr", average="macro")       
        train_acc = get_score(class_hits, class_counts, True)      
        print('Epoch [%d/%d] - loss %.4f -accuracy %.4f' % (epoch+1, epochs, np.mean(losses), train_acc))        

def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    print("model 2")
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    net.to(device)
    with torch.no_grad():
        losses = []
        class_hits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #
        class_counts = [0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7, 0.0+1e-7] #        
        correct = 0
        test_sum = 0  
        ##auc_pred = np.array([])
        ##auc_label = np.array([])        
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  #|torch.Size([1])
            outputs = net(images)             #torch.Size([1, 3])
            
            y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()
            ##if auc_pred.size == 0:
            ##    auc_pred = y_pred
            ##else:
            ##    auc_pred = np.vstack((auc_pred, y_pred))
            y_label = labels.cpu().numpy()
            ##auc_label = np.append(auc_label, y_label)            
            
            loss = criterion(outputs, labels).item()
            losses.append(loss)            
            _, preds = torch.max(outputs.data, 1)
            
            for idx in range(preds.shape[0]):
                class_counts[labels[idx].item()] += 1.0  
                if preds[idx].item() == labels[idx].item():
                    class_hits[labels[idx].item()] += 1.0  
                    
            correct += (preds == labels).sum().item()
            test_sum+= labels.shape[0]
        val_acc = get_score(class_hits, class_counts, True) 
        ##val_auc = roc_auc_score(auc_label, auc_pred, multi_class="ovr", average="macro")    
                
    print("loss %.4f - accuracy %.4f" % (np.mean(losses), val_acc))     
            
    return loss, val_acc