from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn5 import cnn5, cnn4
from resnet import resnet10
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import pickle

class NetS(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(NetS, self).__init__()
        self.model_s = cnn5(num_classes = 2)
        #self.model_s = resnet10()
        #self.input_features_s = self.model_s.fc.in_features
        #self.model_s.fc = nn.Linear(self.input_features_s, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_s(x)

        return x  
    
class NetT(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(NetT, self).__init__()
        self.model_t = models.resnet18(pretrained=True)
        self.input_features_t = self.model_t.fc.in_features
        self.model_t.fc = nn.Linear(self.input_features_t, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model_t(x)

        return x   

def get_score(hits, counts, pflag=False):
    acc = (hits[0]+hits[1])/(counts[0]+counts[1])

    if pflag:
        print("*************Metrics******************")               
        print("Healthy: {}, Unhealthy: {}".format(hits[0]/counts[0], hits[1]/counts[1]))
        ##print("Accuracy: {}".format(acc))        
    return acc             

########################## LDP ##########################
def snapshot_params(net):
    return {n: p.detach().clone() for n, p in net.named_parameters()}

@torch.no_grad()
def probe_norms(parts, tags=("S", "T")):
    """Logging only, used to CHOOSE the fixed hyperparameters C_S and C_T in a calibration run.
    Prints the L2 norm of each model's uploaded update u^S_{i,r} and u^T_{i,r} separately
    (each one is clipped to its own threshold in privatize_separate).
    Do NOT feed the returned values back into C during a private run: a C computed from
    private data is itself a privacy leak not covered by the Gaussian mechanism."""
    norms = {}
    for (net, keys, start), tag in zip(parts, tags):
        params = dict(net.named_parameters())
        sq = sum(float(((params[k].detach() - start[k]).double() ** 2).sum()) for k in keys)
        dim = sum(params[k].numel() for k in keys)
        norms[tag] = sq ** 0.5
        print(f"[probe-{tag}] L2={norms[tag]:.4f} dim={dim} "
              f"RMS/coord={norms[tag] / max(dim, 1) ** 0.5:.3e}")
    return norms

@torch.no_grad()
def privatize_separate(parts, Cs, zs, tags=("S", "T")):
    """ClientUpload of Alg. 3, student and teacher privatized SEPARATELY.
 
    parts: list of (net, upload_keys, start_snapshot), one entry per model (student, teacher).
    Cs, zs: per-model clipping thresholds (C_S, C_T) and noise multipliers (z_S, z_T).
    For each model m: u^m is clipped to C_m as a whole, then N(0, (z_m*C_m)^2 I) is added.
    Privacy: both releases come from the same data, so per round
        eps_RDP(alpha) = 2*alpha/z_S^2 + 2*alpha/z_T^2   (LDP, sensitivity 2*C_m).
    Noise is drawn from a generator with a non-deterministic seed, so it is independent of
    setup_seed() and independent across clients.
    Returns an OrderedDict {param_name: noisy_update (np.ndarray)}. The local model is not modified;
    the uploaded layers are replaced by the server's denoised result next round.
    """
    noisy = OrderedDict()
    for (net, keys, start), C, z, tag in zip(parts, Cs, zs, tags):
        if not keys:
            continue
        params = dict(net.named_parameters())
        deltas = OrderedDict((k, params[k].detach() - start[k]) for k in keys)
 
        total = float(torch.sqrt(sum((d.double() ** 2).sum() for d in deltas.values())))
        scale = min(1.0, C / (total + 1e-12))           # clip(u^m, C_m)
        sigma = z * C if z > 0 else 0.0                 # z=0 & C=inf in calibration mode
        dim = sum(d.numel() for d in deltas.values())
 
        gen = None
        if sigma > 0:
            gen = torch.Generator(device=next(iter(deltas.values())).device)
            gen.seed()                                  # non-deterministic seed
 
        for k, d in deltas.items():
            nd = d * scale
            if sigma > 0:
                nd = nd + torch.randn(nd.shape, generator=gen, device=nd.device, dtype=nd.dtype) * sigma
            noisy[k] = nd.cpu().numpy()
 
        print(f"[priv-{tag}] C={C:.4f} z={z} raw_norm={total:.4f} clipped={total > C} "
              f"sigma={sigma:.4f} dim={dim} E||noise||={sigma * dim ** 0.5:.3f} "
              f"signal/noise={min(total, C) / (sigma * dim ** 0.5 + 1e-12):.4g}")
    return noisy

def online_distillation_loss(s_outputs, t_outputs, labels, T, criterion):
    soft_student_outputs = F.log_softmax(s_outputs / T, dim=1)
    soft_teacher_outputs = F.softmax(t_outputs.detach() / T, dim=1)  
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(soft_student_outputs, soft_teacher_outputs) * (T * T)
    return distillation_loss

def train(net_s, net_t, trainloader, valloader, optimizer, epochs, current_round, device, s_keys, t_keys, clip_Cs, zs, lambda_kd):       
    criterion = torch.nn.CrossEntropyLoss()
    net_s.train()
    net_s.to(device)     
    net_t.train()
    net_t.to(device)  
    torch.autograd.set_detect_anomaly(True)
    T = 3
    filename = "CLIENT1/output"
    best_score = 0
    
    start_s = snapshot_params(net_s)
    start_t = snapshot_params(net_t)    
    
    for epoch in range(epochs):
        losses_s = []
        class_hits_s = [0.0, 0.0] 
        class_counts_s = [0.0+1e-7, 0.0+1e-7] 
        conf_label_s, conf_pred_s = [], []
        
        KDs = []        
        losses_t = []
        class_hits_t = [0.0, 0.0]  
        class_counts_t = [0.0+1e-7, 0.0+1e-7] 
        conf_label_t, conf_pred_t = [], [] 
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            y_label = labels.cpu().numpy()

            outputs_s = net_s(images)
            outputs_t = net_t(images)            
            
            loss_s = 0.5*(criterion(outputs_s, labels))
            loss_t = (criterion(outputs_t, labels))
            #MFedKD-DP
            #KD_s = lambda_kd*(online_distillation_loss(outputs_s, outputs_t, labels, T, criterion))        
            #Baseline
            KD_s = 0.5*(online_distillation_loss(outputs_s, outputs_t, labels, T, criterion))  
            loss = KD_s+loss_t+loss_s
            
            optimizer.zero_grad()             
            loss.backward()
            optimizer.step()                       

            #y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()            
            losses_s.append(loss_s.item())
            losses_t.append(loss_t.item())
            KDs.append(KD_s.item())
            _, preds_s = torch.max(outputs_s.data, 1)
            _, preds_t = torch.max(outputs_t.data, 1)

            for idx in range(preds_s.shape[0]):
                class_counts_s[labels[idx].item()] += 1.0  
                conf_label_s.append(labels[idx].item())
                conf_pred_s.append(preds_s[idx].item())                  
                if preds_s[idx].item() == labels[idx].item():
                    class_hits_s[labels[idx].item()] += 1.0  
            for idx in range(preds_t.shape[0]):
                class_counts_t[labels[idx].item()] += 1.0  
                conf_label_t.append(labels[idx].item())
                conf_pred_t.append(preds_t[idx].item())                  
                if preds_t[idx].item() == labels[idx].item():
                    class_hits_t[labels[idx].item()] += 1.0                   

        ####################################Caculate metrics#####################################    
        train_acc_s = get_score(class_hits_s, class_counts_s, True)        
        train_conf_matrix_s = confusion_matrix(conf_label_s, conf_pred_s)
        #print("Confusion Matrix", train_conf_matrix)           
        train_f1_micro_s = f1_score(conf_label_s, conf_pred_s, average='micro')
        train_f1_weighted_s = f1_score(conf_label_s, conf_pred_s, average='weighted')
        print('Student model Epoch [%d/%d] - train loss %.4f - train accuracy %.4f - train f1_micro %.4f - train f1_weighted %.4f' % (epoch+1, epochs, np.mean(losses_s), train_acc_s, train_f1_micro_s, train_f1_weighted_s))   
        
        train_acc_t = get_score(class_hits_t, class_counts_t, True)        
        train_conf_matrix_t = confusion_matrix(conf_label_t, conf_pred_t)
        #print("Confusion Matrix", train_conf_matrix)           
        train_f1_micro_t = f1_score(conf_label_t, conf_pred_t, average='micro')
        train_f1_weighted_t = f1_score(conf_label_t, conf_pred_t, average='weighted')
        print('Teacher model Epoch [%d/%d] - train loss %.4f - KD loss %.4f - train accuracy %.4f - train f1_micro %.4f - train f1_weighted %.4f' % (epoch+1, epochs, np.mean(losses_t), np.mean(KDs), train_acc_t, train_f1_micro_t, train_f1_weighted_t))       
        ####################################Caculate metrics#####################################    

        ####################Save best client model#############################
        val_loss, val_acc, val_f1 = val(net_s, valloader, device)
        if best_score < val_acc:    #val_acc
            best_score = val_acc    #val_acc
            print("Student model: best ACC achieved......", best_score)            
            state_dict_s = {name: param for name, param in net_s.state_dict().items()}
            with open(os.path.join(filename,"model_s_test_client1"+".pkl"), 'wb') as file1:
                pickle.dump(state_dict_s, file1)      
            state_dict_t = {name: param for name, param in net_t.state_dict().items()}
            with open(os.path.join(filename,"model_t_test_client1"+".pkl"), 'wb') as file2:
                pickle.dump(state_dict_t, file2)
    
    ################ Alg. 3: ClientUpload ################
    parts = [(net_s, s_keys, start_s), (net_t, t_keys, start_t)]
    norms = probe_norms(parts)               # used by the client to calibrate C_S / C_T
    noisy_update = privatize_separate(parts, clip_Cs, zs)    
 
    # Local (non-uploaded) parts such as fc and BN running stats persist through these files.
    with open(os.path.join(filename, "round_model_s_client1.pkl"), 'wb') as f:
        pickle.dump(dict(net_s.state_dict()), f)
    with open(os.path.join(filename, "round_model_t_client1.pkl"), 'wb') as f:
        pickle.dump(dict(net_t.state_dict()), f)
 
    print(f"  → LDP noise applied to upload after epoch {epoch + 1}")
    return noisy_update, norms 

def val(net, valloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    print("model 2")
    net.to(device)     
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()    
    
    with torch.no_grad():
        losses = []
        class_hits = [0.0, 0.0] #
        class_counts = [0.0+1e-7, 0.0+1e-7] #        
        correct = 0
        val_sum = 0  
        conf_label, conf_pred = [], []
        ##auc_pred = np.array([])
        ##auc_label = np.array([])        
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)  #|torch.Size([1])
            outputs = net(images)             #torch.Size([1, 3])
            
            ##y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()
            ##y_label = labels.cpu().numpy()          
            
            loss = criterion(outputs, labels).item()
            losses.append(loss)            
            _, preds = torch.max(outputs.data, 1)
            
            for idx in range(preds.shape[0]):
                class_counts[labels[idx].item()] += 1.0  
                conf_label.append(labels[idx].item())
                conf_pred.append(preds[idx].item())                  
                if preds[idx].item() == labels[idx].item():
                    class_hits[labels[idx].item()] += 1.0  
                    
            correct += (preds == labels).sum().item()
            val_sum+= labels.shape[0]

        ####################################Caculate metrics#####################################      
        val_acc = get_score(class_hits, class_counts, True)    
        val_conf_matrix = confusion_matrix(conf_label, conf_pred)
        #print("Confusion Matrix", test_conf_matrix)           
        val_f1_micro = f1_score(conf_label, conf_pred, average='micro')
        val_f1_weighted = f1_score(conf_label, conf_pred, average='weighted')
        print('val loss %.4f - val accuracy %.4f - val f1_micro %.4f - val f1_weighted %.4f' % (np.mean(losses), val_acc, val_f1_micro, val_f1_weighted))   
        ####################################Caculate metrics#####################################                 
            
    return np.mean(losses), val_acc, val_f1_weighted
        
def test(net, testloader, current_round, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    print("model 2")
    net.to(device)    
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
                 
    with torch.no_grad():
        losses = []
        class_hits = [0.0, 0.0] #
        class_counts = [0.0+1e-7, 0.0+1e-7] #        
        correct = 0
        test_sum = 0  
        conf_label, conf_pred = [], []
        ##auc_pred = np.array([])
        ##auc_label = np.array([])        
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  #|torch.Size([1])
            outputs = net(images)             #torch.Size([1, 3])
            
            ##y_pred = (torch.softmax(outputs, dim=1)).detach().cpu().numpy()
            ##if auc_pred.size == 0:
            ##    auc_pred = y_pred
            ##else:
            ##    auc_pred = np.vstack((auc_pred, y_pred))
            ##y_label = labels.cpu().numpy()
            ##auc_label = np.append(auc_label, y_label)            
            
            loss = criterion(outputs, labels).item()
            losses.append(loss)            
            _, preds = torch.max(outputs.data, 1)
            
            for idx in range(preds.shape[0]):
                class_counts[labels[idx].item()] += 1.0  
                conf_label.append(labels[idx].item())
                conf_pred.append(preds[idx].item())                  
                if preds[idx].item() == labels[idx].item():
                    class_hits[labels[idx].item()] += 1.0  
                    
            correct += (preds == labels).sum().item()
            test_sum+= labels.shape[0]

        ####################################Caculate metrics#####################################      
        test_acc = get_score(class_hits, class_counts, True)    
        test_conf_matrix = confusion_matrix(conf_label, conf_pred)
        #print("Confusion Matrix", test_conf_matrix)           
        test_f1_micro = f1_score(conf_label, conf_pred, average='micro')
        test_f1_weighted = f1_score(conf_label, conf_pred, average='weighted')
        print('Round %d - test loss %.4f - test accuracy %.4f - test f1_micro %.4f - test f1_weighted %.4f' % (current_round, np.mean(losses), test_acc, test_f1_micro, test_f1_weighted))   
        ####################################Caculate metrics#####################################                 
            
    return np.mean(losses), test_acc, test_f1_weighted