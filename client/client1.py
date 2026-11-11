from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import os
import json
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import pickle
import random
import numpy as np
 
from model_client1 import NetS, NetT, train, val, test
from dataset_client1 import Dataset
from Data_Augmentation import get_training_augmentation, get_validation_augmentation

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
 
 
random_seed = 15
setup_seed(random_seed)
 
# Uploaded layers L^(S,i) and L^(T,i). Must match the patterns in server.py.
S_UPLOAD_PATTERNS = ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "conv4", "bn4", "conv5", "bn5"] 
T_UPLOAD_PATTERNS = ["model_t.conv1.", "model_t.bn1.", "model_t.layer1", "model_t.layer2",
                     "model_t.layer3", "model_t.layer4"]
 
def select_keys(net, patterns):
    """Only trainable parameters are uploaded (they are what gets clipped + noised).
    BN running stats / num_batches_tracked stay local."""
    return [n for n, _ in net.named_parameters() if any(p in n for p in patterns)]
 
def phi(eps):
    return eps / (eps + 1.0)

def create_json_file(model_s_test, model_t_test, json_path_s, json_path_t, valloader, testloader, current_round, device):
###########JSON IN##############
    val_loss_s, val_acc_s, val_f1_s = val(model_s_test, valloader, device)
    val_loss_t, val_acc_t, val_f1_t = val(model_t_test, valloader, device)
    if os.path.exists(json_path_s) and os.path.exists(json_path_t):
        with open(json_path_s, 'r') as file_s:
            with open(json_path_t, 'r') as file_t:
                existing_data_s = json.load(file_s)
                existing_data_s[current_round] = {}
                existing_data_s[current_round]["val_loss"] = val_loss_s
                existing_data_s[current_round]["val_acc"] = val_acc_s  
                existing_data_s[current_round]["val_f1"] = val_f1_s
                
                existing_data_t = json.load(file_t)
                existing_data_t[current_round] = {}
                existing_data_t[current_round]["val_loss"] = val_loss_t
                existing_data_t[current_round]["val_acc"] = val_acc_t  
                existing_data_t[current_round]["val_f1"] = val_f1_t                
                
                if existing_data_s["best_score"] < val_acc_s:    #val_acc
                    existing_data_s["best_score"] = val_acc_s    #val_acc
                    print("Best ACC achieved......", existing_data_s["best_score"])                     
                    test_loss_s, test_acc_s, test_f1_s = test(model_s_test, testloader, current_round, device)    
                    existing_data_s["test_loss"] = test_loss_s
                    existing_data_s["test_acc"] = test_acc_s
                    existing_data_s["test_f1"] = test_f1_s 
                    
                    existing_data_t["corresponding_score"] = val_acc_t    #val_acc
                    print("Corresponding ACC achieved......", existing_data_t["corresponding_score"])                     
                    test_loss_t, test_acc_t, test_f1_t = test(model_t_test, testloader, current_round, device)    
                    existing_data_t["test_loss"] = test_loss_t
                    existing_data_t["test_acc"] = test_acc_t
                    existing_data_t["test_f1"] = test_f1_t 
                    
                    ####################save best model####################
                    state_dict_s_best = {name: param for name, param in model_s_test.state_dict().items()}
                    with open(os.path.join("CLIENT1/output","model_s_best_client1"+".pkl"), 'wb') as file_s:
                        pickle.dump(state_dict_s_best, file_s)      
                    state_dict_t_best = {name: param for name, param in model_t_test.state_dict().items()}
                    with open(os.path.join("CLIENT1/output","model_t_best_client1"+".pkl"), 'wb') as file_t:
                        pickle.dump(state_dict_t_best, file_t)                    
                    

        with open(json_path_s, 'w') as file_s:    
            json.dump(existing_data_s, file_s, indent=4)
        with open(json_path_t, 'w') as file_t:    
            json.dump(existing_data_t, file_t, indent=4)            
    else:
        data_s = {}
        with open(json_path_s, 'w') as file_s:
            data_s[current_round] = {}
            data_s[current_round]["val_loss"] = val_loss_s
            data_s[current_round]["val_acc"] = val_acc_s     
            data_s[current_round]["val_f1"] = val_f1_s
            data_s["best_score"] = val_acc_s    #val_acc
  
            test_loss_s, test_acc_s, test_f1_s = test(model_s_test, testloader, current_round, device)  
            data_s["test_loss"] = test_loss_s
            data_s["test_acc"] = test_acc_s  
            data_s["test_f1"] = test_f1_s            
            json.dump(data_s, file_s, indent=4) 
        data_t = {}
        with open(json_path_t, 'w') as file_t:
            data_t[current_round] = {}
            data_t[current_round]["val_loss"] = val_loss_t
            data_t[current_round]["val_acc"] = val_acc_t     
            data_t[current_round]["val_f1"] = val_f1_t
            data_t["corresponding_score"] = val_acc_t    #val_acc

            test_loss_t, test_acc_t, test_f1_t = test(model_t_test, testloader, current_round, device)  
            data_t["test_loss"] = test_loss_t
            data_t["test_acc"] = test_acc_t 
            data_t["test_f1"] = test_f1_t           
            json.dump(data_t, file_t, indent=4)    
    return val_loss_s, val_acc_s, val_f1_s

def save_kd(current_round, kd, json_path_kd):
    data_kd = {}
    if os.path.exists(json_path_kd):
        with open(json_path_kd, 'r') as f:
            data_kd = json.load(f)

    data_kd[str(current_round)] = float(kd)

    with open(json_path_kd, 'w') as f:
        json.dump(data_kd, f, indent=4) 
###########JSON IN############## 


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, testloader, num_classes, clip_C_S, clip_C_T,
                 noise_z_S, noise_z_T, lambda_kd, calibrate=False) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model_g = NetS(num_classes)   # student
        self.model_l = NetT(num_classes)   # teacher
        self.model_g_test = NetS(num_classes)
        self.model_l_test = NetT(num_classes)
        self.json_path_kd = "CLIENT1/output/client1_kd.json" 
        self.json_path_s = "CLIENT1/output/client1_results_s.json"
        self.json_path_t = "CLIENT1/output/client1_results_t.json"
        self.s_round_file_path = "CLIENT1/output/round_model_s_client1.pkl"
        self.t_round_file_path = "CLIENT1/output/round_model_t_client1.pkl"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        self.clip_Cs = (clip_C_S, clip_C_T)     # (C_S, C_T), same for every client seperately
        self.noise_zs = (noise_z_S, noise_z_T)  # (z_S, z_T) noise multipliers
        self.alpha0 = lambda_kd         # initial distillation weight alpha_0
        self.calibrate = calibrate      # calibration run: no clipping, no noise
        self.norm_hist = {"S": [], "T": []}     # ||u^S||_2, ||u^T||_2 per round, for Calibrate clip
 
        self.s_keys = select_keys(self.model_g, S_UPLOAD_PATTERNS)
        self.t_keys = select_keys(self.model_l, T_UPLOAD_PATTERNS) 
 
        # lambda^m_{i,r}: share of model m's parameters that this client uploads
        def share(net, keys):
            p = dict(net.named_parameters())
            return sum(p[k].numel() for k in keys) / sum(v.numel() for v in p.values())
        self.lam_S = share(self.model_g, self.s_keys)
        self.lam_T = share(self.model_l, self.t_keys)
        print(f"[client1] lambda_S={self.lam_S:.4f} lambda_T={self.lam_T:.4f}")
 
    # ---------------- Alg. 3, ClientUpdate ----------------
    def set_parameters(self, parameters, global_keys):
        """Replace every uploaded layer l in L_{i,r} with the server's denoised result
        (global_{r-1} + u_check_{r-1}, sent by the server as full parameter values)."""
        print("client 3")
        lookup = dict(zip(global_keys, parameters))
        with torch.no_grad():
            for net, keys in ((self.model_g, self.s_keys), (self.model_l, self.t_keys)):
                params = dict(net.named_parameters())
                for k in keys:
                    if k not in lookup:
                        raise KeyError(f"Server did not send parameter {k}")
                    v = torch.as_tensor(lookup[k], dtype=params[k].dtype, device=params[k].device)
                    params[k].copy_(v)                     
 
    def adaptive_alpha(self, config): # Alg 3. Line 23-25
        """alpha_{i,r} = 2*alpha_0*q^T / (q^T + q^S), q^m = (1-lambda^m) + lambda^m * phi(eps^m)."""
        if "eps_S" not in config and "eps_T" not in config:   # first round: no SNR yet
            return self.alpha0
        q = {}
        for m, lam in (("S", self.lam_S), ("T", self.lam_T)):
            eps = config.get(f"eps_{m}")
            q[m] = 1.0 if eps is None else (1.0 - lam) + lam * phi(float(eps))
        alpha = 2.0 * self.alpha0 * q["T"] / (q["T"] + q["S"])
        print(f"[client1] eps_S={config.get('eps_S')} eps_T={config.get('eps_T')} "
              f"q_S={q['S']:.4f} q_T={q['T']:.4f} alpha={alpha:.4f} (alpha0={self.alpha0})")
        return alpha
 
    def get_parameters(self, config: Dict[str, Scalar]):
        print("client 1")
        p_g = dict(self.model_g.named_parameters())
        p_l = dict(self.model_l.named_parameters())
        return ([p_g[k].detach().cpu().numpy() for k in self.s_keys] +
                [p_l[k].detach().cpu().numpy() for k in self.t_keys])
 
    def fit(self, parameters, config):
        print("client 2")
        # Restore the local, never-uploaded parts (fc, BN running stats) from the previous round
        if os.path.isfile(self.s_round_file_path):
            with open(self.s_round_file_path, 'rb') as file:
                self.model_g.load_state_dict(pickle.load(file), strict=False)  #?[:-2]
            with open(self.t_round_file_path, 'rb') as file:
                self.model_l.load_state_dict(pickle.load(file), strict=False)                  
        else:
            print("wait for next round")                  
        self.current_round = config["current_round"]    
        global_keys = json.loads(config["global_keys"])
        self.set_parameters(parameters, global_keys)
        alpha = self.adaptive_alpha(config)
        save_kd(self.current_round, alpha, self.json_path_kd)
        optim = torch.optim.Adam(list(self.model_g.parameters()) + list(self.model_l.parameters()),
                                 lr=config["lr"])
 
        noisy_update, norms = train(self.model_g, self.model_l, self.trainloader, self.valloader, optim,
                                    config["local_epochs"], self.current_round, self.device,
                                    self.s_keys, self.t_keys, self.clip_Cs, self.noise_zs, alpha)
 
        # ---------- Calibrate clip: median of this client's per-model update norms ----------
        for m in ("S", "T"):
            self.norm_hist[m].append(norms[m])
        if self.calibrate:
            for m in ("S", "T"):
                h = self.norm_hist[m]
                print(f"[client1] Calibrate clip after round {self.current_round}: "
                      f"clip_C_{m} = {float(np.median(h)):.6g}  (median of {len(h)} rounds, "
                      f"all norms: {[round(v, 4) for v in h]})")
 
        keys = list(noisy_update.keys())
        return [noisy_update[k] for k in keys], len(self.trainloader), {
            "keys": json.dumps(keys),
            # per-coordinate noise std sigma_{i,m} = z_m * C_{i,m}; public hyperparameters, used by the server for V
            "sigma_S": float(self.noise_zs[0] * self.clip_Cs[0]) if self.noise_zs[0] > 0 else 0.0,
            "sigma_T": float(self.noise_zs[1] * self.clip_Cs[1]) if self.noise_zs[1] > 0 else 0.0,
        }    
 
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        with open(os.path.join("CLIENT1/output", "model_s_test_client1.pkl"), 'rb') as file_s:
            self.model_g_test.load_state_dict(pickle.load(file_s))
        with open(os.path.join("CLIENT1/output", "model_t_test_client1.pkl"), 'rb') as file_t:
            self.model_l_test.load_state_dict(pickle.load(file_t))
 
        val_loss_s, val_acc_s, val_f1_s = create_json_file(
            self.model_g_test, self.model_l_test, self.json_path_s, self.json_path_t,
            self.valloader, self.testloader, self.current_round, self.device)
        return float(val_loss_s), len(self.valloader), {"accuracy": val_acc_s}
 
 
def prepare_dataset_client1(path, num_partitions, file_path, batch_size):
    with open(file_path) as file:
        json_data = json.load(file)         
        train_num_ids = json_data["client "+str(1)]["train"]
        val_num_ids = json_data["client "+str(1)]["val"]
        test_num_ids = json_data["client "+str(1)]["test"]
        
        train_num_dataset = Dataset(
            img_ids=train_num_ids,
            img_dir=os.path.join(path),
            transform=get_training_augmentation(224,224))        
        train_dataloader = DataLoader(train_num_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_num_dataset = Dataset(
            img_ids=val_num_ids,
            img_dir=os.path.join(path),
            transform=get_validation_augmentation(224,224))            
        val_dataloader = DataLoader(val_num_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        test_num_dataset = Dataset(
            img_ids=test_num_ids,
            img_dir=os.path.join(path),
            transform=get_validation_augmentation(224,224))            
        test_dataloader = DataLoader(test_num_dataset, batch_size=batch_size, shuffle=False, num_workers=0)        
     
    return train_dataloader, val_dataloader, test_dataloader
 
@hydra.main(config_path="../../../conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    trainloader, validationloader, testloader = prepare_dataset_client1(cfg.path, cfg.num_clients, cfg.json_path, cfg.batch_size)
    calibrate = bool(cfg.get("calibrate", False))
    if calibrate:
        print("[client1] CALIBRATION RUN: no clipping, no noise; clip_C_S / clip_C_T printed after each round")
    clip_C_S = float("inf") if calibrate else cfg.clip_C_S_client1
    clip_C_T = float("inf") if calibrate else cfg.clip_C_T_client1
    noise_z_S = 0.0 if calibrate else cfg.noise_z_S
    noise_z_T = 0.0 if calibrate else cfg.noise_z_T
 
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(
            trainloader=trainloader,
            valloader=validationloader,
            testloader=testloader,
            num_classes=cfg.num_classes,
            clip_C_S=clip_C_S,
            clip_C_T=clip_C_T,
            noise_z_S=noise_z_S,
            noise_z_T=noise_z_T,
            lambda_kd=cfg.lambda_kd,
            calibrate=calibrate,
        ),
    )
 
 
if __name__ == '__main__':
    main()
    