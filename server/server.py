import json
import random
from collections import OrderedDict, defaultdict
from logging import WARNING

import hydra
import numpy as np
import torch
import flwr as fl
from omegaconf import DictConfig
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log

from model_server import Net

UPLOAD_PATTERNS = {
    "cnn5": ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "conv4", "bn4", "conv5", "bn5"],
    "resnet18": ["model.conv1.", "model.bn1.", "model.layer1", "model.layer2",
                 "model.layer3", "model.layer4"],
}


def select_keys(net, patterns):
    return [n for n, _ in net.named_parameters() if any(p in n for p in patterns)]


def layer_of(key):
    """Layer l = module (e.g. weight+bias of one conv / BN). 'model_t.layer1.0.conv1.weight' -> 'model_t.layer1.0.conv1'."""
    return key.rsplit(".", 1)[0]


class FedLDPStrategy(fl.server.strategy.FedAvg):
    def __init__(self, beta, arch="cnn5", num_classes=8, use_shrinkage=True, **kwargs):
        arch = arch.lower()
        if arch not in UPLOAD_PATTERNS:
            raise ValueError(f"Unknown arch '{arch}', choose from {tuple(UPLOAD_PATTERNS)}")
        self.arch = arch
        net = Net(num_classes, arch)
        p = dict(net.named_parameters())
        self.global_params = OrderedDict(
            (k, p[k].detach().cpu().numpy().copy()) for k in select_keys(net, UPLOAD_PATTERNS[arch]))
        self.global_keys = list(self.global_params.keys())
        print(f"[server] arch={arch}, {len(self.global_keys)} uploaded parameter tensors")
 
        self.d = defaultdict(int)                       # d_l
        for k, v in self.global_params.items():
            self.d[layer_of(k)] += v.size
 
        self.beta = beta
        self.use_shrinkage = use_shrinkage             # False: rho = 1 (plain averaging), ablation
        self.P_bar = {}                                 # smoothed effective update power per layer
 
        super().__init__(initial_parameters=ndarrays_to_parameters(list(self.global_params.values())),
                         **kwargs)
        
    def configure_fit(self, server_round, parameters, client_manager):
        print("222222222222222")
        pairs = super().configure_fit(server_round, parameters, client_manager)
        out = []
        for client, fit_ins in pairs:
            cfg = dict(fit_ins.config)                  # copy: FedAvg shares one dict across clients
            cfg["global_keys"] = json.dumps(self.global_keys)
            out.append((client, FitIns(fit_ins.parameters, cfg)))
        return out

    def aggregate_fit(self, server_round, results, failures):
        print("3333333333333333")
        if failures:
            print(f"Failures during round {server_round}: {len(failures)}")
        if not results:
            return None, {}

        # --- collect noisy updates u~_{i,r}, by parameter name ---
        sums = {}
        clients_of_layer = defaultdict(set)             # kappa_r^l
        sig2_sum = defaultdict(float)                   # sum_{i in kappa_r^l} sigma_{i,m}^2
        for client, res in results:
            keys = json.loads(res.metrics["keys"])
            sig = float(res.metrics["sigma"])  
            arrays = parameters_to_ndarrays(res.parameters)
            layers = set()
            for k, a in zip(keys, arrays):
                if k not in self.global_params:
                    raise KeyError(f"Client {client.cid} uploaded unknown parameter {k}")
                sums[k] = sums.get(k, 0.0) + a.astype(np.float64)
                layers.add(layer_of(k))
            for l in layers:
                clients_of_layer[l].add(client.cid)
                sig2_sum[l] += sig ** 2               
            print(f"[round {server_round}] cid={client.cid}: sigma={sig:.4g} layers={len(layers)}")

        keys_of_layer = defaultdict(list)
        for k in sums:
            keys_of_layer[layer_of(k)].append(k)
        rhos, snrs = {}, {}
        for l, ks in keys_of_layer.items():
            n = len(clients_of_layer[l])
            u_bar = {k: sums[k] / n for k in ks}
            V = self.d[l] * sig2_sum[l] / n ** 2                                   # noise power
            P_hat = sum(float((u_bar[k] ** 2).sum()) for k in ks) - V              # signal power estimate
            if l in self.P_bar:
                self.P_bar[l] = self.beta * self.P_bar[l] + (1 - self.beta) * P_hat
            else:
                self.P_bar[l] = P_hat                                               # first observation
            P_pos = max(self.P_bar[l], 0.0)
            if not self.use_shrinkage or V == 0:
                rho = 1.0                                                           # plain averaging
            else:
                rho = P_pos / (P_pos + V)                                           # MSE-optimal shrinkage
            rhos[l] = rho
            snrs[l] = float("inf") if V == 0 else P_pos / V
            for k in ks:                                                            # global += u_check
                self.global_params[k] = (self.global_params[k] + rho * u_bar[k]).astype(self.global_params[k].dtype)
 
        r = list(rhos.values())
        print(f"[round {server_round}] layers={len(r)} rho mean={np.mean(r):.4f} "
              f"min={np.min(r):.4f} max={np.max(r):.4f}")

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                [(res.num_examples, res.metrics) for _, res in results])
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")
 
        return ndarrays_to_parameters(list(self.global_params.values())), metrics_aggregated


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "local_epochs": config.local_epochs,
            "current_round": server_round,
        }
    return fit_config_fn


@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    calibrate = bool(cfg.get("calibrate", False))
    if calibrate:
        print("[server] CALIBRATION RUN: noise_z=0, plain averaging (rho=1)")

    strategy = FedLDPStrategy(
        beta=cfg.beta,
        arch=str(cfg.get("model_arch", "cnn5")),
        num_classes=cfg.num_classes,
        use_shrinkage=bool(cfg.get("use_shrinkage", True)),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
    )    

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )


if __name__ == '__main__':
    main()