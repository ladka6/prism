"""PRISM: a single global ridge router (random projections) over frozen ViT
features, dispatching each sample to a per-task adapted CosineLinear expert
head (see backbone/vit_prism.py)."""

import logging
import os
import re
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.inc_net import SimpleVitNet
from backbone.linears import CosineLinear
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 10


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

        self._task_ranges = []  # task_ranges[i] = (start_class, end_class)

        self._adapter_state_cache = {}
        self._expert_head_cache = {}

        self._ridge_P = None  # [feat_dim, M] fixed random projection
        self._ridge_W_cache = {}  # ("__global__", lambda) -> [M, total_classes]
        self._ridge_G_global = None  # [M, M]
        self._ridge_C_global = None  # [M, total_classes]

    def incremental_train(self, data_manager):
        self._setup_task_loaders(data_manager)
        self._train()

    def _setup_task_loaders(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        self._task_ranges.append((self._known_classes, self._total_classes))
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        logging.info(f"Task ranges so far: {self._task_ranges}")

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=48, shuffle=False, num_workers=num_workers
        )

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )

    def _train(self):
        self._network.to(self._device)
        self._set_trainable()

        inc = self._classes_per_task()
        start = self._task_ranges[self._cur_task][0]
        head = CosineLinear(self._network.feature_dim, inc).to(self._device)

        optimizer = self.get_optimizer(
            lr=self.args["lr"], extra_params=head.parameters()
        )
        scheduler = self.get_scheduler(optimizer, self.args["epochs"])

        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            head.train()
            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in self.train_loader:
                inputs, targets = (
                    inputs.to(self._device),
                    targets.long().to(self._device),
                )
                local_targets = targets - start

                optimizer.zero_grad()
                feats = self._get_backbone()(inputs)
                logits = head(feats)["logits"]
                loss = F.cross_entropy(logits, local_targets)
                l1_loss = sum(
                    p.abs().sum()
                    for p in self._network.backbone.prism_adapter.parameters()
                )
                loss = loss + self.args["l1"] * l1_loss
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(local_targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["epochs"],
                losses / len(self.train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
        self._save_adapter()
        self._save_expert_head(head)
        if self._cur_task == 0:
            self._save_adaptmlp()

        self.replace_fc()
        self._save_ridge_state()

    def _get_backbone(self):
        if isinstance(self._network, torch.nn.DataParallel):
            return self._network.module.backbone
        return self._network.backbone

    def _set_trainable(self):
        for p in self._network.parameters():
            p.requires_grad = False
        backbone = self._get_backbone()
        for p in backbone.prism_adapter.parameters():
            p.requires_grad = True
        if self._cur_task == 0:
            for name, p in backbone.vit.named_parameters():
                if "adaptmlp" in name:
                    p.requires_grad = True

    def _extract_backbone_features(self, inputs):
        backbone = self._get_backbone()
        return backbone.forward_features(inputs)

    def _classes_per_task(self):
        start, end = self._task_ranges[0]
        return end - start

    def _ridge_projection(self, feature_dim):
        if self._ridge_P is None:
            M = int(self.args.get("ridge_proj_dim", 5000))
            gen = torch.Generator(device="cpu").manual_seed(1993)
            P = torch.randn(feature_dim, M, generator=gen)
            self._ridge_P = P.to(self._device)
        return self._ridge_P

    def _ridge_features(self, feats):
        if bool(self.args.get("ridge_normalize", True)):
            feats = F.normalize(feats, p=2, dim=1)
        phi = feats @ self._ridge_projection(feats.size(1))
        if str(self.args.get("ridge_activation", "relu")) == "relu":
            phi = F.relu(phi)
        return phi

    def _save_expert_head(self, head):
        t = self._cur_task
        state = {"weight": head.weight.detach().cpu()}
        if getattr(head, "sigma", None) is not None:
            state["sigma"] = head.sigma.detach().cpu()
        path = os.path.join(self._ckpt_dir(), f"head_task{t}.pth")
        torch.save(state, path)
        logging.info("Expert head saved for task %s.", t)

    def _load_expert_head(self, idx):
        state = self._expert_head_cache.get(idx)
        if state is None:
            state = torch.load(
                os.path.join(self._ckpt_dir(), f"head_task{idx}.pth"),
                map_location=self._device,
            )
            self._expert_head_cache[idx] = state
        return state

    def _routed_ridge_from_tasks(self, vit_features, topk_tasks):
        inc = self._classes_per_task()
        device = vit_features.device
        batch_size = vit_features.size(0)
        total_classes = self._total_classes
        starts = torch.tensor([s for s, _ in self._task_ranges], device=device)

        out_logits = torch.full(
            (batch_size, total_classes), float("-inf"), device=device
        )
        for task_idx in torch.unique(topk_tasks).tolist():
            task_idx = int(task_idx)
            row_ids = (topk_tasks == task_idx).any(dim=1).nonzero(as_tuple=True)[0]
            self._load_adapter(task_idx)
            feats = self._get_backbone().forward_adapter(vit_features[row_ids])
            head = self._load_expert_head(task_idx)
            weight = head["weight"].to(device)
            scores = F.linear(
                F.normalize(feats, p=2, dim=1), F.normalize(weight, p=2, dim=1)
            )
            if "sigma" in head:
                scores = head["sigma"].to(device) * scores
            start = int(starts[task_idx].item())
            out_logits[row_ids, start : start + inc] = scores
        return out_logits

    def _task_scores_from_logits(self, logits):
        device = logits.device
        scores = torch.full(
            (logits.size(0), len(self._task_ranges)), float("-inf"), device=device
        )
        for t, (start, end) in enumerate(self._task_ranges):
            scores[:, t] = logits[:, start:end].max(dim=1).values
        return scores

    def _get_global_routed_ridge_logits(self, inputs, lam=None):
        if lam is None:
            lam = float(self.args.get("ridge_lambda", 1e4))
        vit_features = self._extract_backbone_features(inputs)
        global_logits = self._ridge_features(vit_features) @ self._global_ridge_weight(
            lam
        )
        task_scores = self._task_scores_from_logits(global_logits)
        top1_task = task_scores.argmax(dim=1, keepdim=True)  # [B, 1]
        return self._routed_ridge_from_tasks(vit_features, top1_task)

    def _accumulate_global_ridge(self, frozen, labels):
        total = self._total_classes
        frozen = frozen.to(self._device)
        labels = labels.to(self._device).long()
        phi = self._ridge_features(frozen)  # [N, M]
        onehot = torch.zeros(phi.size(0), total, device=self._device)
        onehot[torch.arange(phi.size(0), device=self._device), labels] = 1.0
        G = (phi.t() @ phi).float()
        C = (phi.t() @ onehot).float()
        if self._ridge_G_global is None:
            self._ridge_G_global = G
            self._ridge_C_global = C
        else:
            self._ridge_G_global = self._ridge_G_global + G
            prev = self._ridge_C_global
            grown = torch.zeros(prev.size(0), total, device=self._device)
            grown[:, : prev.size(1)] = prev
            self._ridge_C_global = grown + C
        self._ridge_W_cache.clear()
        logging.info(
            "Global ridge accumulated Task %s: M=%d, N=%d, classes=%d.",
            self._cur_task,
            phi.size(1),
            phi.size(0),
            total,
        )

    def _global_ridge_weight(self, lam):
        key = ("__global__", float(lam))
        W = self._ridge_W_cache.get(key)
        if W is None:
            G = self._ridge_G_global.double()
            C = self._ridge_C_global.double()
            eye = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
            W = torch.linalg.solve(G + float(lam) * eye, C).float()
            self._ridge_W_cache[key] = W
        return W

    def after_task(self):
        self._network.backbone.reset_adapter()
        self._known_classes = self._total_classes

    def replace_fc(self):
        self._network.eval()
        frozen_list = []
        label_list = []
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                label = label.long().to(self._device)
                vit = self._extract_backbone_features(data)
                frozen_list.append(vit.cpu())
                label_list.append(label.cpu())
        label_list = torch.cat(label_list, dim=0)
        self._accumulate_global_ridge(torch.cat(frozen_list, dim=0), label_list)

    def get_optimizer(self, lr, extra_params=None):
        params = list(filter(lambda p: p.requires_grad, self._network.parameters()))
        if extra_params is not None:
            params += list(extra_params)
        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(
                params,
                momentum=0.9,
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adam":
            optimizer = optim.Adam(
                params,
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        else:
            raise NotImplementedError(
                "Unknown optimizer {}".format(self.args["optimizer"])
            )
        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == "constant":
            scheduler = None
        elif self.args["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=epoch, eta_min=1e-8
            )
        elif self.args["scheduler"] == "steplr":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["milestones"],
            )
        else:
            raise NotImplementedError(
                "Unknown scheduler {}".format(self.args["scheduler"])
            )
        return scheduler

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                outputs = self._get_global_routed_ridge_logits(inputs)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _ckpt_dir(self):
        dataset = str(self.args.get("dataset", "data"))
        prefix = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(self.args.get("prefix", "")).strip()
        )
        tag = f"{dataset}__{prefix}" if prefix else dataset
        seed = self.args.get("seed")
        if isinstance(seed, (int, np.integer)):
            tag = f"{tag}__seed{int(seed)}"
        path = os.path.join("prism", tag)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_adaptmlp(self):
        state = {
            name: param
            for name, param in self._network.state_dict().items()
            if "adaptmlp" in name
        }
        path = os.path.join(self._ckpt_dir(), "adaptmlp.pth")
        torch.save(state, path)
        logging.info("AdaptMLP parameters saved to %s.", path)

    def _load_adaptmlp(self):
        path = os.path.join(self._ckpt_dir(), "adaptmlp.pth")
        state = torch.load(path, map_location=self._device)
        current_state_dict = self._network.state_dict()
        current_state_dict.update(state)
        self._network.load_state_dict(current_state_dict)

    def _save_ridge_state(self):
        t = self._cur_task
        ckpt_dir = self._ckpt_dir()
        lam = float(self.args.get("ridge_lambda", 1e4))
        if self._ridge_G_global is not None:
            torch.save(
                {"W": self._global_ridge_weight(lam).cpu(), "lambda": lam},
                os.path.join(ckpt_dir, "ridge_global.pth"),
            )
        logging.info("Global ridge state saved for task %s.", t)

    def _save_adapter(self):
        path = os.path.join(self._ckpt_dir(), f"task{self._cur_task}.pth")
        adapter_state_dict = {
            name: param
            for name, param in self._network.state_dict().items()
            if "prism_adapter" in name
        }
        torch.save(adapter_state_dict, path)
        logging.info(f"Adapter parameters saved to {path}.")

    def _load_adapter(self, idx):
        adapter_state_dict = self._adapter_state_cache.get(idx)
        if adapter_state_dict is None:
            adapter_state_dict = torch.load(
                os.path.join(self._ckpt_dir(), f"task{idx}.pth"),
                map_location=self._device,
            )
            self._adapter_state_cache[idx] = adapter_state_dict
        current_state_dict = self._network.state_dict()
        current_state_dict.update(adapter_state_dict)
        self._network.load_state_dict(current_state_dict)
