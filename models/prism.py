"""PRISM: a single global ridge router (random projections) over frozen ViT
features, dispatching each sample to a per-task adapted CosineLinear expert
head (see backbone/vit_prism.py)."""

import logging
import os
import re
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.inc_net import SimpleVitNet
from backbone.linears import CosineLinear
from models.base import BaseLearner
from models.router import RidgeRouter
from utils.toolkit import tensor2numpy

num_workers = 10

# How the task is chosen at inference. "class_max" is the default and is what
# the reported numbers use; the others are rebuttal ablations.
#   class_max : ridge over class labels, task score = max class score in task
#   class_sum : ridge over class labels, task score = sum of class scores
#   task      : ridge over task labels (T columns)
#   entropy   : per-sample lowest-entropy expert (evaluates every expert)
#   oracle    : ground-truth task (upper bound)
ROUTER_MODES = ("class_max", "class_sum", "task", "entropy", "oracle")


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

        self._task_ranges = []  # task_ranges[i] = (start_class, end_class)

        self._adapter_state_cache = {}
        self._expert_head_cache = {}

        self._ridge_P = None  # [feat_dim, M] fixed random projection
        self._router = RidgeRouter()

        self._router_mode = str(args.get("router_mode", "class_max"))
        if self._router_mode not in ROUTER_MODES:
            raise ValueError(f"Unknown router_mode {self._router_mode!r}")

        # Per-stage measurements, read by trainer.py.
        self.metrics = {}
        self.router_cost = []  # one dict per task: fit seconds, peak MB

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

    def _expert_scores(self, vit_features, task_idx):
        """Scores of task `task_idx`'s expert over its own classes: [B, inc]."""
        self._load_adapter(task_idx)
        feats = self._get_backbone().forward_adapter(vit_features)
        head = self._load_expert_head(task_idx)
        weight = head["weight"].to(vit_features.device)
        scores = F.linear(
            F.normalize(feats, p=2, dim=1), F.normalize(weight, p=2, dim=1)
        )
        if "sigma" in head:
            scores = head["sigma"].to(vit_features.device) * scores
        return scores

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
            scores = self._expert_scores(vit_features[row_ids], task_idx)
            start = int(starts[task_idx].item())
            out_logits[row_ids, start : start + inc] = scores
        return out_logits

    def _entropy_routed(self, vit_features):
        """Per-sample lowest-entropy expert. Evaluates every stored expert on
        the cached [CLS] feature. Returns (logits [B, total], chosen [B])."""
        inc = self._classes_per_task()
        n_tasks = len(self._task_ranges)
        scores = [self._expert_scores(vit_features, t) for t in range(n_tasks)]
        entropy = torch.stack(
            [
                -(F.softmax(s, dim=1) * torch.log(F.softmax(s, dim=1) + 1e-9)).sum(1)
                for s in scores
            ]
        )  # [T, B]
        chosen = entropy.argmin(dim=0)
        out = torch.full(
            (vit_features.size(0), self._total_classes),
            float("-inf"),
            device=vit_features.device,
        )
        for t, (start, _) in enumerate(self._task_ranges):
            rows = (chosen == t).nonzero(as_tuple=True)[0]
            out[rows, start : start + inc] = scores[t][rows]
        return out, chosen

    def _task_scores_from_logits(self, logits, reduce="max"):
        device = logits.device
        scores = torch.full(
            (logits.size(0), len(self._task_ranges)), float("-inf"), device=device
        )
        for t, (start, end) in enumerate(self._task_ranges):
            block = logits[:, start:end]
            scores[:, t] = block.max(dim=1).values if reduce == "max" else block.sum(1)
        return scores

    def _true_task(self, targets):
        ends = np.array([e for _, e in self._task_ranges])
        return np.searchsorted(ends, targets.cpu().numpy(), side="right")

    def _ridge_task(self, vit_features):
        """Task chosen by the ridge router in the configured mode: [B]."""
        lam = float(self.args.get("ridge_lambda", 1e4))
        ridge_out = self._ridge_features(vit_features) @ self._router.weight(lam)
        if self._router_mode == "task":
            return ridge_out.argmax(dim=1)
        reduce = "sum" if self._router_mode == "class_sum" else "max"
        return self._task_scores_from_logits(ridge_out, reduce).argmax(dim=1)

    def _routed_logits(self, inputs, targets):
        """Returns (logits, chosen_task) for the configured router_mode."""
        vit_features = self._extract_backbone_features(inputs)
        if self._router_mode == "entropy":
            return self._entropy_routed(vit_features)
        if self._router_mode == "oracle":
            chosen = torch.as_tensor(self._true_task(targets), device=inputs.device)
        else:
            chosen = self._ridge_task(vit_features)
        return self._routed_ridge_from_tasks(vit_features, chosen[:, None]), chosen

    def _get_global_routed_ridge_logits(self, inputs, lam=None):
        # Kept for external callers; same as the default class_max path.
        if lam is None:
            lam = float(self.args.get("ridge_lambda", 1e4))
        vit_features = self._extract_backbone_features(inputs)
        global_logits = self._ridge_features(vit_features) @ self._router.weight(lam)
        task_scores = self._task_scores_from_logits(global_logits)
        top1_task = task_scores.argmax(dim=1, keepdim=True)  # [B, 1]
        return self._routed_ridge_from_tasks(vit_features, top1_task)

    def after_task(self):
        self._network.backbone.reset_adapter()
        self._known_classes = self._total_classes

    def _sync(self):
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _fit_router(self, frozen, labels):
        """Accumulate this task's statistics and re-solve. Times only the
        router update (projection, G/C accumulation, solve), not the backbone
        pass that produced the features, and records its peak GPU memory."""
        total = self._total_classes
        if self._router_mode in ("entropy", "oracle"):
            return  # no ridge router in these arms
        cuda = self._device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
        base_mb = torch.cuda.memory_allocated(self._device) / 2**20 if cuda else 0.0
        self._sync()
        t0 = time.time()

        phi = self._ridge_features(frozen.to(self._device))  # [N, M]
        if self._router_mode == "task":
            labels = torch.full_like(labels, self._cur_task)
            n_out = self._cur_task + 1
        else:
            n_out = total
        self._router.accumulate(phi, labels, n_out)
        self._router.weight(float(self.args.get("ridge_lambda", 1e4)))

        self._sync()
        fit_s = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated(self._device) / 2**20 if cuda else 0.0
        cost = {
            "task": self._cur_task,
            "fit_s": round(fit_s, 3),
            "peak_mb": round(peak_mb, 1),
            "peak_above_base_mb": round(peak_mb - base_mb, 1),
            "G_mb": round(self._router.G.numel() * 4 / 2**20, 1),
        }
        self.router_cost.append(cost)
        logging.info(
            "Router cost Task %s: fit %.2fs, peak %.0f MB (%.0f MB above resident), "
            "M=%d, N=%d, columns=%d.",
            self._cur_task, fit_s, peak_mb, peak_mb - base_mb,
            phi.size(1), phi.size(0), n_out,
        )

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
        self._fit_router(torch.cat(frozen_list, dim=0), label_list)

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

    def eval_task(self):
        # Wall-clock over the whole eval pass / #samples, the same scope as
        # tosca-eval's "Eval (ms/smp)" (includes data loading).
        self._sync()
        t0 = time.time()
        cnn_accy, nme_accy = super().eval_task()
        self._sync()
        n = max(len(self.test_loader.dataset), 1)
        self.metrics["eval_ms_per_sample"] = (time.time() - t0) / n * 1000
        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true, routed, routed_true = [], [], [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                outputs, chosen = self._routed_logits(inputs, targets)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            routed.append(chosen.cpu().numpy())
            routed_true.append(self._true_task(targets))

        y_pred, y_true = np.concatenate(y_pred), np.concatenate(y_true)
        self._record_routing(
            np.concatenate(routed), np.concatenate(routed_true), y_pred[:, 0] == y_true
        )
        return y_pred, y_true

    def _record_routing(self, chosen, truth, top1_correct):
        """Separates 'wrong expert chosen' from 'right expert misclassifies',
        and saves the task confusion matrix (rows: true task, cols: routed)."""
        n_tasks = len(self._task_ranges)
        ok = chosen == truth
        self.metrics["routing_acc"] = float(ok.mean() * 100)
        self.metrics["acc_given_correct_routing"] = (
            float(top1_correct[ok].mean() * 100) if ok.any() else 0.0
        )
        conf = np.zeros((n_tasks, n_tasks), dtype=np.int64)
        np.add.at(conf, (truth, chosen), 1)
        per_task = conf.diagonal() / np.maximum(conf.sum(1), 1) * 100
        self.metrics["routing_per_task"] = [round(float(x), 2) for x in per_task]
        np.save(
            os.path.join(self._ckpt_dir(), f"routing_confusion_stage{self._cur_task}.npy"),
            conf,
        )

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
        if self._router.G is not None:
            torch.save(
                {"W": self._router.weight(lam).cpu(), "lambda": lam},
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
