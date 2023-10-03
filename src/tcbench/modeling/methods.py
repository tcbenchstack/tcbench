"""
This modules contains a set of classes 
used to handle training and inference,
namely Trainer objects
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Tuple, Generator, Dict, Any
from collections import defaultdict
from torch import nn
from aim import Run
from sklearn.metrics import classification_report, confusion_matrix
from rich import progress

import torch
import os
import torchsummary
import pathlib
import aim
import logging
import time

from tcbench.modeling import losses, utils, backbone

import xgboost as xgb


def _make_deterministic():
    """Helper method to force pytorch to be deterministic"""
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _reset_optimizer(
    optimizer: torch.optim.Optimizer, net_parameters: Generator
) -> torch.optim.Optimizer:
    """Helper method to recreate a new instance of
    the optimizer passed as input (via introspection)
    with the same configuration and bound to the
    network parameters of a model

    Arguments:
        optimizer: the optimizer to clone
        net_parameters: iterator obtained calling .parameters() from a pytorch module

    Return:
        a new instance of an optimizer
    """
    optimizer_class = getattr(torch.optim, optimizer.__class__.__name__)
    optimizer_params = {
        name: value
        for name, value in optimizer.param_groups[0].items()
        if name != "params"
    }
    optimizer = optimizer_class(net_parameters, **optimizer_params)
    return optimizer


class PatienceMonitorLoss:
    """A callable class implementing monitoring of
    a loss metric

    Attributes:
        steps: the maximum patience
        steps_left: steps left before patience expires
        min_delta: the minimum difference against
            the best loss observed so far to
            be considered as an improvement
        best_loss: the best loss observed so far
        best_epoch: teh epoch when the best loss was observed
    """

    def __init__(self, steps: int = 5, min_delta: float = 0.001):
        """
        Arguments:
            steps: the maximum patience
            min_delta: the minimum difference against
        """
        self.steps = steps
        self.steps_left = steps
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.best_epoch = -1

    def is_improved(self, loss: float) -> bool:
        """Returns true if input loss differ
        from the best observed loss by at least
        min_delta"""
        diff = self.best_loss - loss
        return diff > self.min_delta

    def is_expired(self) -> bool:
        """Returns True if steps_left == 0"""
        return self.steps_left == 0

    def get_best_metrics(self) -> Dict[str, float]:
        """Returns a dictionary with best loss and epoch observed"""
        return dict(best_loss=self.best_loss, best_epoch=self.best_epoch)

    def __call__(self, metrics: Dict[str, float], idx_epoch: int) -> bool:
        """The input metrics is a dictionary expected to have
        a "loss" key, and the related value is compared against
        the best loss observed so far. Returns True if the
        input loss is improved wrt the best loss observed"""
        loss = metrics["loss"]
        if self.is_improved(loss):
            self.best_loss = loss
            self.steps_left = self.steps
            self.best_epoch = idx_epoch
            return True
        self.steps_left -= 1
        return False


class PatienceMonitorAccuracy:
    """A callable class implementing monitoring of
    a performance metric

    Attributes
        steps: the maximum patience
        steps_left: steps left before patience expires
        best_acc: the best loss observed so far
        best_epoch: the epoch when the best loss was observed
        acc_name: the name of the performance metric
    """

    def __init__(self, name: str = "acc", steps: int = 3):
        """
        Arguments:
            name: the name of the performance metric
            steps: the maximum patience
        """
        self.steps = steps
        self.steps_left = steps
        self.best_acc = -np.inf
        self.best_epoch = -1
        self.acc_name = name

    def is_improved(self, acc: float) -> bool:
        """Return True if the input accuracy is
        higher than the best observed so far"""
        return acc > self.best_acc

    def is_expired(self) -> bool:
        """Returns True if steps_left == 0"""
        return self.steps_left == 0

    def get_best_metrics(self) -> Dict[str, float]:
        """Returns a dictionary with best loss and epoch observed"""
        name = f"best_{self.acc_name}"
        return {name: self.best_acc, "best_epoch": self.best_epoch}

    def __call__(self, metrics: Dict[str, float], idx_epoch: int) -> bool:
        """The input metrics is a dictionary expected to have
        a key with the same name provided when instanciating
        the class. The related value is compared against
        the best loss observed so far. Returns True if the
        performance metric is improved wrt the best performance observed"""
        acc = metrics[self.acc_name]
        if self.is_improved(acc):
            self.best_acc = acc
            self.steps_left = self.steps
            self.best_epoch = idx_epoch
            return True
        self.steps_left -= 1
        return False


class SimpleTrainer:
    """A base class offering functionality for
    training and testing a supervised model
    """

    def __init__(
        self,
        net: backbone.BaseNet,
        optimizer: pytorch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str = "cuda:0",
        deterministic: bool = True,
        tracker: aim.Run = None,
        logger: logging.Logger = None,
    ):
        """
        Arguments:
            net: the architecture to use
            optimizer: the optimizer to use
            criterion: the instance of the loss to use
            device: the device to use
            deterministic: see _make_deterministic()
            tracker: the AIM run on which register metrics
            logger: the logging reference
        """
        if deterministic:
            _make_deterministic()

        self.device = device
        self.optimizer = optimizer
        self.logger = logger

        self.net = net
        if self.net:
            self.net = net.double().to(device)
        self.criterion = criterion
        self.tracker = tracker
        self._is_training = False
        self._reset_metrics()

    def log_msg(self, msg: str) -> None:
        """Register a message to file and echoes it
        to the console"""
        utils.log_msg(msg, self.logger)

    def _reset_metrics(self) -> None:
        """Helper method to clean (before training)
        internal objects used for tracking metrics
        """
        self.best_model = None
        self.metrics = defaultdict(list)

    def _track_metrics(
        self, metrics: Dict[str, float], context: str, epoch: int = None
    ) -> None:
        """Helper method invoked during training, validation
        and testing to track loss and performance metrics"""
        for name, value in metrics.items():
            self.metrics[f"{context}_{name}"].append(value)
            if self.tracker:
                self.tracker.track(
                    value, name, epoch=epoch, context=dict(subset=context)
                )

    def _do_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        idx_epoch: int,
        context: str,
        track_preds_targets: bool = False,
    ) -> Dict[str, Any]:
        """Helper method invoked during training, validation
        and testing to perform forward (and backward) propagation

        Arguments:
            data_loader: an instance of a pytorch DataLoader
            idx_epoch: the current epoch
            context: a string (for tracking) for extra semantic (train, val, etc.)
            track_preds_targets: if True, return predicted and target labels

        Returns:
            A dictionary of metrics containing the measured average "loss" and "acc".
            If track_preds_targets is True, the dictionary contains also the keys
           "preds" and "targets", each mapped to a list of integer labels
        """
        cum_loss = 0
        correct = 0
        samples = 0
        preds = []
        targets = []
        num_batches = int(np.ceil(data_loader.dataset.df.shape[0] / data_loader.batch_size))

        for batch_idx, (x, y) in progress.track(enumerate(data_loader), description='', total=num_batches, transient=True):
            # x can be a list if the data loader
            # is associated to a multi-view dataset
            # but in this context we are not using contrastive
            # learning, hence we concat the views
            if isinstance(x, list):
                y = y.repeat(len(x))
                x = torch.cat(x, dim=0)
            x = x.to(self.device)
            y = y.to(self.device)

            scores = self.net.forward(x)
            loss = self.criterion(scores, y)
            cum_loss += loss.item()

            if self._is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            y_pred = scores.argmax(dim=1)
            correct += (y_pred == y).sum().item()
            samples += x.shape[0]

            if track_preds_targets:
                preds.extend(y_pred.cpu().numpy().tolist())
                targets.extend(y.cpu().numpy().tolist())

            #print(".", end="", flush=True)

        #print("\r" + " " * (batch_idx + 1), end="", flush=True)
        #print("\r", end="", flush=True)
        metrics = dict(
            loss=cum_loss / (batch_idx+1),
            acc=100 * correct / samples,
        )
        self._track_metrics(metrics, epoch=idx_epoch, context=context)
        if track_preds_targets:
            metrics["preds"] = preds
            metrics["targets"] = targets
        return metrics

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader, idx_epoch: int, context: str
    ) -> Dict[str, Any]:
        """Set the internal model to train, calls _do_epoch() and return the obtained metrics"""
        self.net.train()
        self._is_training = True
        out = self._do_epoch(train_loader, idx_epoch, context)
        self._is_training = False
        return out

    def train_loop(
        self,
        epochs: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        patience_monitor: PatienceMonitorLoss | PatienceMonitorAccuracy = None,
        quiet: bool = False,
        context: str = None,
    ) -> backbone.BaseNet:
        """
        The entry point for triggering training

        Arguments:
            epochs: number of epochs to run
            train_loader: the data for training
            val_loader: the data for validation
            patience_monitor: the instance of the patience monitor
            quiet: if False, no message on the console is reported
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Returns:
            The best model obtained during training
        """
        if context is None:
            context = ""
        else:
            context += "_"

        if self.optimizer is None:
            raise RuntimeError("optimizer cannot be None when training")

        self.net = self.net.to(self.device)
        self._reset_metrics()

        if backbone.has_dropout_layer(self.net):
            utils.log_msg(
                "---\nWARNING: Detected Dropout layer!\nWARNING: During supervised training, the monitored train_acc will be inaccurate\n---",
                self.logger,
            )

        for idx_epoch in range(epochs):
            if patience_monitor and patience_monitor.is_expired():
                break
            train_metrics = self.train_one_epoch(
                train_loader, idx_epoch, context=f"{context}train"
            )
            msg = f"epoch: {idx_epoch:3d} | "
            msg += "train_loss: {loss:.6f}".format(loss=train_metrics["loss"])
            for metric_name, metric_value in train_metrics.items():
                if "acc" not in metric_name:
                    continue
                metric_name = f"train_{metric_name}"
                msg += f" | {metric_name}: {metric_value:5.1f}%"

            if val_loader:
                val_metrics, _ = self.test_loop(
                    val_loader, idx_epoch, with_reports=False, context=f"{context}val"
                )
                msg += " | val_loss: {loss:.6f}".format(loss=val_metrics["loss"])
                for metric_name, metric_value in val_metrics.items():
                    if "acc" not in metric_name:
                        continue
                    metric_name = f"val_{metric_name}"
                    msg += f" | {metric_name}: {metric_value:5.1f}%"

                if patience_monitor and patience_monitor(val_metrics, idx_epoch):
                    self.best_model = self.net.get_copy()
                    metrics = patience_monitor.get_best_metrics()
                    self._track_metrics(metrics, context=f"{context}val")
                    msg += " | *"
            else:
                if patience_monitor and patience_monitor(train_metrics, idx_epoch):
                    self.best_model = self.net.get_copy()
                    metrics = patience_monitor.get_best_metrics()
                    self._track_metrics(metrics, context=f"{context}train")
                    msg += " | *"
                else:
                    self.best_model = self.net.get_copy()

            if not quiet:
                self.log_msg(msg)

        if not quiet:
            if patience_monitor and patience_monitor.is_expired():
                self.log_msg("run out of patience")
            else:
                self.log_msg("reached max epochs")

        self.net.set_state_dict(self.best_model)
        return self.net

    def test_loop(
        self,
        data_loader: torch.utils.data.DataLoader,
        idx_epoch: int = None,
        with_reports: bool = False,
        context: str = None,
    ) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """
        Run inference on a model (for testing or validation)

        Arguments:
            data_loader: the data to use
            idx_epoch: the current epoch
            with_reports: if True, compute classification report and confusion matrix
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            A tuple with two dictionaries. The first contains the metrics collected
            during inference; the second contains classification report (class_rep)
            and confusion matrix (conf_mtx) or is empty {} if their computation
            was not requested
        """
        self.net.eval()
        if context is None:
            context = "val" if self._is_training else "test"
        with torch.no_grad():
            metrics = self._do_epoch(
                data_loader, idx_epoch, track_preds_targets=True, context=context
            )

        preds = metrics["preds"]
        targets = metrics["targets"]
        del metrics["preds"]
        del metrics["targets"]

        self._track_metrics(metrics, epoch=idx_epoch, context=context)

        reports = {}
        if with_reports:
            reports = dict(
                class_rep=pd.DataFrame(
                    classification_report(targets, preds, output_dict=True)
                ).T,
                conf_mtx=pd.DataFrame(confusion_matrix(targets, preds)),
                preds=preds,
            )

        return metrics, reports


class XGboostTrainer:
    """A base class offering functionality for
    training and testing a supervised model
    """

    def __init__(
        self,
        xgboost_model:Any,
        net:Any=None,
        device:Any=None,
        tracker: aim.Run = None,
        logger: logging.Logger = None,
    ):
        """
        Arguments:
            xgboost_model: XGboost model
            tracker: the AIM run on which register metrics
            logger: the logging reference
        """
        self.xgboost_model = xgboost_model
        self.logger = logger
        self.tracker = tracker
        self._is_training = False
        self._reset_metrics()

    def log_msg(self, msg: str) -> None:
        """Register a message to file and echoes it
        to the console"""
        utils.log_msg(msg, self.logger)

    def _reset_metrics(self) -> None:
        """Helper method to clean (before training)
        internal objects used for tracking metrics
        """
        self.best_model = None
        self.metrics = defaultdict(list)

    def _track_metrics(self, metrics: Dict[str, float], context: str) -> None:
        """Helper method invoked during training, validation
        and testing to track loss and performance metrics"""
        for name, value in metrics.items():
            self.metrics[f"{context}_{name}"].append(value)
            if self.tracker:
                self.tracker.track(value, name, context=dict(subset=context))

    def _do_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        context: str,
        track_preds_targets: bool = False,
    ) -> Dict[str, Any]:
        """Helper method invoked during training, validation
        and testing to perform forward (and backward) propagation

        Arguments:
            data_loader: an instance of a pytorch DataLoader
            context: a string (for tracking) for extra semantic (train, val, etc.)
            track_preds_targets: if True, return predicted and target labels

        Returns:
            A dictionary of metrics containing the measured average "loss" and "acc".
            If track_preds_targets is True, the dictionary contains also the keys
           "preds" and "targets", each mapped to a list of integer labels
        """
        cum_loss = 0
        correct = 0
        samples = 0
        preds = []
        targets = []
        x_all = []
        y_all = []

        for batch_idx, (x, y) in enumerate(data_loader):
            # x can be a list if the data loader
            # is associated to a multi-view dataset
            # but in this context we are not using contrastive
            # learning, hence we concat the views
            # if isinstance(x, list):
            #    y = y.repeat(len(x))
            #    x = torch.cat(x, dim=0)

            # scores = self.net.forward(x)
            # loss = self.criterion(scores, y)
            # cum_loss += loss.item()
            # print(x.reshape(x.shape[0], -1).shape)
            # x = x.cpu().numpy()
            # y = y.cpu().numpy()
            x_all.append(x.reshape(x.shape[0], -1))
            # y_all.append(y.reshape(y.shape[0], -1))
            y_all.append(y)

        x_all = np.concatenate(x_all, axis=0)
        # x_all = xgb.DMatrix(x_all)
        y_all = np.concatenate(y_all, axis=0)
        if self._is_training:
            self.xgboost_model.fit(x_all, y_all)

        y_pred = self.xgboost_model.predict(x_all)
        correct += (y_pred == y_all).sum().item()
        samples += x_all.shape[0]

        if track_preds_targets:
            preds.extend(y_pred.tolist())
            targets.extend(y_all.tolist())

        print(".", end="", flush=True)

        print("\r" + " " * (batch_idx + 1), end="", flush=True)
        print("\r", end="", flush=True)
        metrics = dict(
            # loss=cum_loss / batch_idx,
            acc=100
            * correct
            / samples,
        )
        self._track_metrics(metrics, context=context)
        if track_preds_targets:
            metrics["preds"] = preds
            metrics["targets"] = targets
        return metrics

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader, context: str
    ) -> Dict[str, Any]:
        """Set the internal model to train, calls _do_epoch() and return the obtained metrics"""
        self._is_training = True
        out = self._do_epoch(train_loader, context)
        self._is_training = False
        return out

    def train_loop(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        patience_monitor: PatienceMonitorLoss | PatienceMonitorAccuracy = None,
        quiet: bool = False,
        context: str = None,
    ) -> Any:
        """
        The entry point for triggering training

        Arguments:
            train_loader: the data for training
            val_loader: the data for validation
            patience_monitor: the instance of the patience monitor
            quiet: if False, no message on the console is reported
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Returns:
            The best model obtained during training
        """
        if context is None:
            context = ""
        else:
            context += "_"

        self._reset_metrics()

        t1 = time.perf_counter_ns()
        # for idx_epoch in range(epochs):
        train_metrics = self.train_one_epoch(train_loader, context=f"{context}train")
        t2 = time.perf_counter_ns()
        if self.tracker:
            self.tracker.track((t2 - t1) / 1E9, 'duration', context=dict(subset=f"{context}train"))

        # msg = f"epoch: {idx_epoch:3d} | "
        # msg += "train_loss: {loss:.6f}".format(loss=train_metrics["loss"])
        msg = ""
        for metric_name, metric_value in train_metrics.items():
            if "acc" not in metric_name:
                continue
            metric_name = f"train_{metric_name}"
            msg += f" | {metric_name}: {metric_value:5.1f}%"

        if val_loader:
            val_metrics, _ = self.test_loop(
                val_loader, with_reports=False, context=f"{context}val"
            )
            # msg += " | val_loss: {loss:.6f}".format(loss=val_metrics["loss"])
            for metric_name, metric_value in val_metrics.items():
                if "acc" not in metric_name:
                    continue
                metric_name = f"val_{metric_name}"
                msg += f" | {metric_name}: {metric_value:5.1f}%"

            if patience_monitor and patience_monitor(val_metrics):
                self.best_model = self.net.get_copy()
                metrics = patience_monitor.get_best_metrics()
                self._track_metrics(metrics, context=f"{context}val")
                msg += " | *"
        else:
            if patience_monitor and patience_monitor(train_metrics):
                self.best_model = self.net.get_copy()
                metrics = patience_monitor.get_best_metrics()
                self._track_metrics(metrics, context=f"{context}train")
                msg += " | *"
            else:
                pass

        if not quiet:
            self.log_msg(msg)

        if not quiet:
            self.log_msg("done")

        # self.net.set_state_dict(self.best_model)
        return self.xgboost_model

    def test_loop(
        self,
        data_loader: torch.utils.data.DataLoader,
        with_reports: bool = False,
        context: str = None,
    ) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """
        Run inference on a model (for testing or validation)

        Arguments:
            data_loader: the data to use
            with_reports: if True, compute classification report and confusion matrix
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            A tuple with two dictionaries. The first contains the metrics collected
            during inference; the second contains classification report (class_rep)
            and confusion matrix (conf_mtx) or is empty {} if their computation
            was not requested
        """
        if context is None:
            context = "val" if self._is_training else "test"

        t1 = time.perf_counter_ns()
        metrics = self._do_epoch(data_loader, track_preds_targets=True, context=context)
        t2 = time.perf_counter_ns()
        metrics['duration'] = (t2 - t1) / 1E9

        preds = metrics["preds"]
        targets = metrics["targets"]
        del metrics["preds"]
        del metrics["targets"]

        self._track_metrics(metrics, context=context)

        reports = {}
        if with_reports:
            reports = dict(
                class_rep=pd.DataFrame(
                    classification_report(targets, preds, output_dict=True)
                ).T,
                conf_mtx=pd.DataFrame(confusion_matrix(targets, preds)),
                preds=preds,
            )

        return metrics, reports


class ContrastiveLearningTrainer(SimpleTrainer):
    """A trainer designed for contrastive learning"""

    def __init__(
        self,
        net: backbone.BaseNet,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str = "cuda:0",
        deterministic: bool = True,
        tracker: aim.Run = None,
        logger: logging.Logger = None,
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            deterministic=deterministic,
            tracker=tracker,
            logger=logger,
        )

    @classmethod
    def prepare_net_for_train(
        cls, net: backbone.BaseNet, fname_weights: pathlib.Path = None
    ) -> backbone.BaseNet:
        """
        Clone a backbone.BaseNet a modifies to mask (via torch.nn.Identity)
        its .classifier and the last activation function of .features

        Arguments:
            net: the network to modify
            fname_weights: if provided, the weights are loaded into
                the network after the modification

        Return:
            A new instance of the input network with architecture
            modification required to run training contrastive learning
            training
        """
        new_net = backbone.clone_net(net)
        if not hasattr(net, "prepare_for_contrastivelearning"):
            raise RuntimeError(
                "Did not find a .prepare_for_contrativelearning() method in the network. Cannot adapt the network for training"
            )
        new_net.prepare_for_contrastivelearning(fname_weights)
        new_net = new_net.double()
        return new_net

    @classmethod
    def init_train(
        cls,
        net: backbone.BaseNet,
        optimizer: torch.optim.Optimizer = None,
        fname_weights: pathlib.Path = None,
    ) -> Tuple[backbone.BaseNet, torch.optim.Optimizer]:
        """
        Clones the input network and prepares it for contrastive learning,
        and instanciate a new optimized bounding it to the new network weights

        Arguments:
            net: the network to use
            optimizer: the optimizer to tuse
            fname_weights: if provided, the weights are loaded
                into the new network before returning it

        Return:
            A tuple with the new updated network and the related optimizer
        """
        new_net = cls.prepare_net_for_train(net, fname_weights)
        new_optimizer = None
        if optimizer:
            new_optimizer = _reset_optimizer(optimizer, new_net.parameters())
        return new_net, new_optimizer

    def _do_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        idx_epoch: int,
        context: str = "train",
    ) -> Dict[str, Any]:
        """Helper method invoked during training, validation
        and testing to perform forward (and backward) propagation

        Arguments:
            data_loader: an instance of a pytorch DataLoader
            idx_epoch: the current epoch
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            A dictionary of metrics collected
        """
        cum_metrics = defaultdict(float)
        num_batches = int(np.ceil(data_loader.dataset.df.shape[0] / data_loader.batch_size))

        #for batch_idx, (x, y) in enumerate(data_loader):
        for batch_idx, (x, y) in progress.track(enumerate(data_loader), description='', total=num_batches, transient=True):
            # x is a list with the multiple views
            x = torch.cat(x, dim=0).to(self.device)
            y = y.to(self.device)
            bsz = y.shape[0]

            # forward pass
            features = self.net(x)
            # apply L2 normalization
            features = torch.nn.functional.normalize(features, dim=1)

            # compute loss
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            metrics = self.criterion(features)
            loss = metrics["loss"]

            if self._is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            #print(".", end="", flush=True)

            for name, value in metrics.items():
                cum_metrics[name] += value.item()

        #print("\r" + " " * batch_idx + " ", end="", flush=True)
        #print("\r", end="", flush=True)
        metrics = {}
        for name, value in cum_metrics.items():
            value = value / (batch_idx + 1)
            if name.startswith("acc"):
                value *= 100
            metrics[name] = value
        self._track_metrics(metrics, epoch=idx_epoch, context=context)

        return metrics

    def train_one_epoch(self, train_loader, idx_epoch, context):
        """Set the internal model to train, calls _do_epoch() and return the obtained metrics"""
        self.net.train()
        self._is_training = True
        out = self._do_epoch(train_loader, idx_epoch, context)
        self._is_training = False
        return out

    def train_loop(
        self,
        epochs: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoaer = None,
        patience_monitor: PatienceMonitorLoss | PatienceMonitorAccuracy = None,
        quiet: bool = False,
        run_init_train: bool = True,
        context: str = None,
    ) -> backbone.BaseNet:
        """
        The entry point for triggering training

        Arguments:
            epochs: number of epochs to run
            train_loader: the data for training
            val_loader: the data for validation
            patience_monitor: the instance of the patience monitor
            quiet: if False, no message on the console is reported
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            The best model obtained during training
        """
        assert context is not None

        if run_init_train:
            self.net, self.optimizer = self.init_train(self.net, self.optimizer)
            self.net = self.net.to(self.device)
            x, y = next(iter(train_loader))
            self.log_msg(
                "\n======= net adapted for contrastive learning training =========",
                self.logger
            )
            #torchsummary.summary(self.net.float(), tuple(x[0][0].shape))
            utils.log_torchsummary(self.net.float(), tuple(x[0][0].shape), self.logger)
            self.net.double()

        t1 = time.perf_counter_ns()
        res = super().train_loop(
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            patience_monitor=patience_monitor,
            quiet=quiet,
            context=context,
        )
        t2 = time.perf_counter_ns()
        if self.tracker:
            self.tracker.track((t2 - t1) / 1E9, 'duration', context=dict(subset=f"{context}train"))

        return res

    def test_loop(
        self,
        data_loader: torch.utils.data.DataLoader,
        idx_epoch: ind = None,
        context: str = None,
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Run inference on a model (for testing or validation)

        Arguments:
            data_loader: the data to use
            idx_epoch: the current epoch
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            A tuple with two dictionaries. The first contains the metrics collected
            during inference; the second is an empty dictionary
        """
        self.net.eval()

        if context is None:
            context = "val" if self._is_training else "test"

        with torch.no_grad():
            t1 = time.perf_counter_ns()
            metrics = self._do_epoch(data_loader, idx_epoch, context=context)
            t2 = time.perf_counter_ns()
            metrics['duration'] = (t2 - t1) / 1E9

        # we return an empty report just to have
        # consistency with the return types
        # of SimpleTrainer
        reports = {}
        return metrics, reports


class MonolithicTrainer(SimpleTrainer):
    """A wrapper around SimpleTrainer and designed
    to be used in supervised classification scenarios
    """

    def __init__(
        self,
        net: backbone.BaseNet,
        optimizer: torch.optim.Optimizer = None,
        criterion: torch.nn.Module = nn.CrossEntropyLoss(),
        device: str = "cuda:0",
        deterministic: bool = True,
        tracker: aim.Run = None,
        logger: logging.Logger = None,
        reset_classifier: bool = False,
        num_classes: int = None,
        xgboost_model=None,
    ):
        """
        Arguments:
            net: the architecture to use
            optimizer: the optimizer to use
            criterion: the instance of the loss to use
            device: the device to use
            deterministic: see _make_deterministic()
            tracker: the AIM run on which register metrics
            logger: the logging reference
            reset_classifier: if True, the network is modified
                to have a new layer
            num_classes: number of units to use for the new
                classifier head
        """
        if reset_classifier:
            if num_classes is None:
                raise RuntimeError(
                    f"num_classes cannot be None when resetting the model head"
                )
            net = net.reset_classifier(num_classes)
        super().__init__(
            net=net,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            deterministic=deterministic,
            tracker=tracker,
            logger=logger,
        )


class SimCLRTrainer:
    """A trainer designed for SimCLR (https://arxiv.org/abs/2002.05709).
    Differently from the other trainers which are based on
    inheritancs, it is based on nesting a ContrastiveLearningTrainer
    object (for contrastive learning) and a MonolithicTrainer object (for finetune)
    """

    def __init__(
        self,
        pretrain_config: Dict[str, Any] = None,
        finetune_config: Dict[str, Any] = None,
        device: str = "cuda:0",
        deterministic: bool = True,
        tracker: aim.Run = None,
        logger: logging.Logger = None,
        xgboost_model=None,
    ):
        """
        Arguments:
            pretrain_config: a set of configuration required for pretraining.
                The dictionary should contain an "optimizer" and the related instance
                (None if missing) and a "loss_temperature" (0.07 if missing)
            finetune_config: a set of configuration required for finetune.
                The dictionary should contain an "optimizer" and the related instance
                (None if missing).
            device: the device for training and inference
            tracker: the AIM run for metric tracking
            logger: a logging object for console and file logging
        """
        self.tracker = tracker
        self.logger = logger
        self.device = device
        self.pretrain_config = pretrain_config
        self.pretrain_criterion = None
        self.pretrain_trainer = None
        self.pretrain_best_net = None

        self.finetune_config = finetune_config
        self.finetune_criterion = None
        self.finetune_trainer = None
        self.finetune_best_net = None

        trainer_params = dict(
            net=None,
            tracker=tracker,
            logger=logger,
            deterministic=False,
            device=device,
        )

        if pretrain_config is not None:
            self.pretrain_criterion = losses.SimCLRLoss(
                temperature=pretrain_config.get("loss_temperature", 0.07),
                base_temperature=pretrain_config.get("loss_base_temperature", 0.07),
                contrast_mode="all",
            )
            self.pretrain_trainer = ContrastiveLearningTrainer(
                optimizer=pretrain_config.get("optimizer", None),
                criterion=self.pretrain_criterion,
                **trainer_params,
            )

        if finetune_config is not None:
            self.finetune_criterion = nn.CrossEntropyLoss()
            self.finetune_trainer = MonolithicTrainer(
                optimizer=finetune_config.get("optimizer", None),
                criterion=self.finetune_criterion,
                **trainer_params,
            )

        if deterministic:
            _make_deterministic()

    @classmethod
    def init_pretrain(
        cls,
        net: backbone.BaseNet,
        optimizer: torch.optim.Optimizer = None,
        fname_weights: pathlib.Path = None,
    ) -> Tuple[backbone.BaseNet, Any]:
        """
        Clones the input network and prepares it for contrastive learning,
        and instanciate a new optimized bounding it to the new network weights

        Arguments:
            net: the network to use
            optimizer: the optimizer to tuse
            fname_weights: if provided, the weights are loaded
                into the new network before returning it

        Return:
            A tuple with the new updated network and the related optimizer
        """
        return ContrastiveLearningTrainer.init_train(net, optimizer, fname_weights)

    def pretrain_loop(
        self,
        net: backbone.BaseNet,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.Data.DataLoader = None,
        patience_monitor: PatienceMonitorLoss | PatienceMonitorAccuracy = None,
        epochs: int = 50,
        run_init_pretrain: bool = True,
        fname_weights: pathlib.Path = None,
        context: str = None,
    ) -> backbone.BaseNet:
        """
        The entry point for triggering training

        Arguments:
            net: the network to use
            train_loader: the data for training
            val_loader: the data for validation
            patience_monitor: the instance of the patience monitor
            epochs: number of epochs to run
            run_init_pretrain: if True, invokes .init_pretrain()
            fname_weights: a file with the weights to load after
                preparing the model for training
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            The best model obtained during training
        """
        if run_init_pretrain:
            net, optimizer = self.init_pretrain(
                net, self.pretrain_trainer.optimizer, fname_weights
            )
            params = list(net.parameters())
            expected_dtype = params[0].dtype
            self.pretrain_trainer.net = net.to(self.device)
            self.pretrain_trainer.optimizer = optimizer
            x, y = next(iter(train_loader))
            utils.log_msg("\n==== network adapted for pretrain ====", self.logger)
            #torchsummary.summary(net.float(), tuple(x[0][0].shape))
            utils.log_torchsummary(net.float(), tuple(x[0][0].shape), self.logger)
            net.double()

        self.pretrain_trainer.net = net
        t1 = time.perf_counter_ns()
        self.pretrain_best_model = self.pretrain_trainer.train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            patience_monitor=patience_monitor,
            epochs=epochs,
            run_init_train=False,
            context=context,
        )
        t2 = time.perf_counter_ns()
        duration = (t2 - t1) / 1E9
        if self.tracker:
            self.tracker.track(duration, "duration", context=dict(subset="pretrain-train"))
        return self.pretrain_best_model

    @classmethod
    def prepare_net_for_finetune(
        cls,
        net: backbone.BaseNet,
        num_classes: int = 5,
        fname_pretrain_weights: pathlib.Path = None,
        fname_finetune_weights: pathlib.Path = None,
    ) -> backbone.BaseNet:
        """
        Clone a backbone.BaseNet related to contrastive learning an
        prepare it for finetune. Specifically, the last linear layer
        of the newtwork (the projection layer) is masked (via a nn.Identity)
        and a classifier is added to the network

        Arguments:
            net: the network to modify
            num_classes: the number of units for the classifier

        Return:
            A new instance of the input network with architecture
            modification required to run training contrastive learning
            training
        """
        if not hasattr(net, "prepare_for_finetune"):
            raise RuntimeError(
                "Did not find a .prepare_for_finetune() method in the network. Cannot adapt the network for training"
            )
        new_net = backbone.clone_net(net)
        new_net.prepare_for_finetune(
            num_classes=num_classes,
            fname_pretrain_weights=fname_pretrain_weights,
            fname_finetune_weights=fname_finetune_weights,
        )
        return new_net

    @classmethod
    def init_finetune(
        cls,
        net: backbone.BaseNet,
        num_classes: int,
        fname_pretrain_weights: pathlib.Path = None,
        fname_finetune_weights: pathlib.Path = None,
        optimizer=None,
    ) -> Tuple[backbone.BaseNet, torch.optim.Optimizer]:
        """
        Initialize the network for finetuning adapting
        it from contrastive-learning. Specifically, the
        input network is the first modified for contrastive-learning,
        and then further adjusted for finetune.

        Arguments:
            net: the network to use
            num_classes: the number of classes for the classifier
            fname_pretrain_weights: if specified, the weights
                are loaded before adapting the network from
                pretraining
            fname_finetune_weights: if specified, the weights
                are loaded after adapting the network for
                finetune
        """
        new_net = cls.prepare_net_for_finetune(
            net, num_classes, fname_pretrain_weights, fname_finetune_weights
        )

        new_optimizer = None
        if optimizer:
            new_optimizer = _reset_optimizer(optimizer, new_net.classifier.parameters())

        return new_net, new_optimizer

    def finetune_loop(
        self,
        net: backbone.BaseNet,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        patience_monitor: PatienceMonitorLoss | PatienceMonitorAccuracy = None,
        epochs: int = 50,
        num_classes: int = 5,
        run_init_finetune: bool = True,
        fname_pretrain_weights: pathlib.PAth = None,
        context: str = None,
    ) -> backbone.BaseNet:
        """
        The entry point for triggering training

        Arguments:
            net: the network to use
            train_loader: the data for training
            val_loader: the data for validation
            patience_monitor: the instance of the patience monitor
            epochs: number of epochs to run
            num_classes: the number of units for the classifier
            run_init_finetune: if True, invokes .init_finetune()
            fname_pretrain_weights: a file with the weights to load after
                preparing the model for training
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            The best model obtained during training
        """
        if run_init_finetune:
            net, optimizer = self.init_finetune(
                net=net,
                num_classes=num_classes,
                fname_pretrain_weights=fname_pretrain_weights,
                optimizer=self.finetune_trainer.optimizer,
            )
            net = net.to(self.device)
            self.finetune_trainer.net = net
            self.finetune_trainer.optimizer = optimizer
            self.finetune_net = net

            x, y = next(iter(train_loader))
            # x can be a list if the underlining
            # dataset is multi-view
            if isinstance(x, list):
                x = x[0]
            utils.log_msg("\n==== network adapted for fine-tuning ====", self.logger)
            #torchsummary.summary(net.float(), tuple(x[0].shape))
            utils.log_torchsummary(net.float(), tuple(x[0].shape), self.logger)
            net.double()

        self.finetune_trainer.net = net
        t1 = time.perf_counter_ns()
        self.finetune_best_net = self.finetune_trainer.train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            patience_monitor=patience_monitor,
            epochs=epochs,
            context=context,
        )
        t2 = time.perf_counter_ns()
        duration = (t2 - t1) / 1E9
        if self.tracker:
            self.tracker.track(duration, "duration", context=dict(subset="finetune-train"))
        return self.finetune_best_net

    def finetune_test_loop(
        self,
        data_loader: torch.utils.data.DataLoader,
        idx_epoch: int = None,
        with_reports: bool = False,
        context: str = None,
    ) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """
        Run inference on a (supervised) model (for testing or validation)

        Arguments:
            data_loader: the data to use
            idx_epoch: the current epoch
            with_reports: if True, compute classification report and confusion matrix
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            A tuple with two dictionaries. The first contains the metrics collected
            during inference; the second contains classification report (class_rep)
            and confusion matrix (conf_mtx) or is empty {} if their computation
            was not requested
        """
        t1 = time.perf_counter_ns()
        res = self.finetune_trainer.test_loop(
            data_loader, idx_epoch, with_reports, context
        )
        t2 = time.perf_counter_ns()
        if self.tracker:
            self.tracker.track((t2 - t1) / 1E9, "duration", context=dict(subset="finetune-test"))
        return res

    def pretrain_test_loop(
        self,
        data_loader: torch.utils.data.DataLoader,
        idx_epoch: int = None,
        context: str = None,
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """
        Run inference on a (unsupervised) model (for testing or validation)

        Arguments:
            data_loader: the data to use
            idx_epoch: the current epoch
            context: a string (for tracking) for extra semantic (train, val, etc.)

        Return:
            A tuple with two dictionaries. The first contains the metrics collected
            during inference; the second contains classification report (class_rep)
            and confusion matrix (conf_mtx) or is empty {} if their computation
            was not requested
        """
        t1 = time.perf_counter_ns()
        res = self.pretrain_trainer.test_loop(data_loader, idx_epoch, context)
        t2 = time.perf_counter_ns()
        if self.tracker:
            self.tracker.track((t2 - t1) / 1E9, "duration", context=dict(subset="pretrain-test"))
        return res


METHOD_CLASSES = {
    "monolithic": MonolithicTrainer,
    "simclr": SimCLRTrainer,
    "xgboost": XGboostTrainer,
}


def trainer_factory(method: str, *args: Any, **kwargs: Any) -> Any:
    """Helper function to instanciate trainer objects

    Arguments:
        method: either "monolithic" or "simclr" or "xgboost"
        args: positional arguments to use when instanciating the class (if any)
        kwargs: key/value arguments to use when instanciating the class (if any)

    Return:
        a trainer object
    """
    return METHOD_CLASSES[method](*args, **kwargs)
