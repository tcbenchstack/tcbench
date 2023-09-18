"""
This module collects all network architectures.

A few convetion are used for networks

1. All networks are expected to inherity from 
the archetype class called BaseNet

2. BaseNet is composing layers by means
of two attribytes: .features is used
for feature extraction while .classifier 
corresponds to the final model head
"""
from __future__ import annotations

import numpy as np

from typing import Tuple, Any

Self = Any

from torch import nn
import xgboost as xgb

from copy import deepcopy

import torch
import pathlib
import sys


def have_same_layers_and_types(net1: BaseNet, net2: BaseNet) -> bool:
    """Compares to networks based on architecture

    Arguments:
        net1: a network
        net2: another netwokr

    Return:
        True if the two network have the same
        number of layers each with the same type
    """
    if len(net1.features) != len(net2.features):
        return False

    for block1, block2 in zip(net1.features, net2.features):
        if len(block1) != len(block2):
            return False
        for layer1, layer2 in zip(block1, block2):
            if type(layer1) != type(layer2):
                return False

    has_classifier1 = net1.classifier is None
    has_classifier2 = net2.classifier is None
    has_classifier = has_classifier1 + has_classifier2

    if has_classifier == 1:
        return False
    elif has_classifier == 0:
        return True
    return type(net1.classifier) == type(net2.classifier)


def are_equal(net1: BaseNet, net2: BaseNet) -> bool:
    """Compare two networks considering both architecture and weights

    Arguments:
        net1: a network
        net2: another netwokr

    Return:
        True if the two network have the same
        number of layers each with the same type
        and they have the same weights
    """
    if not have_same_layers_and_types(net1, net2):
        return False
    for (name1, weights1), (name2, weights2) in zip(
        net1.to("cpu").state_dict().items(), net2.to("cpu").state_dict().items()
    ):
        if not (weights1 == weights2).all():
            return False
    return True


def compute_features_size(input_size: Tuple[int], modules: List[nn.Module]) -> int:
    """Compute the number of units at the end of a chain of modules

    Attributes:
        input_size: the shape of the input tensor
        modules: a list of modules processing the input in sequence

    Return:
        The number of units in the ouput generated processing
        a input of the specified shape through the list of modules
    """
    x_dummy = torch.autograd.Variable(torch.ones(1, *input_size))
    for m in modules:
        x_dummy = m(x_dummy)
    return int(np.prod(x_dummy.shape))


def has_dropout_layer(module: nn.Module) -> bool:
    """Detect if the input module is a dropout layer
    or is a network containing a dropout layer
    """
    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
        return True
    return any(map(has_dropout_layer, module.children()))


def clone_net(net: BaseNet) -> BaseNet:
    """An utility function to clone a network

    Arguments:
        net: the network to clone

    Return:
        A new instance of the same network passed
        as input initialized with the same weights
    """
    curr_module = sys.modules[__name__]
    net_class = getattr(curr_module, net.__class__.__name__)
    new_net = net_class(*net._init_args, **net._init_kwargs)

    # the classifier might have been added after instanciation
    if net.classifier:
        new_net.reset_classifier(num_classes=net.num_classes)

    for idx in range(len(net.features)):
        new_block = new_net.features[idx]
        old_block = net.features[idx]
        for idx2 in range(len(new_block)):
            new_layer = new_block[idx2]
            old_layer = old_block[idx2]
            if isinstance(old_layer, nn.Identity) and not isinstance(
                new_layer, nn.Identity
            ):
                new_block[idx2] = nn.Identity()
    if isinstance(net.classifier, nn.Identity):
        new_net.classifier = nn.Identity()

    new_net.set_state_dict(net.get_copy())
    return new_net


class BaseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._init_args = args
        self._init_kwargs = kwargs
        self.features = None
        self.classifier = None

    def forward(self, x):
        if self.features is None:
            raise RuntimeError("self.features in None, i.e., no architecture defined")
        x = self.features(x)
        if self.classifier:
            x = self.classifier(x)
        return x

    def get_copy(self) -> Self:
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict) -> Self:
        """Set weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return self

    def load_weights(self, fname: pathlib.Path, drop_classifier: bool = False) -> Self:
        """Load into the network the weights stored into a file

        Argument:
            fname: the file storing the weights
            drop_classifier: if the network needs need to remove the classifier
        """
        state_dict = torch.load(fname)
        if drop_classifier:
            keys_to_drop = [key for key in state_dict if key.startswith("classifier")]
            for key in keys_to_drop:
                del state_dict[key]
        self.set_state_dict(state_dict)
        return self

    def save_weights(self, fname: pathlib.Path) -> None:
        """Store to file the weights of the network

        Arguments:
            fname: the file where to store the weights
        """
        fname = pathlib.Path(fname)
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        torch.save(self.state_dict(), fname)

    def _find_index_of_last_linear_layer(self) -> int:
        """Introspect the network architecture to identify
        the index of the last linear layer"""
        last_block = self.features[-1]
        for idx in range(len(last_block) - 1, -1, -1):
            layer = last_block[idx]
            if isinstance(layer, nn.Identity):
                continue
            elif isinstance(layer, nn.Linear):
                return idx
        raise RuntimeError(f"Didn't find any linear layer")

    def latent_space_dim(self) -> int:
        """Returns the number of units in the latent space
        of the model (i.e., the shape of the output generated
        be .features)
        """
        idx = self._find_index_of_last_linear_layer()
        layer = self.features[-1][idx]
        return list(layer.parameters())[-1].shape[0]

    def reset_classifier(self, num_classes) -> Self:
        """Recreate the classifier of the network

        Attributes:
            num_classes: the number of units for the new classifier layer
        """
        self.classifier = nn.Linear(self.latent_space_dim(), num_classes)
        self.initialize_weights(self.classifier)
        return self

    @property
    def num_classes(self) -> int:
        """The number of units for the classifier of the network"""
        if self.classifier is None or isinstance(self.classifier, nn.Identity):
            return None
        return list(self.classifier.parameters())[0].shape[0]

    def is_equal_to(self, other_net: BaseNet) -> bool:
        """Returns True if other_net is identical
        (architecture and weights) to the current
        network"""
        return are_equal(self, other_net)

    def initialize_weights(self, m: nn.Module) -> None:
        """Initialize the weights of the network using Kaiming He"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


# As from Fig.6 in Appendix A of the IMC22 paper
# which is also much better detailed in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9395707
class LeNet5FlowpicIMC22_Full(BaseNet):
    def __init__(self, num_classes=None, flowpic_dim=1500, with_dropout=True, **kwargs):
        super().__init__(
            num_classes=num_classes,
            flowpic_dim=flowpic_dim,
            with_dropout=with_dropout,
            **kwargs,
        )

        dropout_or_identity = [nn.Dropout2d(0.25), nn.Dropout1d(0.5)]
        if not with_dropout:
            dropout_or_identity = [nn.Identity(), nn.Identity()]

        conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 10, stride=5, padding=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 10, stride=5, padding=3),
            nn.ReLU(),
            dropout_or_identity[0],
            nn.MaxPool2d((2, 2)),
        )

        fc_size = compute_features_size((1, flowpic_dim, flowpic_dim), [conv1, conv2])

        fullyconnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_size, 64),
            nn.ReLU(),
            dropout_or_identity[1],
        )

        self.features = nn.Sequential()
        self.features.add_module("conv1", conv1)
        self.features.add_module("conv2", conv2)
        self.features.add_module("fullyconnected", fullyconnected)
        self.classifier = None
        if num_classes:
            self.reset_classifier(num_classes)

        self.apply(self.initialize_weights)


# As from Fig.7 in Appendix A of the IMC22 paper
# which corresponds to the original LeNet5
# as from Fig.2 of https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9395707
class LeNet5FlowpicIMC22_Mini(BaseNet):
    def __init__(
        self,
        num_classes=None,
        flowpic_dim=32,
        with_dropout=True,
        projection_layer_dim=None,
    ):
        super().__init__(
            num_classes=num_classes,
            flowpic_dim=flowpic_dim,
            with_dropout=with_dropout,
            projection_layer_dim=projection_layer_dim,
        )

        self.projection_layer_dim = projection_layer_dim

        dropout_or_identity = [nn.Dropout2d(0.25), nn.Dropout1d(0.5)]
        if not with_dropout:
            dropout_or_identity = [nn.Identity(), nn.Identity()]

        conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            dropout_or_identity[0],
            nn.MaxPool2d((2, 2)),
        )

        fc_size = compute_features_size((1, flowpic_dim, flowpic_dim), [conv1, conv2])

        fullyconnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            dropout_or_identity[1],
        )

        self.features = nn.Sequential()
        self.features.add_module("conv1", conv1)
        self.features.add_module("conv2", conv2)
        self.features.add_module("fullyconnected", fullyconnected)
        self.classifier = None
        if num_classes:
            self.reset_classifier(num_classes)

        self.apply(self.initialize_weights)

    def prepare_for_contrastivelearning(self, fname_weights=None):
        # new_net = clone_net(net)

        # Quote from Appendix A.2 of IMC22 paper
        # > For the representation extractor 𝑓()
        # we employed the 5 first layers of the CNN
        # architectures described in A.1 and
        # replaced the last 2 layers with 2 linear
        # layers sized 120 and 30
        fullyconnected = self.features[-1]
        fullyconnected[-1] = nn.Identity()  # masking dropout
        fullyconnected[-3] = nn.Linear(120, 120)  # replacing 84 with 120

        if self.projection_layer_dim is None:
            self.projection_layer_dim = 30
        self.reset_classifier(self.projection_layer_dim)  # final projection

        if fname_weights:
            self.load_weights(fname_weights)
        else:
            self.initialize_weights(fullyconnected[-3])
            self.initialize_weights(self.classifier)
        return self

    def prepare_for_finetune(
        self, num_classes, fname_pretrain_weights=None, fname_finetune_weights=None
    ):
        self.prepare_for_contrastivelearning(fname_pretrain_weights)

        fullyconnected = self.features[-1]
        fullyconnected[-1] = nn.Identity()
        fullyconnected[-2] = nn.Identity()
        fullyconnected[-3] = nn.Identity()

        self.reset_classifier(num_classes)

        if fname_finetune_weights:
            self.load_weights(fname_finetune_weights)
        else:
            self.initialize_weights(self.classifier)
        return self


class XGboost_model(object):
    def __init__(self, random_state=42):
        self.xgb_model = xgb.XGBClassifier(
            objective="multi:softprob", random_state=random_state
        )

    def fit(self, X, y):
        self.xgb_model.fit(X, y)

    def predict(self, X):
        return self.xgb_model.predict(X)

    def save_model(self, path):
        self.xgb_model.save_model(path)

    def load_model(path):
        new_model = XGboost_model(0)
        new_model.xgb_model.load_model(path)
        return new_model


def net_factory(
    num_classes: int = 5,
    flowpic_dim: int = 32,
    with_dropout: bool = True,
    projection_layer_dim: int = None,
) -> BaseNet:
    """An utilify function to create Flowpic-related networks

    Arguments:
        num_classes: the number of classes for the classifier
        flowpic_dim: the resolution of the flowpic representation
        with_dropout: if False, the network use nn.Identity to mask out dropout layers
        projection_layer_dim: the number of units for the SimCLR projection layer

    Return:
        the instanciated network
    """
    kwargs = dict(
        num_classes=num_classes,
        flowpic_dim=flowpic_dim,
        with_dropout=with_dropout,
        projection_layer_dim=projection_layer_dim,
    )
    if flowpic_dim in (32, 64):
        return LeNet5FlowpicIMC22_Mini(**kwargs)
    return LeNet5FlowpicIMC22_Full(**kwargs)


def xgboost_factory(random_state=42):
    kwargs = dict(random_state=random_state)
    return XGboost_model(**kwargs)
