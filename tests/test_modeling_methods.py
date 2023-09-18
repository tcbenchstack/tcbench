import pytest

import torch

from tcbench.modeling import backbone, methods
from tcbench.modeling.backbone import LeNet5FlowpicIMC22_Mini


@pytest.mark.parametrize(
    "net, optimizer_class",
    [
        (LeNet5FlowpicIMC22_Mini(), None),
        (LeNet5FlowpicIMC22_Mini(), torch.optim.Adam),
    ],
)
def test_simclr_init_pretrain(net, optimizer_class):
    net = LeNet5FlowpicIMC22_Mini()

    optimizer = None
    if optimizer_class:
        optimizer = optimizer_class(net.parameters(), lr=0.001)

    new_net1, optimizer1 = methods.ContrastiveLearningTrainer.init_train(net, optimizer)
    new_net2, optimizer2 = methods.SimCLRTrainer.init_pretrain(net, optimizer)
    # the two networks need to have the same architecture
    # but weights are not be the same overall because
    # new layers are added

    assert backbone.have_same_layers_and_types(new_net1, new_net2)
    assert id(new_net1) != id(new_net2)

    # compare first convolutional layer
    assert (list(new_net1.parameters())[0] == list(new_net2.parameters())[0]).all()

    # compare last linear layer weights (bias is 0)
    assert (list(new_net1.parameters())[-2] != list(new_net2.parameters())[-2]).any()

    if optimizer:
        assert id(optimizer1) != id(optimizer2)
        assert id(optimizer) != id(optimizer1)
        assert id(optimizer) != id(optimizer2)
        params1 = optimizer1.param_groups[0]["params"]
        params2 = optimizer2.param_groups[0]["params"]
        assert len(params1) == len(params2)
        assert (params1[0] == params2[0]).all()


@pytest.mark.parametrize(
    "net, optimizer_class",
    [
        (LeNet5FlowpicIMC22_Mini(), None),
        (LeNet5FlowpicIMC22_Mini(), torch.optim.Adam),
    ],
)
def test_simclr_init_finetune(net, optimizer_class):
    net = LeNet5FlowpicIMC22_Mini()

    optimizer = None
    if optimizer_class:
        optimizer = optimizer_class(net.parameters(), lr=0.001)

    new_net, new_optimizer = methods.SimCLRTrainer.init_finetune(
        net, optimizer=optimizer, num_classes=5
    )
    assert not new_net.is_equal_to(net)
    assert new_net.classifier is not None
    if optimizer:
        assert len(new_optimizer.param_groups[0]["params"]) == 2
        for p1, p2 in zip(
            new_net.classifier.parameters(), new_optimizer.param_groups[0]["params"]
        ):
            assert (p1 == p2).all()
