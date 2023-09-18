import pytest
import pathlib

from tcbench.modeling import backbone
from tcbench.modeling.backbone import LeNet5FlowpicIMC22_Mini
from tcbench.modeling.methods import ContrastiveLearningTrainer

@pytest.mark.parametrize(
    "net1, net2, expected",
    [
        (LeNet5FlowpicIMC22_Mini(), LeNet5FlowpicIMC22_Mini(), True),
        (LeNet5FlowpicIMC22_Mini(), LeNet5FlowpicIMC22_Mini(num_classes=5), False),
        (
            LeNet5FlowpicIMC22_Mini(),
            ContrastiveLearningTrainer.prepare_net_for_train(LeNet5FlowpicIMC22_Mini()),
            False,
        ),
        (
            LeNet5FlowpicIMC22_Mini(),
            ContrastiveLearningTrainer.init_train(LeNet5FlowpicIMC22_Mini(), None)[0],
            False,
        ),
    ],
)
def test_have_same_layers_and_types(net1, net2, expected):
    assert backbone.have_same_layers_and_types(net1, net2) == expected


@pytest.mark.parametrize(
    "num_classes1, num_classes2",
    [
        (5, 5),
        (None, 5),
        (5, None),
        (None, None),
    ],
)
def test_have_same_layers_and_types_after_reloading_from_file(
    tmp_path, num_classes1, num_classes2
):
    net1 = LeNet5FlowpicIMC22_Mini(num_classes=num_classes1)
    net1 = ContrastiveLearningTrainer.prepare_net_for_train(net1)
    net1.save_weights(tmp_path / "weights.pt")

    net2 = LeNet5FlowpicIMC22_Mini(num_classes=num_classes2)
    net2, _ = ContrastiveLearningTrainer.init_train(net2, None, tmp_path / "weights.pt")
    assert backbone.have_same_layers_and_types(net1, net2)


@pytest.mark.parametrize(
    "net1, net2, expected",
    [
        (LeNet5FlowpicIMC22_Mini(), LeNet5FlowpicIMC22_Mini(), False),
        (LeNet5FlowpicIMC22_Mini(), LeNet5FlowpicIMC22_Mini(num_classes=5), False),
    ],
)
def test_are_equal(net1, net2, expected):
    assert backbone.are_equal(net1, net2) == expected


@pytest.mark.parametrize(
    "num_classes1, num_classes2",
    [
        (5, 5),
        (None, 5),
        (5, None),
        (None, None),
    ],
)
def test_are_equal_after_reloading_from_file(tmp_path, num_classes1, num_classes2):
    net1 = LeNet5FlowpicIMC22_Mini(num_classes=num_classes1)
    net1 = ContrastiveLearningTrainer.prepare_net_for_train(net1)
    net1.save_weights(tmp_path / "weights.pt")

    net2 = LeNet5FlowpicIMC22_Mini(num_classes=num_classes2)
    net2, _ = ContrastiveLearningTrainer.init_train(net2, None, tmp_path / "weights.pt")
    assert backbone.are_equal(net1, net2)


@pytest.mark.parametrize(
    "net",
    [
        LeNet5FlowpicIMC22_Mini(),
        LeNet5FlowpicIMC22_Mini(num_classes=5),
    ],
)
def test_clone_net(net):
    new_net = backbone.clone_net(net)
    assert backbone.are_equal(net, new_net)
    assert id(net) != id(new_net)
