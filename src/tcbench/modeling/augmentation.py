"""
This module contains the logic to generate Flowpic data representations
and apply augmentations on either flowpic or raw time series.

Each augmentation is handled as a subclass of Augmentation
which is a callable object.

Moreover, each augmentation is designed to have its own 
random generator and a set of hyperparameters which which 
generate the parameters of an augmentation. Differently
from pytorch APIs, this enables
visibility on the set of params used for an augmentation
(use .get_params())
"""
from __future__ import annotations

import numpy as np
import torchvision.transforms as T

from typing import Tuple, Dict
from numpy.typing import NDArray

import torch
import abc
import numbers

MAX_PACKET_SIZE = 1500


def get_flowpic(
    timetofirst: NDArray,
    pkts_size: NDArray,
    dim: int = 32,
    max_block_duration: int = 15,
) -> NDArray:
    """Generate a Flowpic from time series

    Arguments:
        timetofirst: time series (in seconds) of the intertime between a packet and the first packet of the flow
        pkts_size: time series of the packets size
        dim: pixels size of the output representation
        max_block_duration: how many seconds of the input time series to process

    Return:
        a 2d numpy array encoding a flowpic
    """
    indexes = np.where(timetofirst < max_block_duration)[0]

    timetofirst = timetofirst[indexes]
    pkts_size = np.clip(pkts_size[indexes], a_min=0, a_max=MAX_PACKET_SIZE)

    timetofirst_norm = (timetofirst / max_block_duration) * dim
    pkts_size_norm = (pkts_size / MAX_PACKET_SIZE) * dim
    bins = range(dim + 1)
    mtx, _, _ = np.histogram2d(x=pkts_size_norm, y=timetofirst_norm, bins=(bins, bins))

    # Quote from Sec.2.1 of the IMC22 paper
    # > If more than max value (255) packets of
    # > a certain size arrive in a time slot,
    # > we set the pixel value to max value
    mtx = np.clip(mtx, a_min=0, a_max=255).astype("uint8")
    return mtx


def numpy_to_tensor(mtx: np.array) -> torch.Tensor:
    """Transforms a 2d numpy array into a 3d Tensor adding an extra dimension"""
    if len(mtx.shape) == 2:
        return torch.from_numpy(np.expand_dims(mtx, 0))
    return torch.from_numpy(mtx)


def tensor_to_numpy(tensor: torch.Tensor) -> np.array:
    """Transforms a 3d tensor into a 2d numpy array removing the first dimension"""
    mtx = tensor.numpy()
    if mtx.ndim > 2 and mtx.shape[0] == 1:
        return mtx.squeeze()
    return mtx


def _copy_and_delete_from_numpy_array(array: np.array, indexes: np.array) -> np.array:
    """Helper function to remove elements from a numpy array"""
    return np.delete(np.copy(array), indexes)


class Augmentation:
    """
    Base class for augmentation functions

    Attributes:
        rng: the numpy random generator used for sampling parameters
        hyper_params: a dictionary of hypter parameters to set up the
            sampling of the augmentation parameters
        paramgs: a dictionary with the latest parameters generated
            for a transformation
    """

    def __init__(
        self,
        rng: np.random.Generator,
        randomize_at_every_call: bool = True,
        **hyper_params: Dict[str, Any],
    ):
        """
        Arguments:
            rng: a numpy random number generator
            randomize_at_every_call: if True, the parameters for the augmentation
                are generated at each call
        """
        self.rng = rng
        self.hyper_params = hyper_params
        self.randomize_at_every_call = randomize_at_every_call
        self.is_first_call = True
        self.params = {}

    def update_params(self) -> None:
        pass

    def get_params(self) -> Dict[str, Any]:
        return self.params


# Quote from Sec.3.2
# > Rotate: [...] we tested this augmentation with angle
# > rotation uniformly distributed in the range [−10, 10]
# > degrees
class AugmentationRotate(Augmentation):
    """
    An augmentation for random rotation

    The rotation can be configured passing a "min_degree" and "max_degree"
    as hyper parameters (-10, 10) by default.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyper_params.setdefault("min_degree", -10)
        self.hyper_params.setdefault("max_degree", 10)
        self.params = dict(angle=0)

    def update_params(self) -> None:
        self.params["angle"] = self.rng.uniform(
            self.hyper_params["min_degree"], self.hyper_params["max_degree"]
        )

    def __call__(self, mtx: np.array) -> np.array:
        if self.is_first_call or self.randomize_at_every_call:
            self.update_params()
        self.is_first_call = False
        tensor = numpy_to_tensor(mtx)

        # .rotate() is counter-clockwise (which
        # is counter intuitive, so we change the sign
        tensor = T.functional.rotate(tensor, -self.params["angle"])
        return tensor_to_numpy(tensor)


class AugmentationHorizontalFlip(Augmentation):
    """
    An augmentation for static horizontal flip
    """

    def __init__(self, *args, **kwargs):
        super().__init__(rng=None)

    def __call__(self, mtx: np.array) -> np.array:
        tensor = numpy_to_tensor(mtx)
        tensor = T.functional.hflip(tensor)
        return tensor_to_numpy(tensor)


# Quote from Sec.3.2
# > ColorJitter: [...] The parameters we chose
# > were brightness = 0.8, contrast = 0.8,
# > saturation = 0.8, and hue = 0.2.
class AugmentationColorJitter(Augmentation):
    """
    An augmentation for applying random modification of
    brightness, saturation, contrast and hue
    """

    # Code inspired by the default logic of torchvision.transformations.ColorJitter
    # https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#ColorJitter
    def __init__(
        self,
        rng,
        randomize_at_every_call=True,
        brightness=0.8,
        saturation=0.8,
        contrast=0.8,
        hue=0.2,
    ):
        super().__init__(
            rng,
            randomize_at_every_call,
            brightness=brightness,
            saturation=saturation,
            contrast=contrast,
            hue=hue,
        )
        (
            self.hyper_params["min_brightness"],
            self.hyper_params["max_brightness"],
        ) = self._check_input(self.hyper_params["brightness"], "brightness")
        (
            self.hyper_params["min_saturation"],
            self.hyper_params["max_saturation"],
        ) = self._check_input(self.hyper_params["saturation"], "saturation")
        (
            self.hyper_params["min_contrast"],
            self.hyper_params["max_contrast"],
        ) = self._check_input(self.hyper_params["contrast"], "contrast")
        self.hyper_params["min_hue"], self.hyper_params["max_hue"] = self._check_input(
            self.hyper_params["hue"],
            "hue",
            center=0,
            bound=(-0.5, 0.5),
            clip_first_on_zero=False,
        )
        self.params = dict()

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(
                f"{name} values should be between {bound}, but got {value}."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    def update_params(self) -> None:
        self._order = self.rng.permutation(4)
        self.params["brightness"] = self.rng.uniform(
            self.hyper_params["min_brightness"], self.hyper_params["max_brightness"]
        )
        self.params["saturation"] = self.rng.uniform(
            self.hyper_params["min_saturation"], self.hyper_params["max_saturation"]
        )
        self.params["contrast"] = self.rng.uniform(
            self.hyper_params["min_contrast"], self.hyper_params["max_contrast"]
        )
        self.params["hue"] = self.rng.uniform(
            self.hyper_params["min_hue"], self.hyper_params["max_hue"]
        )

    def __call__(self, mtx: np.array) -> np.array:
        if self.is_first_call or self.randomize_at_every_call:
            self.update_params()

        tensor = numpy_to_tensor(mtx)
        for idx in self._order:
            if idx == 0:
                tensor = T.functional.adjust_brightness(
                    tensor, self.params["brightness"]
                )
            elif idx == 1:
                tensor = T.functional.adjust_saturation(
                    tensor, self.params["saturation"]
                )
            elif idx == 2:
                tensor = T.functional.adjust_contrast(tensor, self.params["contrast"])
            else:
                tensor = T.functional.adjust_hue(tensor, self.params["hue"])
        self.is_first_call = False
        return tensor_to_numpy(tensor)


# Quote from Sec. 3.2
# > We simulate a simple
# > loss process where we delete all packets in the time
# > interval [t-Delta_t; t+Delta_t] is randomly selected in the
# > session interval, and Delta_t = 0.1seconds.
class AugmentationPacketLoss(Augmentation):
    """
    An augmentation for applying packet loss (according
    to the logic of IMC22 paper "A Few Shots Traffic Classification with mini-FlowPic
    Augmentations"
    """

    def __init__(self, rng, randomize_at_every_call=True, delta_time=0.1):
        super().__init__(rng, randomize_at_every_call, delta_time=delta_time)

    # Note: differently from the other augmentations,
    # we intentionally sample a new t
    # at every call as to shape the augmentation
    # in function of the input. This is due to
    # possible padding in the flowpic: if the
    # traffic is occurring only at the beginning
    # of the flow, sampling uniformly t from a large
    # window (as from above quote) unlikely alter
    # the input data
    def __call__(
        self, timetofirst: np.array, pkts_size: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        session_time = timetofirst[-1]
        random_t_in_session = self.rng.uniform(low=0.0, high=session_time)
        min_ts = random_t_in_session - self.hyper_params["delta_time"]
        max_ts = random_t_in_session + self.hyper_params["delta_time"]
        self.params["min_ts"] = min_ts
        self.params["max_ts"] = max_ts

        indexes_to_drop = np.where((timetofirst >= min_ts) & (timetofirst <= max_ts))[0]
        new_timetofirst = _copy_and_delete_from_numpy_array(
            timetofirst, indexes_to_drop
        )
        new_pkts_size = _copy_and_delete_from_numpy_array(pkts_size, indexes_to_drop)
        return new_timetofirst, new_pkts_size, indexes_to_drop


# Quote from Sec. 3.2
# > We simulate a Time Shift by adding a constant
# > b in [b_min, b_max] to the arrival time of each packet
# > and rebuild FlowPic. The constant b is uniformly sampled from
# > [b_min, b_max] so t_new = t_old + b we chose
# > b_min = -1 and b_max = 1 seconds
class AugmentationTimeShift(Augmentation):
    """
    An augmentation for applying time shift (according
    to the logic of IMC22 paper "A Few Shots Traffic Classification with mini-FlowPic
    Augmentations"
    """

    def __init__(self, rng, randomize_at_every_call=True, delta_time=1):
        super().__init__(rng, randomize_at_every_call, delta_time=delta_time)

    def update_params(self):
        self.params["shift"] = self.rng.uniform(
            -self.hyper_params["delta_time"], self.hyper_params["delta_time"]
        )

    def __call__(
        self, timetofirst: np.array, pkts_size: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        if self.is_first_call or self.randomize_at_every_call:
            self.update_params()
        new_timetofirst = np.copy(timetofirst) + self.params["shift"]
        indexes = np.where(new_timetofirst < 0)[0]
        new_pkts_size = pkts_size
        if len(indexes) > 0:
            new_timetofirst = new_timetofirst[len(indexes) :]
            new_pkts_size = np.copy(new_pkts_size)[len(indexes) :]
        return new_timetofirst, new_pkts_size, indexes


# Quote from Sec 3.2
# > We simulate a change in the RTT by
# > multiply the arrival time of each packets
# > by a factor alpha and rebuild
# > the FlowPic. The factor alpha is uniformly selected from
# > [alpha_min, alpha_max], namely RTT_new = alpha * RTT_old.
# > We chose alpha_min = 0.5 and alpha_max = 1.5
class AugmentationChangeRTT(Augmentation):
    """
    An augmentation for applying change rtt (according
    to the logic of IMC22 paper "A Few Shots Traffic Classification with mini-FlowPic
    Augmentations"
    """

    def __init__(self, rng, randomize_at_every_call=True, min_alpha=0.5, max_alpha=1.5):
        super().__init__(
            rng, randomize_at_every_call, min_alpha=min_alpha, max_alpha=max_alpha
        )
        self.params = dict()

    def update_params(self):
        self.params["alpha"] = self.rng.uniform(
            self.hyper_params["min_alpha"], self.hyper_params["max_alpha"]
        )

    def __call__(
        self, timetofirst: np.array, pkts_size: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        if self.is_first_call or self.randomize_at_every_call:
            self.update_params()
        new_timetofirst = np.copy(timetofirst) * self.params["alpha"]
        return new_timetofirst, pkts_size, None


AUGMENTATION_CLASSES = {
    "rotate": AugmentationRotate,
    "horizontalflip": AugmentationHorizontalFlip,
    "colorjitter": AugmentationColorJitter,
    "packetloss": AugmentationPacketLoss,
    "timeshift": AugmentationTimeShift,
    "changertt": AugmentationChangeRTT,
}

AUGMENTATION_DEFAULT_HPARAMS = {
    "rotate": dict(degree=10),
    "horizontalflip": dict(),
    "colorjitter": dict(brightness=0.8, saturation=0.8, contrast=0.8, hue=0.2),
    "packetloss": dict(delta_time=0.1),
    "timeshift": dict(delta_time=1),
    "changertt": dict(min_alpha=0.5, max_alpha=1.5),
}


def augmentation_factory(
    aug_name: str, rng: np.random.Generator, hyper_params: Dict[str, Any]
) -> Augmentation:
    """A factory method to create instances of augmentation classes"""
    if aug_name not in AUGMENTATION_CLASSES:
        return None

    if hyper_params is None or len(hyper_params) == 0:
        hyper_params = AUGMENTATION_DEFAULT_HPARAMS.get(aug_name, {}).copy()

    return AUGMENTATION_CLASSES[aug_name](rng, **hyper_params)


def apply_augmentation(
    aug_name: str, aug: Augmentation, **kwargs: Dict[str, Any]
) -> Dict[str, NDArray]:
    """Applies a transformation to the input features"""
    if aug_name in {"rotate", "horizontalflip", "colorjitter"}:
        kwargs["flowpic"] = aug(kwargs["flowpic"])
    elif aug_name in {"timeshift", "packetloss", "changertt"}:
        new_timetofirst, new_pkts_size, _ = aug(
            kwargs["timetofirst"], kwargs["pkts_size"]
        )
        kwargs["timetofirst"] = new_timetofirst
        kwargs["pkts_size"] = new_pkts_size
        kwargs["flowpic"] = get_flowpic(new_timetofirst, new_pkts_size)
    return kwargs
