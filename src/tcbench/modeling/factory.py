from __future__ import annotations

from typing import Iterable, Dict

from tcbench.modeling import (
    #MLMODEL_NAME,
    MODELING_METHOD_NAME
)
from tcbench.modeling.ml import (
    classifiers as mlclassifiers,
    core as mlcore
)

MODEL_NAME_TO_CLASS = {
    MODELING_METHOD_NAME.XGBOOST: mlclassifiers.XGBoostClassifier
}


def _get_model_class(name: MODELING_METHOD_NAME) -> mlcore.MLModel:
    return MODEL_NAME_TO_CLASS.get(name, None)

def mlmodel_factory(
    name: MODELING_METHOD_NAME,
    labels: Iterable[str],
    feature_names: Iterable[str],
    seed: int = 1,
    **hyperparams: Dict[str, Any]
) -> mlcore.MLModel:
    cls = _get_model_class(name)
    if cls:
        return cls(
            labels=labels,
            feature_names=feature_names,
            seed=seed,
            **hyperparams
        )
    raise RuntimeError(f"ModelNotFound: unrecognized model name {name}")
