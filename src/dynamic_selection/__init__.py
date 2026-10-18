"""Case-specific dynamic model selection utilities.

The package is intentionally separate from the historical validation-routing
code.  It is activated only when a selector artifact trained without final
evaluation labels is available.
"""

from .selector import (
    build_selector_artifact,
    load_selector_artifact,
    select_model,
    train_competence_models,
)

__all__ = [
    "build_selector_artifact",
    "load_selector_artifact",
    "select_model",
    "train_competence_models",
]
