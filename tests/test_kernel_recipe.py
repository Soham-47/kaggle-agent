import numpy as np
import pandas as pd
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SPEC = spec_from_file_location(
    "kernel_recipe", Path(__file__).parents[1] / "competitions/rsna_knee/pipeline/kernel_recipe.py"
)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_grouped_cv_validate_handles_original_index_labels():
    namespace = {}
    exec(_MODULE.KERNEL_RECIPE_SOURCE, namespace)

    ids = [f"study-{i}" for i in range(20)]
    train = pd.DataFrame(
        {
            "StudyInstanceUID": ids,
            "ACL": [i % 2 for i in range(20)],
        },
        index=np.arange(100, 120),
    )
    sft = pd.DataFrame({"feature": np.arange(20, dtype=float)}, index=ids)

    result = namespace["grouped_cv_validate"](train, sft)

    assert result is not None
