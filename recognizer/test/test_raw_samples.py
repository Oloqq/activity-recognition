from ..raw_samples import *
import pandas as pd
from pandas import DataFrame as DF
from math import nan
from deepdiff import DeepDiff

def test_split_based_on_time():
    df = DF({
        "dt_ms": [nan, 10, 20, 10, 10, 20, 10, 10, 10, 20]
    })
    split = split_for_continuity(df)
    for df in split:
        print(df)
        print()

    expected = [
        [nan, 10.0],
        [20.0, 10.0, 10.0],
        [20.0, 10.0, 10.0, 10.0],
        [20.0],
    ]

    assert len(split) == len(expected)
    tmp = [s.dt_ms.tolist() for s in split]
    dd = DeepDiff(tmp, expected)
    assert len(dd.items()) == 1 # because nan != nan
