import numpy as np
from hypothesis import given, settings, example
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import pytest
from sparse import COO

from affinis import associations as aff

# def test_version():
#     assert __version__ == "0.1.0"


from hypothesis import settings

settings.register_profile("no_deadlines", deadline=None)
settings.load_profile("no_deadlines")

affinis_funcs = [
    aff.binary_cosine_similarity,
    aff.coocur_prob,
    aff.ochiai,
    aff.high_salience_skeleton,
    aff.mutual_information,
    aff.odds_ratio,
    aff.yule_q,
    aff.yule_y,
    aff.chow_liu,
    aff.resource_project,
    aff.doubly_stochastic_filter,
    aff.hyperbolic_project
]

@st.composite  
def make_shapes(draw):
    return draw(hnp.array_shapes(max_dims=2, min_dims=2, min_side=2, max_side=100))


@st.composite
def make_bools(draw, shape=(2,5)): 
    arr = draw(hnp.arrays(
        # bool,
        int,
        shape,
        elements=st.just(True),
        fill=st.just(False),
    ))
    return arr


# @example((np.array([[0,1,0,1],[0,1,1,0]]))).xfail() #ints aren't bools
@pytest.mark.parametrize(
    "assoc_func",
    affinis_funcs
                                  
)
@given(data=st.data())
def test_assoc_shapes(assoc_func, data):
    X = COO.from_numpy(data.draw(make_bools()))
    D = assoc_func(X)
    assert D.shape == (X.shape[-1], X.shape[-1])


@pytest.mark.parametrize(
    "fp_mode",
    ["edge-prob", "interaction", "forest-max", "counts"]
                                  
)
@given(data=st.data())
@settings(deadline=None)
def test_fp_shapes(fp_mode, data):
    X = data.draw(make_bools())
    D = aff.forest_pursuit(X, mode=fp_mode)
    assert D.shape == (X.shape[-1], X.shape[-1])

