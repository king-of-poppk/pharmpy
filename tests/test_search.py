import numpy as np

from pharmpy.search.algorithms import exhaustive
from pharmpy.search.rankfuncs import aic, bic, ofv


class DummyResults:
    def __init__(self, ofv=None, aic=None, bic=None):
        self.ofv = ofv
        self.aic = aic
        self.bic = bic


class DummyModel:
    def __init__(self, name, **kwargs):
        self.name = name
        self.modelfit_results = DummyResults(**kwargs)

    def copy(self):
        return DummyModel(self.name, ofv=self.modelfit_results.ofv)


def test_ofv():
    run1 = DummyModel("run1", ofv=0)
    run2 = DummyModel("run2", ofv=-1)
    run3 = DummyModel("run3", ofv=-14)
    res = ofv(run1, [run2, run3])
    assert [run3] == res

    run4 = DummyModel("run4", ofv=2)
    run5 = DummyModel("run5", ofv=-2)
    res = ofv(run1, [run2, run3, run4, run5], cutoff=2)
    assert [run3, run5] == res


def test_aic():
    run1 = DummyModel("run1", aic=0)
    run2 = DummyModel("run2", aic=-1)
    run3 = DummyModel("run3", aic=-14)
    res = aic(run1, [run2, run3])
    assert [run3] == res


def test_bic():
    run1 = DummyModel("run1", bic=0)
    run2 = DummyModel("run2", bic=-1)
    run3 = DummyModel("run3", bic=-14)
    res = bic(run1, [run2, run3])
    assert [run3] == res


def test_exhaustive():
    base = DummyModel("run1", ofv=0)

    def do_nothing(model):
        return model

    trans = [do_nothing]
    res = exhaustive(base, trans, do_nothing, ofv)
    assert list(res['rank']) == [np.nan]

    def set_ofv(models):
        for i, model in enumerate(models):
            model.modelfit_results.ofv = -4 - i * 2

    res = exhaustive(base, trans, set_ofv, ofv)
    assert len(res) == 1
    assert list(res['dofv']) == [4]
