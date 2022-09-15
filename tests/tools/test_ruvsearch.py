import pytest

from pharmpy.modeling import remove_covariance_step
from pharmpy.tools import run_tool
from pharmpy.tools.ruvsearch.results import psn_resmod_results
from pharmpy.tools.ruvsearch.tool import create_workflow
from pharmpy.workflows import Workflow


def test_resmod_results(testdata):
    res = psn_resmod_results(testdata / 'psn' / 'resmod_dir1')
    assert list(res.cwres_models['dOFV']) == [
        -1.31,
        -3.34,
        -13.91,
        -18.54,
        -8.03,
        -4.20,
        -0.25,
        -1.17,
        -0.00,
        -0.09,
        -2.53,
        -3.12,
        -3.60,
        -25.62,
        -7.66,
        -0.03,
        -5.53,
    ]


def test_resmod_results_dvid(testdata):
    res = psn_resmod_results(testdata / 'psn' / 'resmod_dir2')
    df = res.cwres_models
    assert df['dOFV'].loc[1, '1', 'autocorrelation'] == -0.74
    assert df['dOFV'].loc[1, 'sum', 'tdist'] == -35.98


def test_create_workflow():
    assert isinstance(create_workflow(), Workflow)


@pytest.mark.parametrize(
    ('model_path', 'groups', 'p_value', 'skip'),
    [
        (
            None,
            3.1415,
            0.05,
            None,
        ),
        (
            None,
            4,
            1.01,
            None,
        ),
        (
            None,
            4,
            0.05,
            'ABC',
        ),
        (
            None,
            4,
            0.05,
            1,
        ),
        (
            None,
            4,
            0.05,
            ('IIV_on_RUV', 'power', 'time'),
        ),
    ],
)
def test_create_workflow_raises(
    load_model_for_test,
    testdata,
    model_path,
    groups,
    p_value,
    skip,
):

    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    with pytest.raises((ValueError, TypeError)):
        create_workflow(
            groups=groups,
            p_value=p_value,
            skip=skip,
            model=model,
        )


def test_run_tool_raises(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    remove_covariance_step(model)

    with pytest.raises(TypeError, match="Invalid groups"):
        run_tool('ruvsearch', model, groups=4.5, p_value=0.05, skip=[])

    with pytest.raises(ValueError, match="Invalid p_value"):
        run_tool('ruvsearch', model, groups=4, p_value=1.2, skip=[])

    with pytest.raises(ValueError, match="Invalid skip"):
        run_tool(
            'ruvsearch',
            model,
            groups=4,
            p_value=0.05,
            skip=['tume_varying', 'RUV_IIV', 'powder'],
        )

    with pytest.raises(ValueError, match="Please check mox3.mod"):
        del model.modelfit_results.residuals['CWRES']
        run_tool('ruvsearch', model, groups=4, p_value=0.05, skip=[])
