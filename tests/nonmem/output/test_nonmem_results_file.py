from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from numpy import nan

import pharmpy.model.external.nonmem.table as table
import pharmpy.tools.external.nonmem.results_file as rf
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.workflows.log import Log

anan = pytest.approx(nan, nan_ok=True)

def _assert_estimation_status(_actual: rf.TermInfo, _expected: rf.TermInfo):
    expected = asdict(_expected)
    actual = asdict(_actual)

    assert actual.keys() == expected.keys()
    for key in expected.keys():
        assert type(actual[key]) is type(expected[key])
        if isinstance(expected[key], pd.DataFrame):
            assert str(actual[key]) == str(expected[key])
        elif expected[key] is nan:
            assert actual[key] is nan
        else:
            assert actual[key] == expected[key]

def test_supported_version():
    assert rf.NONMEMResultsFile.supported_version(None) is False
    assert rf.NONMEMResultsFile.supported_version('7.1.0') is False
    assert rf.NONMEMResultsFile.supported_version('7.2.0') is True
    assert rf.NONMEMResultsFile.supported_version('7.3.0') is True


def test_data_io(pheno_lst):
    rfile = rf.NONMEMResultsFile(pheno_lst)
    assert rfile.nonmem_version == "7.4.2"


@pytest.mark.parametrize(
    'file, table_number, expected, covariance_step_ok',
    [
        (
            'phenocorr.lst',
            1,
            rf.TermInfo(
                ebv_shrinkage = None,
                eps_shrinkage = None,
                minimization_successful = True,
                eta_shrinkage = None,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = 4.9,
                function_evaluations = 98,
                warning = False,
                ofv_with_constant = None,
            ),
            True,
        ),
        (
            'hessian_error.lst',
            1,
            rf.TermInfo(
                ebv_shrinkage = None,
                eps_shrinkage = None,
                minimization_successful = False,
                eta_shrinkage = None,
                estimate_near_boundary = None,
                rounding_errors = None,
                maxevals_exceeded = None,
                significant_digits = nan,
                function_evaluations = nan,
                warning = None,
                ofv_with_constant = None,
            ),
            False,
        ),
        (
            'large_s_matrix_cov_fail.lst',
            1,
            rf.TermInfo(
                minimization_successful = True,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = 3.1,
                function_evaluations = 62,
                warning = True,
                ofv_with_constant = None,
                ebv_shrinkage = pd.DataFrame(
                    data={
                        "EBVshrink(%):":  [3.6501E+01,3.6538E+01,5.3119E+01,3.3266E+01,4.0104E+01,3.7548E+01,2.0563E+01,2.2176E+01,1.6328E+01,1.5847E+01,3.3517E+01,3.3526E+01,4.9502E+01,4.9855E+01]
                    },
                ),
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [2.3471E+01]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [3.9737E+01,5.5124E+01,4.3564E+01,3.3676E+01,3.0325E+01,3.4879E+01,3.8156E+01,2.1724E+01,1.6271E+01,2.9704E+01,4.4234E+01,4.8502E+01,5.4318E+01,4.7864E+01]
                  },
                ),
            ),
            False,
        ),
        (
            'nm710_fail_negV.lst',
            1,
            rf.TermInfo(
                ebv_shrinkage = None,
                eps_shrinkage = None,
                minimization_successful = None,
                eta_shrinkage = None,
                estimate_near_boundary = None,
                rounding_errors = None,
                maxevals_exceeded = None,
                significant_digits = nan,
                function_evaluations = nan,
                warning = None,
                ofv_with_constant = None,
            ),
            None,
        ),
        (
            'sparse_matrix_with_msfi.lst',
            1,
            rf.TermInfo(
                minimization_successful = True,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = 3.1,
                function_evaluations = 112,
                warning = True,
                ofv_with_constant = None,
                ebv_shrinkage = pd.DataFrame(
                    data={
                        "EBVshrink(%):":  [4.5964E+01,3.4348E+01,5.2009E+01,3.3063E+01,3.3978E+01,3.7211E+01,2.1345E+01,2.3314E+01,1.5965E+01,1.6086E+01,3.2729E+01,3.3373E+01,5.1050E+01,4.9976E+01]
                    },
                ),
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [3.2079E+01]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [4.7663E+01,5.2882E+01,4.3524E+01,3.6822E+01,3.8634E+01,3.5256E+01,4.0755E+01,2.8192E+01,2.0729E+01,2.9972E+01,4.9818E+01,4.8307E+01,6.6686E+01,4.9792E+01]
                    },
                ),
            ),
            True,
        ),
        (
            'warfarin_ddmore.lst',
            1,
            rf.TermInfo(
                minimization_successful = True,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                warning = False,
                ofv_with_constant = None,
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [7.6245E+00]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [1.3927E+00, 1.3092E+01, 4.9181E+01, 5.3072E+01]
                    },
                ),
            ),
            False,
        ),
        (
            'mox_fail_nonp.lst',
            1,
            rf.TermInfo(
                minimization_successful = False,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                function_evaluations = 153,
                warning = False,
                ofv_with_constant = None,
                ebv_shrinkage = pd.DataFrame(
                    data={
                        "EBVshrink(%):":  [3.7232E+00,2.5777E+01,1.4788E+01,2.4381E+01,1.6695E+01]
                    },
                ),
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [2.5697E+01]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [3.1277E+01,5.2053E+01,7.5691E+00,7.8837E+01,8.2298E+01]
                    },
                ),
            ),
            False,
        ),
        (
            'mox_nocov_nonp.lst',
            1,
            rf.TermInfo(
                minimization_successful = False,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = nan,
                function_evaluations = 153,
                warning = False,
                ofv_with_constant = None,
                ebv_shrinkage = pd.DataFrame(
                    data={
                        "EBVshrink(%):":  [3.7232E+00, 2.5777E+01, 1.4788E+01, 2.4381E+01, 1.6695E+01]
                    },
                ),
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [2.5697E+01]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [3.1277E+01, 5.2053E+01, 7.5691E+00, 7.8837E+01, 8.2298E+01]
                    },
                ),
            ),
            False,
        ),
        (
            'pheno_nonp.lst',
            1,
            rf.TermInfo(
                minimization_successful = True,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = 3.6,
                function_evaluations = 107,
                warning = False,
                ofv_with_constant = None,
                ebv_shrinkage = pd.DataFrame(
                    data={
                        "EBVshrink(%):":  [3.8428E+01, 4.4592E+00]
                    },
                ),
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [2.7971E+01]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [3.8721E+01, 4.6492E+00]
                    },
                ),
            ),
            True,
        ),
        (
            'theo.lst',
            1,
            rf.TermInfo(
                ebv_shrinkage = None,
                eps_shrinkage = None,
                minimization_successful = True,
                eta_shrinkage = None,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = 4.2,
                function_evaluations = 208,
                warning = False,
                ofv_with_constant = None,
            ),
            True,
        ),
        (
            'theo_nonp.lst',
            1,
            rf.TermInfo(
                minimization_successful = False,
                estimate_near_boundary = True,
                rounding_errors = True,
                maxevals_exceeded = False,
                significant_digits = nan,
                function_evaluations = 735,
                warning = False,
                ofv_with_constant = None,
                ebv_shrinkage = pd.DataFrame(
                    data={
                        "EBVshrink(%):":  [9.6560E+01, 1.6545E+01, 1.6532E+01, 1.0000E+02]
                    },
                ),
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [3.2692E+00]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [9.6393E+01, 1.2506E+01, 1.2492E+01, 1.0000E+02]
                    },
                ),
            ),
            False,
        ),
        (
            'theo_withcov.lst',
            1,
            rf.TermInfo(
                ebv_shrinkage = None,
                eps_shrinkage = None,
                minimization_successful = True,
                eta_shrinkage = None,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = 4.2,
                function_evaluations = 208,
                warning = False,
                ofv_with_constant = None,
            ),
            True,
        ),
        (
            'UseCase7.lst',
            1,
            rf.TermInfo(
                minimization_successful = True,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = nan,
                function_evaluations = nan,
                warning = False,
                ofv_with_constant = None,
                ebv_shrinkage = pd.DataFrame(
                    data={
                        "EBVshrink(%):":   [9.0057E+00, 1.5538E+01, 4.7502E+01, 6.4241E+01]
                    },
                ),
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [1.3572E+01]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":   [9.0067E+00, 1.5526E+01, 4.7506E+01, 1.2182E+01]
                    },
                ),
            ),
            False,
        ),
        (
            'example6b_V7_30_beta.lst',
            1,
            rf.TermInfo(
                minimization_successful = True,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = False,
                significant_digits = nan,
                function_evaluations = nan,
                warning = False,
                ofv_with_constant = None,
                ebv_shrinkage = None,
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":  [1.5539E+01, 6.8462E+00],
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):": [6.3623E-01, 4.5490E+00, 9.5378E+00, 2.1243E+00, 1.5371E+00, 6.0355E+00, 4.0732E-01, 1.7659E+00],
                    },
                ),
            ),
            False,
        ),
        (
            'maxeval3.lst',
            1,
            rf.TermInfo(
                minimization_successful = False,
                estimate_near_boundary = False,
                rounding_errors = False,
                maxevals_exceeded = True,
                significant_digits = nan,
                function_evaluations = 5,
                warning = False,
                ofv_with_constant = 3376.151276351326,
                ebv_shrinkage = pd.DataFrame(
                    data={
                        "EBVSHRINKSD(%)":  [8.0993E+00, 1.8003E+00],
                        "EBVSHRINKVR(%)":  [1.5543E+01, 3.5683E+00],
                    },
                ),
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSSHRINKSD(%)":  [1.0000E-10],
                        "EPSSHRINKVR(%)":  [1.0000E-10],
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETASHRINKSD(%)": [1.0000E-10, 1.4424E+01],
                        "ETASHRINKVR(%)": [1.0000E-10, 2.6768E+01],
                    },
                ),
            ),
            False,
        ),
    ],
)
def test_estimation_status(testdata, file, table_number, expected, covariance_step_ok):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'oneEST' / 'noSIM')
    log = Log()
    rfile = rf.NONMEMResultsFile(p / file, log=log)
    actual = rfile.estimation_status(table_number)
    _assert_estimation_status(actual, expected)
    if covariance_step_ok is None:
        assert rfile.covariance_status(table_number).covariance_step_ok is None
    else:
        assert rfile.covariance_status(table_number).covariance_step_ok == covariance_step_ok


@pytest.mark.parametrize(
    'file, table_number, expected, covariance_step_ok',
    [
        (
            'anneal2_V7_30_beta.lst',
            2,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                warning=False,
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [-7.7375E+01]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [4.7323E+01, 3.9939E+01, 2.4474E+01, 0.0000E+00]
                    },
                ),
            ),
            True,
        ),
        (
            'superid2_6_V7_30_beta.lst',
            2,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                warning=False,
                eps_shrinkage = pd.DataFrame(
                    data={
                        "EPSshrink(%):":   [1.1964E+01]
                    },
                ),
                eta_shrinkage = pd.DataFrame(
                    data={
                        "ETAshrink(%):":  [1.1115E+01, 1.0764E+01, -1.9903E-01, -4.3835E-02, -9.1709E-02, -1.9494E-02]
                  },
                ),
            ),
            True,
        ),
    ],
)
def test_estimation_status_multest(testdata, file, table_number, expected, covariance_step_ok):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM')
    rfile = rf.NONMEMResultsFile(p / file)
    _assert_estimation_status(
        rfile.estimation_status(table_number),
        expected
    )
    assert rfile.covariance_status(table_number).covariance_step_ok == covariance_step_ok


def test_estimation_status_empty():
    rfile = rf.NONMEMResultsFile()
    assert rfile._supported_nonmem_version is False
    assert rfile.estimation_status(1) == rf.TermInfo(
        significant_digits=anan,
        function_evaluations=anan,
    )


def test_estimation_status_withsim(testdata):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'oneEST' / 'withSIM')
    rfile = rf.NONMEMResultsFile(p / 'control3boot.res', log=Log())

    assert rfile.estimation_status(45) == rf.TermInfo(
        minimization_successful=True,
        estimate_near_boundary=False,
        rounding_errors=False,
        maxevals_exceeded=False,
        significant_digits=3.3,
        function_evaluations=192,
        warning=False,
    )
    assert rfile.covariance_status(45).covariance_step_ok is False

    assert rfile.estimation_status(70) == rf.TermInfo(
        minimization_successful=True,
        estimate_near_boundary=True,
        rounding_errors=False,
        maxevals_exceeded=False,
        significant_digits=3.6,
        function_evaluations=202,
        warning=False,
    )
    assert rfile.covariance_status(70).covariance_step_ok is False

    assert rfile.estimation_status(100) == rf.TermInfo(
        minimization_successful=True,
        estimate_near_boundary=False,
        rounding_errors=False,
        maxevals_exceeded=False,
        significant_digits=5.6,
        function_evaluations=100,
        warning=False,
    )
    assert rfile.covariance_status(100).covariance_step_ok is True


def test_ofv_table_gap(testdata):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'multPROB' / 'multEST' / 'withSIM')
    rfile = rf.NONMEMResultsFile(p / 'multprobmix_nm730.lst', log=Log())

    _assert_estimation_status(
        rfile.estimation_status(2),
        rf.TermInfo(
            minimization_successful=False,
            estimate_near_boundary=False,
            rounding_errors=False,
            maxevals_exceeded=True,
            significant_digits=nan,
            function_evaluations=16,
            warning=False,
            eta_shrinkage=pd.DataFrame(
                data = {
                    'ETAshrink(%):': [1.7703,  12.038,  8.5112],
                },
            ),
            ebv_shrinkage=pd.DataFrame(
                data = {
                    'EBVshrink(%):': [1.1841, 9.9088, 9.0686],
                },
            ),
            eps_shrinkage=pd.DataFrame(
                data = {
                    'EPSshrink(%):': [10.166],
                },
            ),
        )
    )

    table_numbers = (1, 2, 3, 4, 6, 8, 10, 11, 12, 13)
    ext_table_file = table.NONMEMTableFile(p / 'multprobmix_nm730.ext')

    for n in table_numbers:
        ext_table = ext_table_file.table_no(n)
        assert isinstance(ext_table, table.ExtTable)
        assert rfile.ofv(n) == pytest.approx(ext_table.final_ofv)


@pytest.mark.parametrize(
    'file_name, ref_start, no_of_rows, idx, no_of_errors',
    [
        (
            'control_stream_error.lst',
            'AN ERROR WAS FOUND IN THE CONTROL STATEMENTS.',
            6,
            0,
            1,
        ),
        (
            'no_header_error.lst',
            'PRED EXIT CODE = 1',
            9,
            1,
            2,
        ),
        (
            'no_header_error.lst',
            'PROGRAM TERMINATED BY OBJ',
            2,
            2,
            2,
        ),
        (
            'rounding_error.lst',
            'MINIMIZATION TERMINATED\nDUE TO ROUNDING',
            2,
            0,
            2,
        ),
        (
            'zero_gradient_error.lst',
            'MINIMIZATION TERMINATED\nDUE TO ZERO',
            2,
            0,
            2,
        ),
        (
            'hessian.lst',
            'HESSIAN OF',
            1,
            0,
            1,
        ),
    ],
)
def test_errors(testdata, file_name, ref_start, no_of_rows, idx, no_of_errors):
    lst = rf.NONMEMResultsFile(testdata / 'nonmem' / 'errors' / file_name, log=Log())
    log_df = lst.log.to_dataframe()
    message = log_df['message'].iloc[idx]
    assert message.startswith(ref_start)
    assert len(message.split('\n')) == no_of_rows
    assert log_df['category'].value_counts()['ERROR'] == no_of_errors


@pytest.mark.parametrize(
    'file_name, ref, idx',
    [
        (
            'no_header_error.lst',
            'THE NUMBER OF PARAMETERS TO BE ESTIMATED\n'
            'EXCEEDS THE NUMBER OF INDIVIDUALS WITH DATA.',
            0,
        ),
        (
            'estimate_near_boundary_warning.lst',
            'PARAMETER ESTIMATE IS NEAR ITS BOUNDARY',
            0,
        ),
        (
            'est_step_warning.lst',
            'MINIMIZATION SUCCESSFUL\nHOWEVER, PROBLEMS OCCURRED WITH THE MINIMIZATION.',
            0,
        ),
    ],
)
def test_warnings(testdata, file_name, ref, idx):
    lst = rf.NONMEMResultsFile(testdata / 'nonmem' / 'errors' / file_name, log=Log())
    message = lst.log.to_dataframe()['message'].iloc[idx]
    assert message == ref


def test_covariance_status(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'modelfit_results' / 'covariance' / 'pheno_nocovariance.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert all(res.standard_errors.isna())
    assert res.covariance_matrix is None
    assert res.correlation_matrix is None
    assert res.precision_matrix is None
