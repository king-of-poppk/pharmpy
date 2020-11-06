import os
import re

import pytest

from pharmpy import cli, source
from pharmpy.plugins.nonmem.records import etas_record


# Skip pkgutil, reload source
@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('operation', ['*', '+'])
def test_add_covariate_effect(datadir, fs, operation):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'add_cov_effect', 'run1.mod', 'CL', 'WGT', 'exp', '--operation', operation]
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search('CLWGT', mod_ori)
    assert re.search('CLWGT', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize(
    'transformation, eta', [('boxcox', 'ETAB1'), ('tdist', 'ETAT1'), ('john_draper', 'ETAD1')]
)
def test_eta_transformation(datadir, fs, transformation, eta):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', transformation, 'run1.mod', '--etas', 'ETA(1)']
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_box:
        mod_ori = f_ori.read()
        mod_box = f_box.read()

    assert mod_ori != mod_box

    assert not re.search(eta, mod_ori)
    assert re.search(eta, mod_box)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('operation', ['*', '+'])
def test_add_etas(datadir, fs, operation):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'add_etas', 'run1.mod', 'S1', 'exp', '--operation', operation]
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'EXP\(ETA\(3\)\)', mod_ori)
    assert re.search(r'EXP\(ETA\(3\)\)', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, cli]]], indirect=True)
def test_results_linearize(datadir, fs):
    path = datadir / 'linearize' / 'linearize_dir1'
    fs.create_dir('linearize_dir1')
    fs.add_real_file(path / 'pheno_linbase.mod', target_path='linearize_dir1/pheno_linbase.mod')
    fs.add_real_file(path / 'pheno_linbase.ext', target_path='linearize_dir1/pheno_linbase.ext')
    fs.add_real_file(path / 'pheno_linbase.lst', target_path='linearize_dir1/pheno_linbase.lst')
    fs.add_real_file(path / 'pheno_linbase.phi', target_path='linearize_dir1/pheno_linbase.phi')
    fs.create_dir('linearize_dir1/scm_dir1')
    fs.add_real_file(
        path / 'scm_dir1' / 'derivatives.mod', target_path='linearize_dir1/scm_dir1/derivatives.mod'
    )
    fs.add_real_file(
        path / 'scm_dir1' / 'derivatives.ext', target_path='linearize_dir1/scm_dir1/derivatives.ext'
    )
    fs.add_real_file(
        path / 'scm_dir1' / 'derivatives.lst', target_path='linearize_dir1/scm_dir1/derivatives.lst'
    )
    fs.add_real_file(
        path / 'scm_dir1' / 'derivatives.phi', target_path='linearize_dir1/scm_dir1/derivatives.phi'
    )

    args = ['results', 'linearize', 'linearize_dir1']
    cli.main(args)

    assert os.path.exists('linearize_dir1/results.json')


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('eta_args', [['--etas', 'ETA(1) ETA(2)'], []])
def test_create_rv_block(datadir, fs, eta_args):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno_real.ext', target_path='run1.ext')
    fs.add_real_file(datadir / 'pheno_real.phi', target_path='run1.phi')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'create_rv_block', 'run1.mod'] + eta_args
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'BLOCK\(2\)', mod_ori)
    assert re.search(r'BLOCK\(2\)', mod_cov)
