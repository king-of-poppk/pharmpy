import hashlib
import os.path
import uuid
from itertools import repeat
from pathlib import Path

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model.external.nonmem import convert_model
from pharmpy.modeling import create_rng, get_observations, write_csv, write_model
from pharmpy.workflows import ModelEntry
from pharmpy.workflows.log import Log
from pharmpy.workflows.results import ModelfitResults

PARENT_DIR = f'..{os.path.sep}'


def execute_model(model_entry, context):
    assert isinstance(model_entry, ModelEntry)
    model = model_entry.model

    database = context.model_database
    model = convert_model(model)
    path = Path.cwd() / f'NONMEM_run_{model.name}-{uuid.uuid1()}'

    # NOTE: This deduplicates the dataset before running NONMEM, so we know which
    # filename to give to this dataset.
    database.store_model(model)
    # NOTE: We cannot reuse model_with_correct_datapath as the model object
    # later because it might have lost some of the ETA names mapping due to the
    # current incomplete implementation of serialization of Pharmpy Model
    # objects through the NONMEM plugin. Hopefully we can get rid of this
    # hack later.
    model_with_correct_datapath = database.retrieve_model(model)
    stream = model_with_correct_datapath.internals.control_stream
    data_record = stream.get_records('DATA')[0]
    relative_dataset_path = data_record.filename

    # NOTE: We set up a directory tree that replicates the structure generated by
    # the database so that NONMEM writes down the correct relative paths in
    # generated files such as results.lst.
    # NOTE: It is important that we do this in a DB-agnostic way so that we do
    # not depent on its implementation.
    depth = relative_dataset_path.count(PARENT_DIR)
    # NOTE: This creates an FS tree branch x/x/x/x/...
    model_path = path.joinpath(*repeat('x', depth))
    meta = model_path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
    # NOTE: This removes the leading ../
    relative_dataset_path_suffix = relative_dataset_path[len(PARENT_DIR) * depth :]
    # NOTE: We do not support non-leading ../, e.g. a/b/../c
    assert PARENT_DIR not in relative_dataset_path_suffix
    dataset_path = path / Path(relative_dataset_path_suffix)
    datasets_path = dataset_path.parent
    datasets_path.mkdir(parents=True, exist_ok=True)

    # NOTE: Write dataset and model files so they can be used by NONMEM.
    model = write_csv(model, path=dataset_path, force=True)
    model = write_model(model, path=model_path / "model.ctl", force=True)

    if model_entry.parent:
        key = context.retrieve_key(model_entry.parent.name)
        res = context.model_database.retrieve_modelfit_results(key)
        ofv_parent = res.ofv
    else:
        ofv_parent = None

    # Create dummy ModelfitResults object
    modelfit_results = create_dummy_modelfit_results(model, ref=ofv_parent)

    log = modelfit_results.log if modelfit_results else None
    model_entry = model_entry.attach_results(modelfit_results=modelfit_results, log=log)

    modelfit_results.to_json(path=model_path / f'{model.name}_results.json')

    with database.transaction(model_entry) as txn:
        txn.store_local_file(path=model_path / f'{model.name}_results.json')

    context.store_model_entry(model_entry)

    return model_entry


def create_dummy_modelfit_results(model, ref=None):
    try:
        obs = get_observations(model)
    except IndexError:
        obs = None
    else:
        n_obs = len(get_observations(model))

    n = len(model.dataset)
    try:
        id_name = model.datainfo.id_column.name
    except IndexError:
        id_name = 'ID'
    n_id = len(model.dataset[id_name].unique())

    log = Log()

    h = hashlib.sha1(model.name.encode('utf-8')).hexdigest()
    seed = int(h, 16)
    rng = create_rng(seed)

    if ref:
        low = ref / 2 if ref > 0 else ref * 2
        ofv = rng.uniform(low, ref)
    else:
        ofv = rng.uniform(-20, 20)

    params = pd.Series(model.parameters.inits)
    params = params.apply(lambda x: x + rng.random() * 0.1)

    rse = pd.Series(model.parameters.inits)
    rse.iloc[:] = rng.uniform(-1, 1)
    rse.name = 'RSE'

    se = pd.Series(model.parameters.inits)
    se.iloc[:] = rng.uniform(-1, 1)
    se.name = 'SE'

    if obs is None:
        residuals = None
        predictions = None
    else:
        cwres = pd.Series(_rand_array(1, n_obs, rng), name='CWRES')
        residuals = pd.concat([cwres], axis=1)

        cipredi = pd.Series(_rand_array(1, n, rng), name='CIPREDI')
        ipred = pd.Series(_rand_array(1, n, rng), name='IPRED')
        pred = pd.Series(_rand_array(1, n, rng), name='PRED')
        predictions = pd.concat([cipredi, ipred, pred], axis=1)

    eta_names = model.random_variables.etas.names
    data = pd.DataFrame(
        _rand_array(n_id, len(eta_names), rng, 'standard_normal'), columns=eta_names
    )
    individual_ests = data.set_index(model.dataset[id_name].unique())

    iofv = pd.Series(_rand_array(1, n_id, rng), name='iofv')
    iofv.index = model.dataset[id_name].unique()

    cov_eta = pd.DataFrame(_rand_array(len(eta_names), len(eta_names), rng), columns=eta_names)
    cov_eta.index = eta_names
    iec = pd.Series([cov_eta] * n_id)
    iec.index = model.dataset[id_name].unique()

    param_names = model.parameters.names
    pes = pd.Series(_rand_array(1, len(param_names), rng), name='estimates')
    pes.index = param_names
    ses = pd.Series(_rand_array(1, len(param_names), rng), name='SE_sdcorr')
    ses.index = param_names

    results_dict = {
        'parameter_estimates': params,
        'log': log,
        'ofv': ofv,
        'minimization_successful': True,
        'warnings': [],
        'relative_standard_errors': rse,
        'standard_errors': se,
        'runtime_total': 10,
        'estimation_runtime': 0.5,
        'residuals': residuals,
        'predictions': predictions,
        'individual_estimates': individual_ests,
        'individual_ofv': iofv,
        'individual_estimates_covariance': iec,
        'significant_digits': 2,
        'log_likelihood': 5,
        'parameter_estimates_sdcorr': pes,
        'standard_errors_sdcorr': ses,
    }
    modelfit_results = ModelfitResults(**results_dict)
    return modelfit_results


def _rand_array(x, y, rng, generator='random'):
    rand_n = getattr(rng, generator)
    if x == 1:
        return np.array([rand_n() for yi in range(y)])
    else:
        a = np.empty((x, y))
        for xi in range(x):
            for yi in range(y):
                a[xi, yi] = rand_n()
        return a
