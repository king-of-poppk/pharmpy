from typing import Optional, Union

import pharmpy.tools.iivsearch.algorithms as algorithms
from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Results
from pharmpy.modeling import add_pk_iiv, copy_model, create_joint_distribution
from pharmpy.modeling.results import RANK_TYPES
from pharmpy.tools.common import create_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.utils import runtime_type_check, same_arguments_as
from pharmpy.workflows import Task, Workflow, call_workflow

IIV_STRATEGIES = frozenset(('no_add', 'add_diagonal', 'fullblock'))
IIV_ALGORITHMS = frozenset(('brute_force',) + tuple(dir(algorithms)))


def create_workflow(
    algorithm: str,
    iiv_strategy: str = 'no_add',
    rank_type: str = 'bic',
    cutoff: Optional[Union[float, int]] = None,
    model: Optional[Model] = None,
):
    """Run IIVsearch tool. For more details, see :ref:`iivsearch`.

    Parameters
    ----------
    algorithm : str
        Which algorithm to run (brute_force, brute_force_no_of_etas, brute_force_block_structure)
    iiv_strategy : str
        If/how IIV should be added to start model. Possible strategies are 'no_add', 'add_diagonal',
        or 'fullblock'. Default is 'no_add'
    rank_type : str
        Which ranking type should be used (OFV, AIC, BIC). Default is BIC
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is None (all models will be ranked)
    model : Model
        Pharmpy model

    Returns
    -------
    IIVSearchResults
        IIVsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> from pharmpy.tools import run_iivsearch     # doctest: +SKIP
    >>> run_iivsearch('brute_force', model=model)   # doctest: +SKIP
    """

    wf = Workflow()
    wf.name = 'iivsearch'
    start_task = Task('start_iiv', start, model, algorithm, iiv_strategy, rank_type, cutoff)
    wf.add_task(start_task)
    task_results = Task('results', _results)
    wf.add_task(task_results, predecessors=[start_task])
    return wf


def create_algorithm_workflow(input_model, base_model, algorithm, iiv_strategy, rank_type, cutoff):
    wf: Workflow[IIVSearchResults] = Workflow()

    start_task = Task(f'start_{algorithm}', _start_algorithm, base_model)
    wf.add_task(start_task)

    if iiv_strategy != 'no_add':
        wf_fit = create_fit_workflow(n=1)
        wf.insert_workflow(wf_fit)
        base_model_task = wf_fit.output_tasks[0]
    else:
        base_model_task = start_task

    algorithm_func = getattr(algorithms, algorithm)
    wf_method = algorithm_func(base_model)
    wf.insert_workflow(wf_method)

    task_result = Task('results', post_process, rank_type, cutoff, input_model, base_model.name)

    post_process_tasks = [base_model_task] + wf.output_tasks
    wf.add_task(task_result, predecessors=post_process_tasks)

    return wf


def start(context, input_model, algorithm, iiv_strategy, rank_type, cutoff):
    if iiv_strategy != 'no_add':
        model_iiv = copy_model(input_model, 'base_model')
        _add_iiv(iiv_strategy, model_iiv)
        base_model = model_iiv
    else:
        base_model = input_model

    if algorithm == 'brute_force':
        list_of_algorithms = ['brute_force_no_of_etas', 'brute_force_block_structure']
    else:
        list_of_algorithms = [algorithm]

    sum_tools, sum_models, sum_inds, sum_inds_count, sum_errs = [], [], [], [], []

    # NOTE This creates an empty results object
    # TODO Somehow create results object only after all algorithms have been run
    res = post_process(rank_type, cutoff, input_model, base_model.name, [base_model])
    prev_models = {model.name for model in res.models}

    for algorithm_cur in list_of_algorithms:

        # NOTE Execute algorithm
        wf = create_algorithm_workflow(
            input_model, base_model, algorithm_cur, iiv_strategy, rank_type, cutoff
        )
        next_res = call_workflow(wf, f'results_{algorithm}', context)

        # NOTE Append results
        new_models = filter(lambda model: model.name not in prev_models, next_res.models)
        res.models.extend(new_models)
        prev_models.update(model.name for model in new_models)
        res.final_model_name = next_res.final_model_name
        sum_tools.append(next_res.summary_tool)
        sum_models.append(next_res.summary_models)
        sum_inds.append(next_res.summary_individuals)
        sum_inds_count.append(next_res.summary_individuals_count)
        sum_errs.append(next_res.summary_errors)
        final_model = [model for model in res.models if model.name == res.final_model_name]
        assert len(final_model) == 1
        base_model = final_model[0]

        # NOTE Force no_add on all algorithms except the first one
        iiv_strategy = 'no_add'

    if len(list_of_algorithms) > 1:
        keys = list(range(1, len(list_of_algorithms) + 1))
    else:
        keys = None

    res.summary_tool = _concat_summaries(sum_tools, keys)
    res.summary_models = _concat_summaries(sum_models, keys)
    res.summary_individuals = _concat_summaries(sum_inds, keys)
    res.summary_individuals_count = _concat_summaries(sum_inds_count, keys)
    res.summary_errors = _concat_summaries(sum_errs, keys)

    return res


def _concat_summaries(summaries, keys):
    if keys:
        return pd.concat(summaries, keys=keys, names=['step'])
    else:
        return pd.concat(summaries)


def _results(res):
    return res


def _start_algorithm(model):
    model.parent_model = model.name
    return model


def _add_iiv(iiv_strategy, model):
    assert iiv_strategy in ['add_diagonal', 'fullblock']
    add_pk_iiv(model)
    if iiv_strategy == 'fullblock':
        create_joint_distribution(model)
    return model


def post_process(rank_type, cutoff, input_model, base_model_name, *models):
    res_models = []
    base_model = None
    for model in models:
        if model.name == base_model_name:
            base_model = model
        else:
            res_models.append(model)

    if not base_model:
        raise ValueError('Error in workflow: No base model')

    res = create_results(
        IIVSearchResults, input_model, base_model, res_models, rank_type, cutoff, bic_type='iiv'
    )

    return res


@runtime_type_check
@same_arguments_as(create_workflow)
def validate_input(
    algorithm,
    iiv_strategy,
    rank_type,
):
    if algorithm not in IIV_ALGORITHMS:
        raise ValueError(
            f'Invalid `algorithm`: got `{algorithm}`, must be one of {sorted(IIV_ALGORITHMS)}.'
        )

    if rank_type not in RANK_TYPES:
        raise ValueError(
            f'Invalid `rank_type`: got `{rank_type}`, must be one of {sorted(RANK_TYPES)}.'
        )

    if iiv_strategy not in IIV_STRATEGIES:
        raise ValueError(
            f'Invalid `iiv_strategy`: got `{iiv_strategy}`,'
            f' must be one of {sorted(IIV_STRATEGIES)}.'
        )


class IIVSearchResults(Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        summary_errors=None,
        final_model_name=None,
        models=None,
        tool_database=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.summary_errors = summary_errors
        self.final_model_name = final_model_name
        self.models = models
        self.tool_database = tool_database
