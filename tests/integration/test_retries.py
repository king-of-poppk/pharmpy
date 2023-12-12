import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import calculate_parameters_from_ucp, calculate_ucp_scale, update_inits
from pharmpy.tools import run_retries


@pytest.mark.parametrize(
    ('scale', 'use_initial'),
    (
        ('UCP', False),
        # ('normal', False),
        # ('UCP', True),
        # ('normal', True),
    ),
)
def test_retries(tmp_path, model_count, scale, use_initial, start_model):
    with chdir(tmp_path):
        fraction = 0.1
        number_of_candidates = 5
        res = run_retries(
            number_of_candidates=number_of_candidates,
            fraction=fraction,
            scale=scale,
            results=start_model.modelfit_results,
            model=start_model,
            use_initial_estimates=use_initial,
        )

        # All candidate models + start model
        assert len(res.summary_tool) == 6
        assert len(res.summary_models) == 6
        assert len(res.models) == 6
        for model in res.models:
            if model != start_model:
                is_within_fraction(start_model, model, scale, fraction, use_initial)
        rundir = tmp_path / 'retries_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == 5  # Not the start model ?
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


def is_within_fraction(start_model, candidate_model, scale, fraction, use_initial):
    if use_initial:
        parameter_value = [(p.name, p.init) for p in start_model.parameters]
    else:
        parameter_value = list(start_model.modelfit_results.parameter_estimates.items())

    allowed_dict = {}
    if scale == "normal":
        for parameter, value in parameter_value:
            allowed_dict[parameter] = (
                value - value * fraction,
                value + value * fraction,
            )
    elif scale == "UCP":
        if not use_initial:
            start_model = update_inits(
                start_model, start_model.modelfit_results.parameter_estimates
            )
        ucp_scale = calculate_ucp_scale(start_model)
        lower = {}
        upper = {}
        for parameter, _ in parameter_value:
            lower[parameter] = 0.1 - (0.1 * fraction)
            upper[parameter] = 0.1 + (0.1 * fraction)
        new_lower_parameters = calculate_parameters_from_ucp(start_model, ucp_scale, lower)
        new_upper_parameters = calculate_parameters_from_ucp(start_model, ucp_scale, upper)
        for parameter, _ in parameter_value:
            allowed_dict[parameter] = (
                new_lower_parameters[parameter],
                new_upper_parameters[parameter],
            )
    for parameter in candidate_model.parameters:
        assert allowed_dict[parameter.name][0] < parameter.init < allowed_dict[parameter.name][1]
