"""
:meta private:
"""

import warnings

from pharmpy.parameter import Parameter
from pharmpy.symbols import symbol as S


def power_on_ruv(model, list_of_eps=None):
    """
    Applies a power effect to provided epsilons.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    list_of_eps : list
        List of epsilons to apply power effect. If None, all epsilons will be used.
        None is default.
    """
    eps = _get_epsilons(model, list_of_eps)
    pset, sset = model.parameters, model.statements

    for i, e in enumerate(eps):
        theta_name = str(model.create_symbol(stem='power', force_numbering=True))
        theta = Parameter(theta_name, 0.01)
        pset.add(theta)

        sset.subs({e.name: model.individual_prediction_symbol ** S(theta.name) * e})

    model.parameters = pset
    model.statements = sset

    model.modelfit_results = None

    return model


def _get_epsilons(model, list_of_eps):
    rvs = model.random_variables

    if list_of_eps is None:
        return rvs.ruv_rvs
    else:
        eps = []
        for e in list_of_eps:
            try:
                eps.append(rvs[e.upper()])
            except KeyError:
                warnings.warn(f'Epsilon "{e}" does not exist')
        return eps
