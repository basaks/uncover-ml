import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from uncoverml.config import Config
from uncoverml.optimise.models import transformed_modelmaps as modelmaps
log = logging.getLogger(__name__)


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


regression_metrics = {
    'r2_score': lambda y, py, w:  r2_score(y, py, sample_weight=w),
    'expvar': lambda y, py, w: explained_variance_score(y, py, sample_weight=w),
    'mse': lambda y, py, w: mean_squared_error(y, py, sample_weight=w),
    'mae': lambda y, py, w: mean_absolute_error(y, py, sample_weight=w),
}


def bayesian_optimisation(X, y, w, groups, conf: Config):
    reg = modelmaps[conf.algorithm](** conf.algorithm_args)
    search_space = {k: eval(v) for k, v in conf.opt_params_space.items()}
    searchcv = BayesSearchCV(
        reg,
        search_spaces=search_space,
        ** conf.opt_searchcv_params,
        fit_params={'sample_weight': w},
        return_train_score=True,
        refit=False
    )

    log.info(f"Optimising params using BayesSearchCV .....")
    searchcv.fit(X, y, groups=groups)

    log.info(f"Finished param optimisation using BayesSearchCV .....")
    log.info(f"Best score found using param optimisation {searchcv.best_score_}")

    with open(conf.optimised_model_params, 'w') as f:
        json.dump(searchcv.best_params_, f, sort_keys=True, indent=4)
        log.info(f"Saved bayesian search optimised params in {conf.optimised_model_params}")

    log.info("Now training final model using the optimised model params")
    opt_model = modelmaps[conf.algorithm](** searchcv.best_params_)
    opt_model.fit(X, y, sample_weight=w)

    pd.DataFrame.from_dict(searchcv.cv_results_).sort_values(by='rank_test_score').to_csv(
        conf.optimisation_output
    )

    return opt_model


def score_model(trained_model, X, y, w=None):
    scores = {}
    y_pred = trained_model.predict(X)
    for k, m in regression_metrics.items():
        scores[k] = m(y, y_pred, w)
    return scores