import subprocess

import optuna

from . import trainer
from . import util


class Objective():

    DICT_DTYPE = {
        'int': int,
        'float': float,
    }

    def __init__(self, main_setting):
        self.main_setting = main_setting

    def __call__(self, trial):
        """Objective function to make optimization for Optuna.

        Args:
            trial: optuna.trial.Trial
        Returns:
            loss: float
                Loss value for the trial
        """

        # Update setting with suggesting hyperparameters
        dict_new_setting = self._create_dict_setting(trial)
        updated_setting = self.main_setting.update_with_dict(dict_new_setting)
        updated_setting.trainer.update_output_directory(
            id_=f"trial{trial.number}")

        # Train
        trainer_ = trainer.Trainer(updated_setting, optuna_trial=trial)
        loss = trainer_.train()
        return loss

    def _suggest_parameter(self, trial, dict_hyperparameter):
        parameter_type = dict_hyperparameter['type']
        if parameter_type == 'categorical':
            choices = dict_hyperparameter['choices']
            index = trial.suggest_int(
                dict_hyperparameter['name'],
                low=0, high=len(choices)-1)
            parameter = choices[index]
            # NOTE: Since list of list, list of dict, ... not supported, just
            # choose index instead of suggest_categorical

        elif parameter_type == 'discrete_uniform':
            if 'dtype' not in dict_hyperparameter:
                dict_hyperparameter['dtype'] = 'float'
            parameter = self.DICT_DTYPE[dict_hyperparameter['dtype']](
                trial.suggest_discrete_uniform(
                    dict_hyperparameter['name'],
                    low=dict_hyperparameter['low'],
                    high=dict_hyperparameter['high'],
                    q=dict_hyperparameter['step']))
        elif parameter_type == 'int':
            parameter = trial.suggest_int(
                dict_hyperparameter['name'],
                low=dict_hyperparameter['low'],
                high=dict_hyperparameter['high'])
        else:
            raise ValueError(f"Unsupported parameter type: {parameter_type}")

        print(f"\t{dict_hyperparameter['name']}: {parameter}")
        return parameter

    def _create_dict_setting(self, trial):
        # Suggest hyperparameters
        print('--')
        print(f"Trial: {trial.number}")
        print('Current hyperparameters:')
        hyperparameters = {
            dict_hyperparameter['name']:
            self._suggest_parameter(trial, dict_hyperparameter)
            for dict_hyperparameter
            in self.main_setting.optuna.hyperparameters}
        return self._replace_dict(
            self.main_setting.optuna.setting, hyperparameters)

    def _replace_dict(self, dict_setting, dict_replace):
        if isinstance(dict_setting, str):
            if dict_setting in dict_replace:
                return dict_replace[dict_setting]
            else:
                return dict_setting
        elif isinstance(dict_setting, int) or isinstance(dict_setting, float):
            return dict_setting
        elif isinstance(dict_setting, list):
            return [self._replace_dict(d, dict_replace) for d in dict_setting]
        elif isinstance(dict_setting, dict):
            for key, value in dict_setting.items():
                dict_setting.update({
                    key:
                    self._replace_dict(dict_setting[key], dict_replace)})
            return dict_setting
        else:
            raise ValueError(f"Unknown data type: {dict_setting.__class__}")


def perform_study(main_setting, db_setting):
    """Perform hyperparameter search study.

    Args:
        main_setting: siml.setting.MainSetting
            Main setting object.
        db_setting: siml.setting.DBSetting
            Database setting object.
    Returns:
        None
    """
    # Prepare study
    if db_setting is None:
        print(f"No DB setting found. No optuna data will be saved.")
        storage = None
    else:
        if db_setting.use_sqlite:
            storage = f"sqlite:///{main_setting.trainer.name}.db"
        else:
            storage = f"mysql://{db_setting.username}:" \
            + f"{db_setting.password}@{db_setting.servername}" \
            + f"/{db_setting.username}"

    top_name = util.get_top_directory().stem
    study_name = f"{top_name}_{main_setting.trainer.name}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True, pruner=optuna.pruners.MedianPruner())
    objective = Objective(main_setting)

    # Optimize
    study.optimize(objective, n_trials=main_setting.optuna.n_trial, catch=())

    # Visualize the best result
    print('=== Best Trial ===')
    print(study.best_trial)
