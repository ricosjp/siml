import optuna

from . import trainer
from . import util


class Objective():

    DICT_DTYPE = {
        'int': int,
        'float': float,
    }

    def __init__(self, main_setting, output_base):
        self.main_setting = main_setting
        self.output_base = output_base

    def __call__(self, trial):
        """Objective function to make optimization for Optuna.

        Parameters
        -----------
            trial: optuna.trial.Trial
        Returns
        --------
            loss: float
                Loss value for the trial
        """

        # Update setting with suggesting hyperparameters
        dict_new_setting = self._create_dict_setting(trial)
        updated_setting = self.main_setting.update_with_dict(dict_new_setting)
        updated_setting.trainer.update_output_directory(
            base=self.output_base, id_=f"trial{trial.number}")

        # Train
        trainer_ = trainer.Trainer(updated_setting, optuna_trial=trial)
        loss = trainer_.train()
        return loss

    def _suggest_parameter(self, trial, dict_hyperparameter):
        parameter_type = dict_hyperparameter['type']
        if parameter_type == 'categorical':
            choices = dict_hyperparameter['choices']
            ids = [c['id'] for c in choices]
            suggested_id = trial.suggest_categorical(
                dict_hyperparameter['name'], ids)
            for choice in choices:
                if choice['id'] == suggested_id:
                    parameter = choice['value']
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

        elif parameter_type == 'uniform':
            parameter = trial.suggest_uniform(
                dict_hyperparameter['name'],
                low=dict_hyperparameter['low'],
                high=dict_hyperparameter['high'])

        elif parameter_type == 'loguniform':
            if 'dtype' not in dict_hyperparameter:
                dict_hyperparameter['dtype'] = 'float'
            parameter = self.DICT_DTYPE[dict_hyperparameter['dtype']](
                trial.suggest_loguniform(
                    dict_hyperparameter['name'],
                    low=dict_hyperparameter['low'],
                    high=dict_hyperparameter['high']))

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
        return self._generate_dict(
            self.main_setting.optuna.setting, hyperparameters)

    def _generate_dict(self, default_settings, dict_replace):
        if isinstance(default_settings, list):
            return [
                self._generate_dict(d, dict_replace) for d in default_settings]
        elif isinstance(default_settings, dict):
            return {
                key: self._generate_dict(value, dict_replace)
                for key, value in default_settings.items()}
        elif isinstance(default_settings, str):
            if default_settings in dict_replace:
                return dict_replace[default_settings]
            else:
                return default_settings
        elif isinstance(default_settings, int) or isinstance(
                default_settings, float):
            return default_settings
        else:
            raise ValueError(
                f"Unknown data type: {default_settings.__class__}")


def study_callback(study, frozen_trial):
    if frozen_trial.state == optuna.structs.TrialState.PRUNED:
        study._log_completed_trial(frozen_trial.number, frozen_trial.value)
    print(f"Current best trial number: {study.best_trial.number}")
    return


def perform_study(main_setting, db_setting=None):
    """Perform hyperparameter search study.

    Parameters
    -----------
        main_setting: siml.setting.MainSetting
            Main setting object.
        db_setting: siml.setting.DBSetting
            Database setting object.

    Returns
    --------
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
    output_directory = main_setting.optuna.output_base_directory / study_name
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True, pruner=optuna.pruners.MedianPruner())
    objective = Objective(main_setting, output_directory)

    # Optimize
    study.optimize(
        objective, n_trials=main_setting.optuna.n_trial, catch=(),
        callbacks=(study_callback,))

    # Visualize the best result
    print('=== Best Trial ===')
    print(study.best_trial)

    return study
