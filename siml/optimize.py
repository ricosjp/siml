
import optuna

from . import setting


class Objective():

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

        # Suggest hyperparameters
        layer_number = trial.suggest_int('layer_number', 2, 5)
        activation_name = trial.suggest_categorical(
            'activation_name', ['relu', 'sigmoid'])
        unit_numbers = [
            trial.suggest_int(f"unit_number_layer{i}", 10, 100)
            for i in range(layer_number - 1)] + [1]
        dropout_ratio = trial.suggest_uniform('dropout_ratio', 0.0, 0.2)

        print('--')
        print(f"Trial: {trial.number}")
        print('Current hyperparameters:')
        print(f"    The number of layers: {layer_number}")
        print(f"    Activation function: {activation_name}")
        print(f"    The number of units for each layer: {unit_numbers}")
        print(f"    The ratio for dropout: {dropout_ratio}")
        print('--')

        # Train
        loss = self.trainer.run()
        return loss


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
    study = optuna.create_study(
        study_name=main_setting.trainer.name,
        storage=f"mysql://{db_setting.username}:"
        "{db_setting.password}@{db_setting.servername}/{db_setting.username}",
        load_if_exists=True, pruner=optuna.pruners.MedianPruner())

    # Optimize
    study.optimize(objective, n_trials=main_setting.n_trial)

    # Visualize the best result
    print('=== Best Trial ===')
    print(study.best_trial)
