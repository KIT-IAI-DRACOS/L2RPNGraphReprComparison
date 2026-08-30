from typing import Dict

from ray.tune.search.optuna import OptunaSearch
import optuna


class MyOptunaSearch(OptunaSearch):
    def on_trial_result(self, trial_id: str, result: Dict):
        # Check if metric exists in result
        if self._metric not in result:
            return

        # Defensive check: ensure trial_id exists in _ot_trials
        if trial_id not in self._ot_trials:
            # This shouldn't happen now that we fixed trial_str_creator,
            # but keep this check to avoid crashes if it does
            return

        super().on_trial_result(trial_id, result)

    def on_trial_complete(self, trial_id: str, result: Dict = None, error: bool = False):
        # Defensive check for trial completion as well
        if trial_id not in self._ot_trials:
            return

        super().on_trial_complete(trial_id, result, error)

    def get_study(self) -> optuna.Study:
        """
        Get the underlying Optuna study object for saving or analysis.

        Returns:
            optuna.Study: The Optuna study object
        """
        return self._ot_study

    def save_study(self, storage_path: str, study_name: str = None) -> str:
        """
        Save the Optuna study to a SQLite database.

        Args:
            storage_path: Directory path where the database will be saved
            study_name: Name for the study (defaults to the study's name)

        Returns:
            str: Full path to the saved database file
        """
        import os

        os.makedirs(storage_path, exist_ok=True)

        if study_name is None:
            study_name = self._ot_study.study_name

        db_filename = f"{study_name}.db"
        db_path = os.path.join(storage_path, db_filename)
        storage_url = f"sqlite:///{db_path}"

        # Copy the study to persistent storage
        optuna.copy_study(
            from_study_name=self._ot_study.study_name,
            from_storage=self._ot_study._storage,
            to_storage=storage_url,
            to_study_name=study_name,
        )

        return db_path

