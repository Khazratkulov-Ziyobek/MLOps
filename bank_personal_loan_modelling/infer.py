import git
import hydra
import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from omegaconf import DictConfig
from utils import get_data, get_metrics, split_data


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    is_mlflow_logging = cfg["mlflow"]["is_mlflow_logging"]

    test_data = get_data(cfg["infer"]["test_data_path"])
    X_test, y_test = split_data(test_data)

    model = CatBoostClassifier()
    model.load_model(cfg["infer"]["model_path"])

    y_predict = model.predict(X_test)
    metrics = get_metrics(y_test, y_predict)

    if is_mlflow_logging:
        mlflow.set_tracking_uri(uri=cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["exp_name"])

        with mlflow.start_run():
            mlflow.log_params(cfg["catboost"])
            mlflow.log_metrics(metrics)

            repo = git.Repo()
            sha = repo.head.object.hexsha
            mlflow.set_tag("commit_id", sha)

            signature = infer_signature(X_test, y_test)
            mlflow.catboost.log_model(
                cb_model=model,
                artifact_path=cfg["infer"]["model_path"],
                signature=signature,
                registered_model_name="catboost_classifier",
            )

    for key, value in metrics.items():
        print(f"{key}: {value}")

    pd.DataFrame(y_predict, columns=["Personal Loan"]).to_csv(
        cfg["infer"]["path_to_save_pred"]
    )

    path_to_save = cfg["infer"]["path_to_save_pred"]
    print(f"===== Predictions saved to {path_to_save} =====")


if __name__ == "__main__":
    main()
