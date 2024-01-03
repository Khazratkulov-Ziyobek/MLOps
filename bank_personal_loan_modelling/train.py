import hydra
from catboost import CatBoostClassifier
from omegaconf import DictConfig
from utils import get_data, split_data


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("===== Training ... =====")
    train_data = get_data(cfg["train"]["train_data_path"])
    X_train, y_train = split_data(train_data)

    cat_features = [
        "Family",
        "Education",
        "Securities Account",
        "CD Account",
        "Online",
        "CreditCard",
    ]

    model = CatBoostClassifier(cat_features=cat_features, **cfg["catboost"])
    model.fit(X_train, y_train)
    model.save_model(cfg["train"]["path_to_save"])
    print(f"===== Model saved to {cfg['train']['path_to_save']} =====")


if __name__ == "__main__":
    main()
