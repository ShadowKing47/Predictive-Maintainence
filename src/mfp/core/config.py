from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MFP_DATA_")

    raw_data_path: Path = Path("machine_data.csv")
    sensor_columns: list[str] = [
        "Temperature1",
        "Temperature2",
        "Pressure1",
        "Pressure2",
        "Temperature3",
        "Temperature4",
        "Pressure3",
        "Pressure4",
        "Temperature5",
        "Temperature6",
        "Pressure5",
        "Pressure6",
    ]
    temp_threshold: float = 85.0
    press_threshold: float = 110.0
    horizon: int = 30


class SplitConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MFP_SPLIT_")

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seq_len: int = 10
    purge_gap: int = 40


class ForecasterConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MFP_FORECASTER_")

    lstm_units: list[int] = [64, 64]
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 10
    gradient_clip: float = 1.0


class ClassifierConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MFP_CLASSIFIER_")

    hidden_units: list[int] = [64, 32]
    dropout: float = 0.3
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 10
    class_weight: dict[int, float] | None = None


class OptunaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MFP_OPTUNA_")

    enabled: bool = False
    n_trials: int = 50
    timeout: int | None = None
    study_name: str = "mfp_tuning"
    storage: str | None = None
    sampler: str = "tpe"
    pruner: str = "median"
    direction: str = "minimize"
    metric: str = "val_loss"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data: DataConfig = DataConfig()
    split: SplitConfig = SplitConfig()
    forecaster: ForecasterConfig = ForecasterConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    optuna: OptunaConfig = OptunaConfig()
    random_seed: int = 42
    mlflow_tracking_uri: str = "file:./mlruns"
    artifact_dir: Path = Path("./artifacts")


settings = Settings()
