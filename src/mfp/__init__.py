from mfp.core import (
    configure_logging,
    get_logger,
    set_global_seeds,
    settings,
)

__version__ = "0.1.0"

__all__ = [
    "settings",
    "configure_logging",
    "get_logger",
    "set_global_seeds",
]

# Lazy imports for heavy modules
def __getattr__(name: str):
    if name in ("SensorForecaster", "RiskClassifier", "run_full_training_pipeline"):
        from mfp.models import RiskClassifier, SensorForecaster, run_full_training_pipeline
        return {"SensorForecaster": SensorForecaster, "RiskClassifier": RiskClassifier, "run_full_training_pipeline": run_full_training_pipeline}[name]
    raise AttributeError(f"module 'mfp' has no attribute '{name}'")

def __dir__():
    return __all__ + ["SensorForecaster", "RiskClassifier", "run_full_training_pipeline"]
