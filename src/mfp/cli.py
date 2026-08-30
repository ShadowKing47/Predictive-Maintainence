import typer
from pathlib import Path

from mfp.core.config import settings
from mfp.core.logging import configure_logging, get_logger
from mfp.models.train import run_full_training_pipeline

app = typer.Typer(help="Machine Failure Prediction CLI")
logger = get_logger(__name__)


@app.command()
def train(
    data_path: str = typer.Option(None, "--data", "-d", help="Path to raw CSV data"),
    artifact_dir: str = typer.Option(None, "--artifacts", "-a", help="Artifact output directory"),
    seed: int = typer.Option(settings.random_seed, "--seed", help="Random seed"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
) -> None:
    """Train forecaster and risk classifier."""
    configure_logging(log_level)
    settings.random_seed = seed

    logger.info("cli_train_start", data_path=data_path, artifact_dir=artifact_dir)
    results = run_full_training_pipeline(data_path, artifact_dir)
    logger.info("cli_train_complete", results=results)


@app.command()
def evaluate(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model artifact"),
    data_path: str = typer.Option(None, "--data", "-d", help="Path to test data"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
) -> None:
    """Evaluate a trained model."""
    configure_logging(log_level)
    logger.info("cli_evaluate", model_path=model_path, data_path=data_path)
    # TODO: implement evaluation


@app.command()
def backfill(
    data_path: str = typer.Option(..., "--data", "-d", help="Path to historical data"),
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model artifact"),
    output_path: str = typer.Option("backfill_predictions.csv", "--output", "-o", help="Output CSV"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
) -> None:
    """Run backfill predictions on historical data."""
    configure_logging(log_level)
    logger.info("cli_backfill", data_path=data_path, model_path=model_path, output=output_path)
    # TODO: implement backfill


if __name__ == "__main__":
    app()