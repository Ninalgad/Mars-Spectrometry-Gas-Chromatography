from loguru import logger
import pandas as pd
from pathlib import Path
import typer
import numpy as np
from src.model import get_model
from src.make_dataset import make_dataset


def main(
    model_dir: Path = typer.Option(
        "../models/", help="Path to the saved model weights"
    ),
    features_path: Path = typer.Option(
        "data/raw/x-submission.npy", help="Path to the test features"
    ),
    submission_save_path: Path = typer.Option(
        "../data/processed/submission.csv", help="Path to save the generated submission"
    ),
    submission_format_path: Path = typer.Option(
        "../data/raw/submission_format.csv", help="Path to the submission format csv"
    ),
    debug: bool = typer.Option(
        False, help="Run on a small subset of the data for debugging"
    )
):
    debug = True
    if debug:
        logger.info("Running in debug mode")
        X = np.load("../data/processed/x-submission.npy")
    else:
        logger.info(f"Loading feature data from {features_path}")
        features = pd.read_csv(features_path, index_col="sequence_id", nrows=nrows)

        logger.info(f"Processing feature data. Shape: {features.shape}")
        X = make_dataset(features)

    logger.info("Creating model")
    model = get_model()

    logger.info(
        f"Predicting labels based on submission format at {submission_format_path}"
    )
    probas = 0
    n_models = 10
    for i in range(n_models):
        h5_path = model_dir / f"s{i}.h5"
        model.load_weights(h5_path)
        logger.info(f"Loading trained model weights from {h5_path}")

        probas += model.predict(X)
    probas /= n_models

    # generate submission
    submission_format = pd.read_csv(submission_format_path)
    my_submission = submission_format.copy()

    for i, c in enumerate(my_submission.columns[1:]):
        my_submission[c] = probas[:, i]

    if not debug:
        my_submission.to_csv(submission_save_path, index=False)
        logger.success(f"Submission saved to {submission_save_path}")


if __name__ == "__main__":
    typer.run(main)