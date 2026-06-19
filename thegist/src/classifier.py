"""Classifier module for theGist.

This module handles training, saving, loading, and applying the
supervised relevance classifier. The classifier is trained on labeled
examples collected during filter-ideas sessions and predicts whether
an idea is relevant or irrelevant to a given subject.

The classifier uses logistic regression on top of sentence transformer
embeddings. Once trained it is saved to disk and loaded automatically
by filter-ideas when available.
"""

import pickle
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from thegist.config import DATA_DIR

MODEL_NAME = "all-MiniLM-L6-v2"
MODELS_DIR = DATA_DIR / "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _sanitize_subject(subject: str) -> str:
    """Converts a subject name into a safe filename string.

    Args:
        subject: The raw subject name to sanitize.

    Returns:
        A sanitized string safe for use as a filename.
    """
    sanitized = re.sub(r"[^\w\s-]", "", subject)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("_")
    return sanitized[:60]


def get_model_path(subject: str) -> Path:
    """Returns the file path for a trained classifier model.

    Args:
        subject: The subject the classifier was trained on.

    Returns:
        A Path object pointing to the model file location.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"filter_{_sanitize_subject(subject)}.pkl"


def model_exists(subject: str) -> bool:
    """Returns True if a trained classifier exists for the subject.

    Args:
        subject: The subject to check for a trained model.

    Returns:
        True if a model file exists, False otherwise.
    """
    return get_model_path(subject).exists()


def train(
    examples: list[dict],
    subject: str,
) -> dict:
    """Trains a relevance classifier on labeled examples.

    Encodes idea texts using sentence transformer embeddings then
    trains a logistic regression classifier with balanced class
    weights to handle label imbalance. Evaluates on a held out
    test set and saves the trained model to disk.

    Args:
        examples: A list of labeled example dictionaries each
            containing 'text' and 'label' keys.
        subject: The subject the classifier is being trained for.
            Used to derive the model filename.

    Returns:
        A metrics dictionary containing 'accuracy', 'precision',
        'recall', 'train_size', and 'test_size'.

    Raises:
        ValueError: If examples is empty or contains only one class.
    """
    if not examples:
        raise ValueError("No labeled examples provided for training.")

    labels = [e["label"] for e in examples]
    unique_classes = set(labels)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Training requires both relevant and irrelevant examples. "
            f"Only found label(s): {unique_classes}"
        )

    texts = [e["text"] for e in examples]

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} examples...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,  # preserve class ratio in both splits
    )

    print("Training classifier...")
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    # Evaluate on held out test set
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
        "precision": round(
            precision_score(y_test, y_pred, pos_label=0, zero_division=0) * 100, 1
        ),
        "recall": round(
            recall_score(y_test, y_pred, pos_label=0, zero_division=0) * 100, 1
        ),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    # Save both the classifier and the embedding model reference
    model_path = get_model_path(subject)
    payload = {
        "classifier": clf,
        "model_name": MODEL_NAME,
        "subject": subject,
        "trained_on": len(examples),
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    return metrics


def predict_irrelevance(
    texts: list[str],
    subject: str,
) -> list[float]:
    """Predicts the probability of each idea being irrelevant.

    Loads the trained classifier for the subject and returns an
    irrelevance probability for each input text. Higher values
    indicate the idea is more likely to be irrelevant.

    Args:
        texts: A list of idea text strings to classify.
        subject: The subject to load the trained classifier for.

    Returns:
        A list of float probabilities between 0.0 and 1.0 where
        higher values indicate higher likelihood of irrelevance.

    Raises:
        FileNotFoundError: If no trained model exists for the subject.
    """
    model_path = get_model_path(subject)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found for subject: {subject}. "
            f"Run: thegist train --subject \"{subject}\" --type filter"
        )

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    clf = payload["classifier"]
    embedding_model = SentenceTransformer(payload["model_name"])
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)

    # Return probability of class 0 (irrelevant)
    probabilities = clf.predict_proba(embeddings)
    irrelevance_probs = probabilities[:, list(clf.classes_).index(0)]

    return irrelevance_probs.tolist()