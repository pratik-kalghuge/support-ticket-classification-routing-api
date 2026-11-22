"""Reproducible dataset acquisition, cleaning, and stratified splitting."""

import hashlib
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_COMMIT = "72ea2203180d14a416b579c657033ecd21e59a11"
SOURCE_URL = (
    "https://raw.githubusercontent.com/bitext/"
    "customer-support-llm-chatbot-training-dataset/"
    f"{SOURCE_COMMIT}/data/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
SOURCE_SHA256 = "6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434"
CATEGORY_MAP = {
    "ACCOUNT": "Account Access",
    "PAYMENT": "Billing",
    "ORDER": "Order Management",
    "REFUND": "Refunds",
}


def combine_text(subject: str, description: str) -> str:
    """Shared normalization for training and serving; no label-derived subjects."""
    return re.sub(r"\s+", " ", unicodedata.normalize(
        "NFKC", f"{subject}\n{description}"
    )).strip()


def download_dataset(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded dataset checksum does not match pinned source")
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dataset checksum mismatch; remove the file and download again")


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"instruction", "category"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    selected = raw.loc[raw["category"].isin(CATEGORY_MAP)].copy()
    selected["category"] = selected["category"].map(CATEGORY_MAP)
    selected["subject"] = ""
    selected["description"] = selected["instruction"].fillna("")
    selected["text"] = selected["description"].map(lambda x: combine_text("", str(x)))
    nonblank = selected.loc[selected["text"].ne("")].copy()
    nonblank["key"] = nonblank["text"].str.casefold()
    # Contradictory labels cannot be resolved safely by keeping an arbitrary row.
    conflicts = nonblank.groupby("key")["category"].nunique()
    conflict_keys = conflicts[conflicts > 1].index
    consistent = nonblank.loc[~nonblank["key"].isin(conflict_keys)]
    clean = consistent.drop_duplicates("key").reset_index(drop=True)
    if set(clean["category"]) != set(CATEGORY_MAP.values()):
        raise ValueError("All four categories must be present after cleaning")
    if clean["category"].value_counts().min() < 10:
        raise ValueError("Need at least 10 unique tickets per category for stratified splits")
    stats = {
        "source_rows": len(raw), "selected_rows": len(selected),
        "blank_rows_removed": len(selected) - len(nonblank),
        "conflicting_rows_removed": len(nonblank) - len(consistent),
        "duplicate_rows_removed": len(consistent) - len(clean),
        "unique_rows": len(clean),
        "class_counts": clean["category"].value_counts().sort_index().to_dict(),
    }
    return clean, stats


def split_data(data: pd.DataFrame, seed: int = 42):
    train, remainder = train_test_split(
        data, test_size=0.30, stratify=data["category"], random_state=seed
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, stratify=remainder["category"], random_state=seed
    )
    return train, validation, test
