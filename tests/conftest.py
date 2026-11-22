import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ticket_router.api import create_app


@pytest.fixture
def app_factory(tmp_path):
    # Real fitted pipeline; the fixture never downloads or loads public test data.
    texts = [
        "password login account locked", "reset account password login",
        "payment card charged twice", "credit card payment declined",
        "cancel order purchase", "track order purchase status",
        "refund money back", "refund return reimbursement",
    ]
    labels = ["Account Access"] * 2 + ["Billing"] * 2 + ["Order Management"] * 2 + ["Refunds"] * 2
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())])
    pipeline.fit(texts, labels)

    def factory(threshold=0.0):
        path = tmp_path / "model.joblib"
        joblib.dump({"schema_version": 1, "pipeline": pipeline, "threshold": threshold}, path)
        return create_app(path)

    return factory
