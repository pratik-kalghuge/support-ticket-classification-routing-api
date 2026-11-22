import numpy as np
import pandas as pd
import pytest

from ticket_router.data import CATEGORY_MAP, combine_text, prepare_data, split_data
from ticket_router.routing import TicketRouter
from ticket_router.train import routing_metrics, select_threshold


def test_deduplication_precedes_disjoint_stratified_splits():
    rows = [{"instruction": f"{category} unique ticket {i}", "category": category}
            for category in CATEGORY_MAP for i in range(40)]
    rows += [rows[0], {"instruction": " account   unique ticket 0 ", "category": "ACCOUNT"}]
    rows += [{"instruction": "ambiguous", "category": "ACCOUNT"},
             {"instruction": "ambiguous", "category": "PAYMENT"},
             {"instruction": None, "category": "ACCOUNT"}]
    clean, stats = prepare_data(pd.DataFrame(rows))
    assert stats["duplicate_rows_removed"] == 2
    assert stats["conflicting_rows_removed"] == 2
    assert stats["blank_rows_removed"] == 1
    train, validation, test = split_data(clean)
    assert [len(x) for x in (train, validation, test)] == [112, 24, 24]
    sets = [set(x["key"]) for x in (train, validation, test)]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert all(set(frame["category"]) == set(CATEGORY_MAP.values()) for frame in (train, validation, test))
    assert train["key"].tolist() == split_data(clean)[0]["key"].tolist()


def test_shared_text_normalization():
    assert combine_text(" Password \n", "  reset\tplease ") == "Password reset please"


def test_threshold_maximizes_coverage_subject_to_accuracy():
    rows = [
        {"threshold": 0.5, "auto_routed": 100, "coverage": 1.0, "accuracy": 0.90},
        {"threshold": 0.7, "auto_routed": 70, "coverage": 0.7, "accuracy": 0.99},
        {"threshold": 0.9, "auto_routed": 30, "coverage": 0.3, "accuracy": 1.0},
    ]
    assert select_threshold(rows, 0.98)["threshold"] == 0.7
    assert select_threshold(rows[:1], 0.98)["threshold"] == 1.0


def test_routing_metrics_and_empty_coverage():
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6]])
    labels = np.array(["A", "B"])
    assert routing_metrics(["A", "A"], probabilities, labels, 0.8) == {
        "threshold": 0.8, "auto_routed": 1, "coverage": 0.5, "accuracy": 1.0,
    }
    empty = routing_metrics(["A", "B"], probabilities, labels, 1.0)
    assert empty["coverage"] == 0 and empty["accuracy"] is None


def test_routing_threshold_boundary():
    class FixedPipeline:
        classes_ = np.array(["Billing", "Refunds"])

        def predict_proba(self, texts):
            return np.array([[0.8, 0.2]])

    assert TicketRouter(FixedPipeline(), 0.8).predict("payment", "")["routing_decision"] == "auto_route"
    assert TicketRouter(FixedPipeline(), 0.81).predict("payment", "")["routing_decision"] == "manual_review"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_reject_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        TicketRouter(None, threshold)
