"""Contract, HTTP and artifact regression tests."""

import io
from pathlib import Path

import pytest

from credit_risk.schema import FEATURES, validate_records
from credit_risk.service import PredictionService
from credit_risk.web import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="service", scope="module")
def fixture_service():
    """Load the reviewed artifact once."""
    return PredictionService(ROOT / "models")


@pytest.fixture(name="client")
def fixture_client():
    """Expose Flask's in-process client."""
    return create_app({"TESTING": True}).test_client()


@pytest.mark.parametrize("value", [[], {}, None, [None], [{}], [{}] * 501])
def test_bad_record_structure(value):
    """Reject invalid batches and missing columns."""
    with pytest.raises(ValueError):
        validate_records(value)


@pytest.mark.parametrize("value", ["bad", float("inf"), float("nan"), True, -5, 1e30])
def test_bad_numeric_values(service, value):
    """Reject non-finite, nonnumeric and impossible values."""
    with pytest.raises(ValueError):
        validate_records([dict(service.sample, NETMONTHLYINCOME=value)])


def test_missing_values(service):
    """Explicit nulls and sentinels use training imputation."""
    assert service.predict(
        [dict(service.sample, NETMONTHLYINCOME=None, num_dbt=-99999)]
    )
    with pytest.raises(ValueError):
        validate_records([{field: None for field in FEATURES}])


def test_account_ages_and_counts(service):
    """Enforce consistent account ages and integer counts."""
    for extra in [{"Age_Newest_TL": 100, "Age_Oldest_TL": 1}, {"Tot_Missed_Pmnt": 1.5}]:
        with pytest.raises(ValueError):
            validate_records([dict(service.sample, **extra)])


def test_prediction_parity(service):
    """Column order, unrelated IDs and batch size do not change results."""
    record = dict(reversed(list(service.sample.items())))
    record["PROSPECTID"] = "ignored"
    predicted = service.predict([service.sample])[0]
    assert service.predict([record] * 3) == [predicted] * 3
    assert abs(sum(predicted["scores"].values()) - 1) < 0.00001


def test_artifact_tampering(tmp_path):
    """Check integrity before deserializing a model."""
    for name in ["metrics.json", "sample.json"]:
        (tmp_path / name).write_bytes((ROOT / "models" / name).read_bytes())
    (tmp_path / "pipeline.joblib").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="integrity"):
        PredictionService(tmp_path)


def test_home_and_form(client, service):
    """Render the app and a scored form using real artifacts."""
    home = client.get("/")
    assert home.status_code == 200
    assert b"Credit Risk Explorer" in home.data
    assert home.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in home.headers["Content-Security-Policy"]
    response = client.post("/predict", data=service.sample)
    assert response.status_code == 200
    assert b"MODEL RESULT" in response.data
    bad = client.post("/predict", data=dict(service.sample, NETMONTHLYINCOME="bad"))
    assert bad.status_code == 400
    assert b'role="alert"' in bad.data


def test_api_and_csv_parity(client, service):
    """Single-record, JSON and CSV predictions agree."""
    expected = service.predict([service.sample])[0]
    response = client.post("/api/predict", json={"records": [service.sample]})
    assert response.status_code == 200
    assert response.json["predictions"][0] == expected
    sample = client.get("/sample.csv")
    response = client.post(
        "/predict-csv", data={"file": (io.BytesIO(sample.data), "sample.csv")}
    )
    assert response.status_code == 200
    assert expected["category"].encode() in response.data
    assert "attachment" in response.headers["Content-Disposition"]


@pytest.mark.parametrize("payload", [None, [], {}, {"records": []}, {"records": [{}]}])
def test_invalid_api(client, payload):
    """Invalid requests are client errors."""
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.json


def test_upload_limits(client):
    """Reject missing, incorrect, excessive and oversized input."""
    assert client.post("/predict-csv").status_code == 400
    assert (
        client.post(
            "/predict-csv", data={"file": (io.BytesIO(b"x\n1\n"), "x.csv")}
        ).status_code
        == 400
    )
    sample = client.get("/sample.csv").data.decode().splitlines()
    content = "\n".join([sample[0]] + [sample[1]] * 501).encode()
    assert (
        client.post(
            "/predict-csv", data={"file": (io.BytesIO(content), "large.csv")}
        ).status_code
        == 400
    )
    response = client.post(
        "/api/predict", data=b"x" * (1024 * 1024 + 1), content_type="application/json"
    )
    assert response.status_code == 413


def test_health_metrics_and_errors(client):
    """Readiness, metric totals and HTTP errors support deployment checks."""
    assert client.get("/health/ready").json["status"] == "ready"
    report = client.get("/api/metrics").json
    assert sum(map(sum, report["test"]["confusion_matrix"])) == report["test"]["rows"]
    assert sum(v["rows"] for v in report["split"].values()) == 51336
    assert len(report["features"]) == len(FEATURES)
    assert 0 <= report["test"]["macro_f1"] <= 1
    assert client.get("/does-not-exist").status_code == 404
