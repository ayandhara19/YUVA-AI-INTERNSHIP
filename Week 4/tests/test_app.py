"""Automated tests for the model and the Flask API.  Run:  pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Mall Customer Segmentation" in r.data


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_info(client):
    data = client.get("/api/info").get_json()
    assert data["n_clusters"] == 5
    assert data["silhouette_score"] > 0.4


@pytest.mark.parametrize("income,score,expected", [
    (90, 90, "Premium Shoppers"),
    (90, 10, "Careful Spenders"),
    (20, 85, "Impulsive Spenders"),
    (20, 15, "Budget Customers"),
    (55, 50, "Average Shoppers"),
])
def test_predict_each_segment(client, income, score, expected):
    r = client.post("/predict", json={"annual_income": income, "spending_score": score})
    assert r.status_code == 200
    body = r.get_json()
    assert body["segment"] == expected
    assert 0 <= body["cluster"] < 5
    assert body["recommendation"]


def test_predict_accepts_numeric_strings(client):
    r = client.post("/predict", json={"annual_income": "60", "spending_score": "40"})
    assert r.status_code == 200


@pytest.mark.parametrize("payload", [
    {},
    {"annual_income": 50},
    {"annual_income": "abc", "spending_score": 50},
    {"annual_income": -5, "spending_score": 50},
    {"annual_income": 50, "spending_score": 0},
    {"annual_income": 50, "spending_score": 101},
])
def test_predict_rejects_bad_input(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_rejects_non_json(client):
    r = client.post("/predict", data="hello", content_type="text/plain")
    assert r.status_code == 400


def test_wrong_method_and_route(client):
    assert client.get("/predict").status_code == 405
    assert client.get("/nope").status_code == 404
