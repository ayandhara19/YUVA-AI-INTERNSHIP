"""
Week 4 - Customer Segmentation Web App + Prediction API (Flask)

Run locally:   python app.py          -> http://127.0.0.1:5000
Production:    gunicorn app:app
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_segmentation.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"

INCOME, SCORE = "Annual Income (k$)", "Spending Score (1-100)"

SEGMENT_ADVICE = {
    "Premium Shoppers": "High income and high spending. Offer loyalty rewards, VIP events and premium product launches.",
    "Careful Spenders": "High income but low spending. Use personalised offers and quality-focused campaigns to raise engagement.",
    "Impulsive Spenders": "Lower income but high spending. Promote trending items, flash sales and instalment offers.",
    "Budget Customers": "Low income and low spending. Use discounts, coupons and value bundles.",
    "Average Shoppers": "Mid income and mid spending. Run seasonal promotions and cross-selling campaigns.",
}


def create_app():
    app = Flask(__name__)

    # ---- load the serialized model once, at start-up
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train_model.py` first.")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    names = {int(k): v for k, v in meta["cluster_names"].items()}

    def validate(payload):
        """Return (income, score, error_message)."""
        if not isinstance(payload, dict):
            return None, None, "Request body must be a JSON object."
        try:
            income = float(payload.get("annual_income"))
            score = float(payload.get("spending_score"))
        except (TypeError, ValueError):
            return None, None, ("'annual_income' and 'spending_score' are "
                                "required and must be numbers.")
        if not 0 <= income <= 1000:
            return None, None, "annual_income must be between 0 and 1000 (in k$)."
        if not 1 <= score <= 100:
            return None, None, "spending_score must be between 1 and 100."
        return income, score, None

    # ------------------------------------------------------------ routes
    @app.route("/")
    def home():
        return render_template("index.html", meta=meta)

    @app.route("/health")
    def health():
        return jsonify(status="ok", model="loaded", clusters=meta["n_clusters"])

    @app.route("/api/info")
    def info():
        return jsonify(
            model=meta["model"], features=meta["features"],
            n_clusters=meta["n_clusters"], n_samples=meta["n_samples"],
            silhouette_score=meta["silhouette_score"],
            segments=names, profiles=meta["cluster_profiles"])

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(silent=True)
        income, score, error = validate(payload)
        if error:
            return jsonify(error=error), 400

        X = pd.DataFrame([[income, score]], columns=[INCOME, SCORE])
        cluster = int(model.predict(X)[0])
        segment = names[cluster]
        return jsonify(
            cluster=cluster, segment=segment,
            recommendation=SEGMENT_ADVICE.get(segment, ""),
            input={"annual_income": income, "spending_score": score})

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="Route not found."), 404

    @app.errorhandler(405)
    def bad_method(_):
        return jsonify(error="Method not allowed."), 405

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
