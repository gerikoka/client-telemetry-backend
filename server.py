from flask import Flask, jsonify
from flask_cors import CORS
import boto3
import pandas as pd
from io import BytesIO
from datetime import datetime

REGION = "us-east-1"
BUCKET = "sagemaker-studio-i0gutcxdy"

RESULTS_PREFIX = "frustration-model/results/"
FEATURES_PREFIX = "frustration-model/preprocessed_data/"

app = Flask(__name__)
CORS(app)

s3 = boto3.client("s3", region_name=REGION)


def to_upper_severity(sev: str) -> str:
    s = str(sev or "").lower()
    if s == "high":
        return "HIGH"
    if s == "medium":
        return "MEDIUM"
    return "LOW"


def load_csv_from_s3(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    return pd.read_csv(BytesIO(data))


def list_s3_objects(prefix: str):
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return response.get("Contents", [])


def get_latest_prediction_run():
    """
    Finds the newest prediction CSV file.

    Supports old format:
      predictions_Run_009_20260309_0249.csv

    Supports new format:
      predictions_model_v0_run197_20260425_1801.csv
    """
    contents = list_s3_objects(RESULTS_PREFIX)

    run_files = [
        obj for obj in contents
        if obj["Key"].endswith(".csv")
        and (
            "predictions_Run_" in obj["Key"]
            or "predictions_model_v0_" in obj["Key"]
        )
    ]

    if not run_files:
        raise FileNotFoundError(
            f"No supported prediction files found under s3://{BUCKET}/{RESULTS_PREFIX}"
        )

    latest = max(run_files, key=lambda x: x["LastModified"])
    latest_key = latest["Key"]
    filename = latest_key.split("/")[-1]

    if filename.startswith("predictions_Run_"):
        run_suffix = filename.replace("predictions_Run_", "").replace(".csv", "")
        naming_format = "old"
    elif filename.startswith("predictions_model_v0_"):
        run_suffix = filename.replace("predictions_model_v0_", "").replace(".csv", "")
        naming_format = "new"
    else:
        raise ValueError(f"Unsupported prediction filename format: {filename}")

    return run_suffix, latest_key, naming_format


def get_matching_features_key(run_suffix: str, naming_format: str):
    """
    Matches prediction file to the correct feature file.
    """
    if naming_format == "new":
        return f"{FEATURES_PREFIX}features_model_v0_{run_suffix}.csv"

    return f"{FEATURES_PREFIX}features_Run_{run_suffix}.csv"


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def load_matching_feature_vectors():
    """
    Load the features file that matches the latest predictions run.
    """
    run_suffix, _predictions_key, naming_format = get_latest_prediction_run()
    features_key = get_matching_features_key(run_suffix, naming_format)

    if not s3_key_exists(features_key):
        raise FileNotFoundError(
            f"Matching features file not found: s3://{BUCKET}/{features_key}"
        )

    df = load_csv_from_s3(features_key)
    print("FEATURES COLUMNS:", df.columns.tolist())
    return run_suffix, features_key, df


def load_sessions():
    """
    Load latest predictions and join with matching feature data.
    """
    run_suffix, predictions_key, _naming_format = get_latest_prediction_run()
    df_pred = load_csv_from_s3(predictions_key)
    print("PREDICTIONS COLUMNS:", df_pred.columns.tolist())

    _, features_key, df_feat = load_matching_feature_vectors()
    print("FEATURES COLUMNS:", df_feat.columns.tolist())

    sessions = []

    for _, r in df_pred.iterrows():
        session_id = r.get("sessionId")

        feature_row = df_feat[df_feat["sessionId"] == session_id]

        event_count = None
        if not feature_row.empty:
            event_count = int(feature_row.iloc[0].get("event_count", 0) or 0)

        sessions.append({
            "sessionId": session_id,
            "frustrationScore": float(r.get("frustrationScore")),
            "severity": to_upper_severity(r.get("severity")),
            "timestamp": r.get("timestamp"),
            "scenario": r.get("scenario", "—"),
            "status": "ENDED",
            "events": event_count,
        })

    return run_suffix, predictions_key, sessions


@app.get("/health")
def health():
    try:
        run_suffix, predictions_key, naming_format = get_latest_prediction_run()
        features_key = get_matching_features_key(run_suffix, naming_format)
        features_exists = s3_key_exists(features_key)

        return jsonify({
            "ok": True,
            "runSuffix": run_suffix,
            "namingFormat": naming_format,
            "predictionsSource": f"s3://{BUCKET}/{predictions_key}",
            "matchingFeaturesSource": f"s3://{BUCKET}/{features_key}",
            "matchingFeaturesExists": features_exists,
            "checkedAt": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "checkedAt": datetime.utcnow().isoformat() + "Z"
        }), 500


@app.get("/api/sessions")
def sessions():
    _run_suffix, _predictions_key, session_rows = load_sessions()
    return jsonify(session_rows)


@app.get("/api/sessions/<session_id>")
def session_by_id(session_id):
    _run_suffix, _predictions_key, all_sessions = load_sessions()

    for s in all_sessions:
        if s["sessionId"] == session_id:
            return jsonify(s)

    return jsonify({"message": "Not found"}), 404


@app.get("/api/sessions/<session_id>/metrics")
def session_metrics(session_id):
    run_suffix, features_key, df = load_matching_feature_vectors()

    row = df[df["sessionId"] == session_id]

    print("\n=== RAW FEATURE ROW ===")
    print(row.to_dict(orient="records"))

    if row.empty:
        return jsonify({
            "message": "Metrics not found",
            "sessionId": session_id,
            "runSuffix": run_suffix,
            "featuresSource": f"s3://{BUCKET}/{features_key}"
        }), 404

    r = row.iloc[0]

    success = int(r.get("flow_success_count", 0) or 0)
    failure = int(r.get("flow_failure_count", 0) or 0)

    if success > 0:
        outcome = "SUCCESS"
    elif failure > 0:
        outcome = "FAILURE"
    else:
        outcome = "Outcome not available"

    metrics = {
        "totalClicks": int(r.get("click_count", 0) or 0),
        "errorCount": int(r.get("error_event_count", 0) or 0),
        "retryCount": int(r.get("retry_count", 0) or 0),
        "rageClickCount": int(r.get("rage_click_count", 0) or 0),

        "navLoopCount": None,
        "formAbandonment": bool((r.get("flow_failure_count", 0) or 0) > 0),
        "backtrackRate": None,
        "idleTimeoutCount": None,
        "refocusCount": None,

        "avgDwellTime": round(float(r.get("total_dwell_ms", 0) or 0) / 1000.0, 2),
        "sessionDurationSec": round(float(r.get("session_duration_ms", 0) or 0) / 1000.0, 2),
        "avgInterEventGapMs": round(float(r.get("avg_inter_event_gap_ms", 0) or 0), 2),

        "eventCount": int(r.get("event_count", 0) or 0),
        "pageViewCount": int(r.get("page_view_count", 0) or 0),
        "uniqueRouteCount": int(r.get("unique_route_count", 0) or 0),
        "fieldChangeCount": int(r.get("field_change_count", 0) or 0),
        "flowSuccessCount": success,
        "flowFailureCount": failure,

        "sessionOutcome": outcome,

        "runSuffix": run_suffix,
        "featuresSource": f"s3://{BUCKET}/{features_key}"
    }

    return jsonify(metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
