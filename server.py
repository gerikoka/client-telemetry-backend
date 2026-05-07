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

RECENT_RUNS_TO_CHECK = 3

# Demo-only setting:
# This does NOT change S3 data. It only changes what the dashboard displays.
DEMO_SIMULATE_ONGOING = True

DEMO_ONGOING_SESSION_IDS = {
    "S-pw1-1772502475925-7915",
    "S1772502490686-2410",
    "S-pw7-1772502511498-1982",
}

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


def parse_prediction_file(obj):
    key = obj["Key"]
    filename = key.split("/")[-1]

    if not filename.endswith(".csv"):
        return None

    if filename.startswith("predictions_Run_"):
        run_suffix = filename.replace("predictions_Run_", "").replace(".csv", "")
        naming_format = "old"
    elif filename.startswith("predictions_model_v0_"):
        run_suffix = filename.replace("predictions_model_v0_", "").replace(".csv", "")
        naming_format = "new"
    else:
        return None

    return {
        "key": key,
        "filename": filename,
        "runSuffix": run_suffix,
        "namingFormat": naming_format,
        "lastModified": obj["LastModified"],
    }


def get_prediction_runs():
    contents = list_s3_objects(RESULTS_PREFIX)

    runs = []
    for obj in contents:
        parsed = parse_prediction_file(obj)
        if parsed:
            runs.append(parsed)

    if not runs:
        raise FileNotFoundError(
            f"No supported prediction files found under s3://{BUCKET}/{RESULTS_PREFIX}"
        )

    runs.sort(key=lambda x: x["lastModified"], reverse=True)
    return runs


def get_latest_prediction_run():
    latest = get_prediction_runs()[0]
    return latest["runSuffix"], latest["key"], latest["namingFormat"]


def get_recent_prediction_runs(limit=RECENT_RUNS_TO_CHECK):
    return get_prediction_runs()[:limit]


def get_matching_features_key(run_suffix: str, naming_format: str):
    if naming_format == "new":
        return f"{FEATURES_PREFIX}features_model_v0_{run_suffix}.csv"

    return f"{FEATURES_PREFIX}features_Run_{run_suffix}.csv"


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def get_ongoing_session_ids():
    """
    Real prototype heuristic:
    If a sessionId appears in the latest run AND at least one recent previous run,
    mark it as ONGOING. Otherwise, mark it as ENDED.
    """
    recent_runs = get_recent_prediction_runs(RECENT_RUNS_TO_CHECK)

    if len(recent_runs) < 2:
        return set()

    latest_df = load_csv_from_s3(recent_runs[0]["key"])
    latest_ids = set(latest_df["sessionId"].dropna().astype(str))

    previous_ids = set()

    for run in recent_runs[1:]:
        try:
            df = load_csv_from_s3(run["key"])
            if "sessionId" in df.columns:
                previous_ids.update(df["sessionId"].dropna().astype(str))
        except Exception as e:
            print(f"Could not load previous run {run['key']}: {e}")

    ongoing_ids = latest_ids.intersection(previous_ids)
    return ongoing_ids


def load_matching_feature_vectors():
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
    run_suffix, predictions_key, _naming_format = get_latest_prediction_run()
    df_pred = load_csv_from_s3(predictions_key)
    print("PREDICTIONS COLUMNS:", df_pred.columns.tolist())

    _, features_key, df_feat = load_matching_feature_vectors()
    print("FEATURES COLUMNS:", df_feat.columns.tolist())

    ongoing_ids = get_ongoing_session_ids()
    print(f"ONGOING SESSION COUNT FROM REAL RUN MATCHING: {len(ongoing_ids)}")
    print(f"DEMO SIMULATE ONGOING: {DEMO_SIMULATE_ONGOING}")

    sessions = []

    for _, r in df_pred.iterrows():
        session_id = str(r.get("sessionId"))

        feature_row = df_feat[df_feat["sessionId"].astype(str) == session_id]

        event_count = None
        if not feature_row.empty:
            event_count = int(feature_row.iloc[0].get("event_count", 0) or 0)

        score = float(r.get("frustrationScore") or 0)

        real_status = "ONGOING" if session_id in ongoing_ids else "ENDED"

        demo_status = (
            "ONGOING"
            if DEMO_SIMULATE_ONGOING and session_id in DEMO_ONGOING_SESSION_IDS
            else real_status
        )

        sessions.append({
            "sessionId": session_id,
            "frustrationScore": score,
            "severity": to_upper_severity(r.get("severity")),
            "timestamp": r.get("timestamp"),
            "scenario": r.get("scenario", "—"),
            "status": demo_status,
            "events": event_count,
        })

    return run_suffix, predictions_key, sessions


@app.get("/health")
def health():
    try:
        run_suffix, predictions_key, naming_format = get_latest_prediction_run()
        features_key = get_matching_features_key(run_suffix, naming_format)
        features_exists = s3_key_exists(features_key)
        recent_runs = get_recent_prediction_runs()
        ongoing_ids = get_ongoing_session_ids()

        return jsonify({
            "ok": True,
            "runSuffix": run_suffix,
            "namingFormat": naming_format,
            "predictionsSource": f"s3://{BUCKET}/{predictions_key}",
            "matchingFeaturesSource": f"s3://{BUCKET}/{features_key}",
            "matchingFeaturesExists": features_exists,
            "recentRunsChecked": [r["filename"] for r in recent_runs],
            "ongoingSessionCountFromRealRunMatching": len(ongoing_ids),
            "demoSimulateOngoing": DEMO_SIMULATE_ONGOING,
            "demoOngoingSessionIds": list(DEMO_ONGOING_SESSION_IDS),
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


@app.get("/api/alerts")
def alerts():
    _run_suffix, _predictions_key, session_rows = load_sessions()
    alert_rows = [
        s for s in session_rows
        if s["severity"] in ["MEDIUM", "HIGH"]
    ]
    return jsonify(alert_rows)


@app.get("/api/queue")
def queue():
    _run_suffix, _predictions_key, session_rows = load_sessions()

    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    status_rank = {"ONGOING": 0, "ENDED": 1}

    sorted_rows = sorted(
        session_rows,
        key=lambda s: (
            status_rank.get(s.get("status"), 1),
            severity_rank.get(s.get("severity"), 2),
            pd.to_datetime(s.get("timestamp"), errors="coerce")
        )
    )

    for idx, row in enumerate(sorted_rows, start=1):
        row["queuePosition"] = idx

    return jsonify(sorted_rows)


@app.get("/api/sessions/<session_id>/metrics")
def session_metrics(session_id):
    run_suffix, features_key, df = load_matching_feature_vectors()

    row = df[df["sessionId"].astype(str) == str(session_id)]

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
