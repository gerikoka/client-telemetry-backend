from flask import Flask, jsonify
from flask_cors import CORS
import boto3
import pandas as pd
from io import BytesIO
from datetime import datetime
import json

REGION = "us-east-1"
BUCKET = "sagemaker-studio-i0gutcxdy"

RESULTS_PREFIX = "frustration-model/results/"
FEATURES_PREFIX = "frustration-model/preprocessed_data/"

RAW_BUCKET = "sagemaker-us-east-1-197337164107"
RAW_PREFIX = "raw/"

RECENT_RUNS_TO_CHECK = 3

# Demo-only setting:
# This does NOT change S3 data. It only changes what the dashboard displays.
DEMO_SIMULATE_ONGOING = False

DEMO_ONGOING_SESSION_IDS = {
    "S-pw2-1776554466129-4563",
    "S1776554475224-1480",
    "S-pw5-1776554467638-8482",
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


def format_scenario_name(route: str) -> str:
    if not route:
        return "—"

    clean = str(route).strip().lower()

    if clean in ["blank", "unknown", "none", "null", ""]:
        return "Uncategorized"

    return str(route).replace("_", " ").replace("-", " ").title()


def load_csv_from_s3(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    return pd.read_csv(BytesIO(data))


def list_s3_objects(prefix: str):
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return response.get("Contents", [])


def list_raw_telemetry_files():
    files = []
    continuation_token = None

    while True:
        kwargs = {"Bucket": RAW_BUCKET, "Prefix": RAW_PREFIX}

        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**kwargs)
        files.extend(response.get("Contents", []))

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")

    return [
        obj["Key"]
        for obj in files
        if obj["Key"].endswith(".ndjson")
    ]


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

    return latest_ids.intersection(previous_ids)


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


def extract_pw_number(session_id: str):
    parts = str(session_id).split("-")
    for part in parts:
        if part.startswith("pw"):
            return part.replace("pw", "")
    return None


def event_matches_session(event, session_id):
    """
    Exact match only for metrics/timeline.
    This prevents one session from accidentally pulling events for another user.
    """
    event_session_id = (
        event.get("sessionId")
        or event.get("session_id")
        or event.get("sessionID")
        or event.get("metadata", {}).get("sessionId")
    )

    return str(event_session_id) == str(session_id)


def get_raw_events_for_session(session_id):
    events = []
    raw_files = list_raw_telemetry_files()

    for key in raw_files:
        try:
            file_obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)

            for line in file_obj["Body"].iter_lines():
                if not line:
                    continue

                try:
                    event = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                if event_matches_session(event, session_id):
                    events.append(event)

        except Exception as e:
            print("Error reading file:", key, e)
            continue

    events.sort(key=lambda e: str(e.get("timestamp", "")))
    return events


def calculate_raw_event_metrics(session_id):
    events = get_raw_events_for_session(session_id)

    event_types = [
        str(e.get("eventType") or e.get("event") or e.get("type") or "").lower()
        for e in events
    ]

    routes = [
        e.get("pageRoute")
        or e.get("route")
        or e.get("metadata", {}).get("pageRoute")
        for e in events
    ]

    routes = [str(r) for r in routes if r]

    idle_timeout_count = sum(
        1 for t in event_types
        if t in ["idle_time", "idle_timeout", "timeout"]
    )

    refocus_count = sum(
        1 for t in event_types
        if t in ["refocus", "window_focus", "focus", "visibility_change"]
    )

    nav_loop_count = 0
    for i in range(len(routes) - 2):
        if routes[i] == routes[i + 2] and routes[i] != routes[i + 1]:
            nav_loop_count += 1

    backtrack_count = 0
    for i in range(1, len(routes)):
        if routes[i] in routes[:i - 1]:
            backtrack_count += 1

    backtrack_rate = None
    if len(routes) > 1:
        backtrack_rate = round(backtrack_count / (len(routes) - 1), 2)

    route_journey = []
    for route in routes:
        if not route_journey or route_journey[-1] != route:
            route_journey.append(route)

    route_journey_text = (
        " → ".join(format_scenario_name(route) for route in route_journey[:6])
        if route_journey
        else "—"
    )

    return {
        "rawEventCount": len(events),
        "idleTimeoutCount": idle_timeout_count,
        "refocusCount": refocus_count,
        "navLoopCount": nav_loop_count,
        "backtrackRate": backtrack_rate,
        "routeJourney": route_journey_text,
    }
def load_scenario_map_from_raw(session_ids):
    """
    Builds sessionId -> scenario using real raw telemetry.

    Better rule:
      1. Collect all real flowName/pageRoute values for each session
      2. Ignore generic values like home/help/blank
      3. Pick the most frequent meaningful value
      4. Fall back to General Navigation if nothing meaningful exists
    """
    session_ids = set(str(sid) for sid in session_ids)

    generic_values = {"blank", "home", "login", "help", "unknown", "none", "null", ""}

    scenario_counts = {sid: {} for sid in session_ids}
    fallback_map = {}

    try:
        raw_files = list_raw_telemetry_files()
        print(f"RAW TELEMETRY FILES FOUND: {len(raw_files)}")

        for key in raw_files:
            try:
                obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)

                for line in obj["Body"].iter_lines():
                    if not line:
                        continue

                    try:
                        event = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue

                    event_session_id = str(
                        event.get("sessionId")
                        or event.get("session_id")
                        or event.get("sessionID")
                        or event.get("metadata", {}).get("sessionId")
                        or ""
                    )

                    if event_session_id not in session_ids:
                        continue

                    metadata = event.get("metadata", {}) or {}

                    flow_name = metadata.get("flowName") or event.get("flowName")
                    page_route = (
                        event.get("pageRoute")
                        or metadata.get("pageRoute")
                        or event.get("route")
                    )

                    # Save fallback route in case no meaningful flow/page exists
                    if page_route and event_session_id not in fallback_map:
                        fallback_map[event_session_id] = page_route

                    candidates = [flow_name, page_route]

                    for candidate in candidates:
                        if not candidate:
                            continue

                        clean = str(candidate).strip().lower()

                        if clean in generic_values:
                            continue

                        scenario_counts[event_session_id][clean] = (
                            scenario_counts[event_session_id].get(clean, 0) + 1
                        )

            except Exception as e:
                print(f"Error reading raw telemetry file {key}: {e}")
                continue

    except Exception as e:
        print(f"Could not load scenario map from raw telemetry: {e}")

    scenario_map = {}

    for sid in session_ids:
        counts = scenario_counts.get(sid, {})

        if counts:
            best_value = max(counts, key=counts.get)
            scenario_map[sid] = best_value
        elif sid in fallback_map:
            fallback = str(fallback_map[sid]).strip().lower()
            if fallback in generic_values:
                scenario_map[sid] = "general_navigation"
            else:
                scenario_map[sid] = fallback
        else:
            scenario_map[sid] = "general_navigation"

    formatted = {
        sid: format_scenario_name(route)
        for sid, route in scenario_map.items()
    }

    print(f"SCENARIOS FOUND FROM RAW TELEMETRY: {len(formatted)}")
    return formatted

def load_sessions():
    run_suffix, predictions_key, _naming_format = get_latest_prediction_run()
    df_pred = load_csv_from_s3(predictions_key)
    print("PREDICTIONS COLUMNS:", df_pred.columns.tolist())

    _, features_key, df_feat = load_matching_feature_vectors()
    print("FEATURES COLUMNS:", df_feat.columns.tolist())

    ongoing_ids = get_ongoing_session_ids()
    print(f"ONGOING SESSION COUNT FROM REAL RUN MATCHING: {len(ongoing_ids)}")
    print(f"DEMO SIMULATE ONGOING: {DEMO_SIMULATE_ONGOING}")

    latest_session_ids = set(df_pred["sessionId"].dropna().astype(str))
    scenario_map = load_scenario_map_from_raw(latest_session_ids)

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
            "scenario": scenario_map.get(session_id, "—"),
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
            "rawScenarioSource": f"s3://{RAW_BUCKET}/{RAW_PREFIX}",
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


@app.get("/api/sessions/<session_id>/events")
def get_session_events(session_id):
    events = get_raw_events_for_session(session_id)

    formatted_events = []

    for event in events:
        formatted_events.append({
            "timestamp": event.get("timestamp") or event.get("time"),
            "eventType": event.get("eventType") or event.get("event") or event.get("type"),
            "pageRoute": event.get("pageRoute") or event.get("route") or event.get("url"),
            "userId": event.get("userId") or event.get("user_id"),
            "metadata": event.get("metadata", {})
        })

    return jsonify(formatted_events), 200


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

    raw_metrics = calculate_raw_event_metrics(session_id)

    error_count = int(r.get("error_event_count", 0) or 0)
    retry_count = int(r.get("retry_count", 0) or 0)
    rage_click_count = int(r.get("rage_click_count", 0) or 0)

    drivers = []

    if error_count > 0:
        drivers.append("System errors detected")

    if retry_count > 0:
        drivers.append("Repeated retries")

    if rage_click_count > 0:
        drivers.append("Rage click behavior")

    if raw_metrics["idleTimeoutCount"] > 0:
        drivers.append("Idle timeout activity")

    if raw_metrics["navLoopCount"] > 0:
        drivers.append("Navigation loop behavior")

    if failure > 0:
        drivers.append("Flow failure / abandonment")

    if not drivers:
        drivers.append("No major frustration driver detected")

    primary_signal = drivers[0]

    metrics = {
        "totalClicks": int(r.get("click_count", 0) or 0),
        "errorCount": error_count,
        "retryCount": retry_count,
        "rageClickCount": rage_click_count,

        "navLoopCount": raw_metrics["navLoopCount"],
        "formAbandonment": bool(failure > 0),
        "backtrackRate": raw_metrics["backtrackRate"],
        "idleTimeoutCount": raw_metrics["idleTimeoutCount"],
        "refocusCount": raw_metrics["refocusCount"],

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
        "rawEventCount": raw_metrics["rawEventCount"],

        "primarySignal": primary_signal,
        "severityDrivers": drivers,
        "routeJourney": raw_metrics["routeJourney"],

        "runSuffix": run_suffix,
        "featuresSource": f"s3://{BUCKET}/{features_key}"
    }

    return jsonify(metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
