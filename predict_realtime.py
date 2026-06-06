
import argparse, json, time, joblib, os, smtplib, ssl
from email.mime.text import MIMEText
import numpy as np
import pandas as pd

LABEL_NAMES = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    10: "PortScan",
    11: "SSH-Patator",
}

def label_to_name(label):
    try:
        return LABEL_NAMES.get(int(label), f"Unknown ({label})")
    except Exception:
        return f"Unknown ({label})"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def send_email_alert(body):
    server = os.environ.get("SMTP_SERVER", "").strip()
    port = int(os.environ.get("SMTP_PORT", "0") or 0)
    user = os.environ.get("SMTP_USER", "").strip()
    pwd  = os.environ.get("SMTP_PASS", "").strip()
    to   = os.environ.get("ALERT_TO", "").strip()
    if not (server and port and user and pwd and to):
        return False, "SMTP not fully configured; skipping email."
    msg = MIMEText(body)
    msg["Subject"] = "Zeek Alert"
    msg["From"] = user
    msg["To"] = to
    try:
        if port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(server, port, context=ctx, timeout=20) as s:
                s.login(user, pwd); s.send_message(msg)
        else:
            with smtplib.SMTP(server, port, timeout=20) as s:
                s.ehlo()
                try:
                    s.starttls(context=ssl.create_default_context())
                except Exception:
                    pass
                s.login(user, pwd); s.send_message(msg)
        return True, "Email sent."
    except Exception as e:
        return False, str(e)

def zeek_row_to_features(row, feature_columns, maps, translator):
    t = {translator.get(k, k): v for k, v in row.items()}
    feat = {c: 0.0 for c in feature_columns}

    def as_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    def encode(val, mapping):
        sval = str(val) if val is not None else ""
        return mapping.get(sval, 0)

    
    if "protocol_type" in feat:
        feat["protocol_type"] = encode(t.get("proto"), maps.get("proto", {}))
    if "service" in feat:
        feat["service"] = encode(t.get("service"), maps.get("service", {}))
    if "duration" in feat:
        feat["duration"] = as_float(t.get("duration"))
    if "src_bytes" in feat:
        feat["src_bytes"] = as_float(t.get("orig_bytes"))
    if "dst_bytes" in feat:
        feat["dst_bytes"] = as_float(t.get("resp_bytes"))
    if "src_pkts" in feat:
        feat["src_pkts"] = as_float(t.get("orig_pkts"))
    if "dst_pkts" in feat:
        feat["dst_pkts"] = as_float(t.get("resp_pkts"))
    if "byte_ratio" in feat:
        o, r = feat.get("src_bytes", 0.0), feat.get("dst_bytes", 0.0)
        feat["byte_ratio"] = (o / r) if r not in (0, 0.0) else 0.0
    if "pkt_ratio" in feat:
        o, r = feat.get("src_pkts", 0.0), feat.get("dst_pkts", 0.0)
        feat["pkt_ratio"] = (o / r) if r not in (0, 0.0) else 0.0


    return [feat[c] for c in feature_columns]

def tail_json_lines(path):
    with open(path, "r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.25); continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", required=True)
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--feature-columns", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--translator", required=True)
    args = ap.parse_args()

    meta = load_json(args.feature_columns)
    feature_columns = meta.get("all_input_cols", meta) if isinstance(meta, dict) else meta
    maps = load_json(args.maps)
    translator = load_json(args.translator)
    model = joblib.load(args.model)

    log_path = os.environ.get("PREDICTION_LOG", "/home/aditi/zeek_predictions.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "a", buffering=1)

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            logf.write(line + "\n")
        except Exception:
            pass

    log(f"Predictor started. Tailing: {args.tail} | Using features: {feature_columns}")

    gen = tail_json_lines(args.conn) if args.tail else (json.loads(l) for l in open(args.conn, "r"))
    for row in gen:
        try:
            vals = zeek_row_to_features(row, feature_columns, maps, translator)
            X = pd.DataFrame([vals], columns=feature_columns)  
            pred = model.predict(X)[0]

            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = float(np.max(model.predict_proba(X)))
                except Exception:
                    proba = None

            uid = row.get("uid", "-")
            id_orig_h = row.get("id.orig_h", row.get("id_orig_h", "-"))
            id_resp_h = row.get("id.resp_h", row.get("id_resp_h", "-"))
            pred_name = label_to_name(pred)

            log(f"uid={uid} pred={pred} ({pred_name}) proba={proba if proba is not None else 'NA'} orig={id_orig_h} resp={id_resp_h}")

            if int(pred) != 0:
                body = (
                    "Zeek real-time alert\n\n"
                    f"uid: {uid}\n"
                    f"id.orig_h: {id_orig_h}\n"
                    f"id.resp_h: {id_resp_h}\n"
                    f"prediction: {pred_name} (label={pred})\n"
            
                )
                ok, info = send_email_alert(body)
                log("Alert email sent." if ok else f"Email: {info}")
        except Exception as e:
            log(f"Error: {e}")

    logf.close()

if __name__ == "__main__":
    main()
