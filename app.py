"""
HouseLens - House Price Prediction
College Project | Run: python app.py
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, json, numpy as np, os, sqlite3, datetime

app = Flask(__name__, static_folder=".")
CORS(app)
# encoders not used → remove or comment
# encoders = joblib.load("model/encoders.pkl")
# ── Load ML Model ─────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
model = joblib.load("model.pkl")

with open(os.path.join(BASE, "model", "meta.json")) as f:
    meta = json.load(f)
print(f"✅ Model loaded — Accuracy: {meta['accuracy']}%")

USD_TO_INR = 95.05  # 1 USD = ₹95.05 (June 2026)

def to_inr(usd):
    return round(usd * USD_TO_INR, -2)  # round to nearest ₹100

# ── SQLite Database ───────────────────────────────────────
DB = os.path.join(BASE, "history.db")

def init_db():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created      TEXT,
            summary      TEXT,
            neighborhood TEXT,
            price        REAL,
            confidence   REAL,
            tier         TEXT,
            low          REAL,
            high         REAL,
            insight      TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

def get_db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.json

    # Build feature vector
    row = {}
    for col in meta["numeric_features"]:
        row[col] = float(d.get(col, 0) or 0)
    for col in meta["categorical_features"]:
        le  = encoders[col]
        val = str(d.get(col, le.classes_[0]))
        if val not in set(le.classes_):
            val = le.classes_[0]
        row[col + "_enc"] = float(le.transform([val])[0])

    X = np.array([[row[c] for c in meta["feature_cols"]]])
    price_usd = float(np.clip(model.predict(X)[0], 50_000, 2_000_000))
    price_inr = to_inr(price_usd)

    # Confidence & range
    conf = round(min(meta["r2"] * 100 + min(d.get("OverallQual", 5) / 10, 1) * 2, 97.5), 1)
    low  = to_inr(price_usd * 0.90)
    high = to_inr(price_usd * 1.10)

    # Market tier
    med = to_inr(meta["price_stats"]["median"])
    if   price_inr > med * 1.5: tier, insight = "luxury",  "Luxury tier — significantly above market median"
    elif price_inr > med * 1.1: tier, insight = "premium", "Above average — premium features command higher value"
    elif price_inr > med * 0.9: tier, insight = "mid",     "Mid-market — well-aligned with neighborhood median"
    elif price_inr > med * 0.7: tier, insight = "entry",   "Below average — good entry-level opportunity"
    else:                        tier, insight = "value",   "Value segment — priced well below market median"

    # Save to DB (store INR)
    summary = (f"{d.get('BedroomAbvGr',0)}bd/{d.get('FullBath',0)}ba, "
               f"{int(d.get('GrLivArea',0)):,} sqft, Qual {d.get('OverallQual',0)}/10")
    con = get_db()
    con.execute(
        "INSERT INTO history (created,summary,neighborhood,price,confidence,tier,low,high,insight) VALUES (?,?,?,?,?,?,?,?,?)",
        (datetime.datetime.now().isoformat(), summary, d.get("Neighborhood","N/A"),
         price_inr, conf, tier, low, high, insight)
    )
    con.commit(); con.close()

    return jsonify(
        predicted_price  = price_inr,
        price_range_low  = low,
        price_range_high = high,
        confidence       = conf,
        market_tier      = tier,
        market_insight   = insight,
        model_accuracy   = meta["accuracy"]
    )

@app.route("/api/history")
def history():
    con  = get_db()
    rows = con.execute("SELECT * FROM history ORDER BY id DESC LIMIT 100").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/history/<int:rid>", methods=["DELETE"])
def delete_history(rid):
    con = get_db()
    con.execute("DELETE FROM history WHERE id=?", (rid,))
    con.commit(); con.close()
    return jsonify(ok=True)

@app.route("/api/stats")
def stats():
    con  = get_db()
    rows = con.execute("SELECT price, confidence, tier FROM history").fetchall()
    con.close()
    if not rows:
        return jsonify(total=0, avg=0, high=0, avg_conf=0, tiers={})
    prices = [r["price"] for r in rows]
    tiers  = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    return jsonify(
        total    = len(rows),
        avg      = round(sum(prices) / len(prices), 2),
        high     = max(prices),
        avg_conf = round(sum(r["confidence"] for r in rows) / len(rows), 1),
        tiers    = tiers
    )

if __name__ == "__main__":
    print("🏠 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
