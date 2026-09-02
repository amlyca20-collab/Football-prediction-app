import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Prediction Analyzer", page_icon="⚽", layout="wide")

def implied(odds):
    return 1 / odds if odds > 0 else 0

def normalize(odds):
    raw = [implied(o) for o in odds]
    total = sum(raw)
    return [x / total for x in raw]

def fair_odds(p):
    return 1 / p if p > 0 else float("inf")

def ev(p, odds):
    return p * odds - 1

def poisson(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def model(lh, la, max_goals=10):
    scores = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            scores[(h,a)] = poisson(h,lh) * poisson(a,la)
    total = sum(scores.values())
    scores = {k:v/total for k,v in scores.items()}

    hw = sum(p for (h,a),p in scores.items() if h>a)
    draw = sum(p for (h,a),p in scores.items() if h==a)
    aw = sum(p for (h,a),p in scores.items() if h<a)
    over15 = sum(p for (h,a),p in scores.items() if h+a>=2)
    over25 = sum(p for (h,a),p in scores.items() if h+a>=3)
    over35 = sum(p for (h,a),p in scores.items() if h+a>=4)
    btts = sum(p for (h,a),p in scores.items() if h>=1 and a>=1)
    top = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    return {
        "scores": scores, "home": hw, "draw": draw, "away": aw,
        "over15": over15, "under15": 1-over15,
        "over25": over25, "under25": 1-over25,
        "over35": over35, "under35": 1-over35,
        "btts": btts, "no_btts": 1-btts, "top": top
    }

def assessment(p, odds):
    value = ev(p, odds)
    if value >= .05: return "🟢 Strong value"
    if value >= .02: return "🟡 Possible value"
    if value >= 0: return "⚪ Tiny edge"
    return "🔴 No value"

st.title("⚽ Football Prediction Analyzer")
st.caption("Probability → Poisson → Fair Odds → Value")

with st.sidebar:
    st.header("Match")
    home_team = st.text_input("Home team", "LDU Quito")
    away_team = st.text_input("Away team", "Mushuc Runa")
    competition = st.text_input("Competition", "Ecuador Liga Pro")

    st.header("Expected Goals (λ)")
    st.caption("Enter your estimated expected goals for each team.")
    lh = st.number_input("Home λ", 0.05, 6.0, 1.55, 0.05)
    la = st.number_input("Away λ", 0.05, 6.0, 0.95, 0.05)

    st.header("1X2 Odds")
    h_odds = st.number_input("Home odds", 1.01, 100.0, 1.68, 0.01)
    d_odds = st.number_input("Draw odds", 1.01, 100.0, 3.71, 0.01)
    a_odds = st.number_input("Away odds", 1.01, 100.0, 4.88, 0.01)

    st.header("Goal Odds")
    o15 = st.number_input("Over 1.5", 1.01, 100.0, 1.26, 0.01)
    u15 = st.number_input("Under 1.5", 1.01, 100.0, 3.29, 0.01)
    o25 = st.number_input("Over 2.5", 1.01, 100.0, 1.87, 0.01)
    u25 = st.number_input("Under 2.5", 1.01, 100.0, 1.77, 0.01)
    o35 = st.number_input("Over 3.5", 1.01, 100.0, 3.18, 0.01)
    u35 = st.number_input("Under 3.5", 1.01, 100.0, 1.27, 0.01)

    st.header("BTTS")
    btts_yes = st.number_input("BTTS Yes", 1.01, 100.0, 1.85, 0.01)
    btts_no = st.number_input("BTTS No", 1.01, 100.0, 1.79, 0.01)

    max_goals = st.slider("Score matrix maximum goals", 5, 12, 8)

m = model(lh, la, max_goals)

st.subheader(f"{home_team} vs {away_team}")
st.write(f"**Competition:** {competition}")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Home λ", f"{lh:.2f}")
c2.metric("Away λ", f"{la:.2f}")
c3.metric("Expected total", f"{lh+la:.2f}")
c4.metric("BTTS Yes", f"{m['btts']*100:.1f}%")

st.header("1. 1X2 Analysis")
odds = [h_odds,d_odds,a_odds]
market = normalize(odds)
probs = [m["home"],m["draw"],m["away"]]

df = pd.DataFrame({
    "Outcome":[f"{home_team} Win","Draw",f"{away_team} Win"],
    "Odds":odds,
    "Raw implied %":[implied(x)*100 for x in odds],
    "Market %":[x*100 for x in market],
    "Model %":[x*100 for x in probs],
    "Fair odds":[fair_odds(x) for x in probs],
    "EV %":[ev(p,o)*100 for p,o in zip(probs,odds)],
    "Assessment":[assessment(p,o) for p,o in zip(probs,odds)]
})
st.dataframe(df, use_container_width=True, hide_index=True)
st.info(f"1X2 bookmaker overround: {sum(implied(x) for x in odds)*100:.2f}%")

st.header("2. Goal & BTTS Markets")
markets = [
    ("Over 1.5",m["over15"],o15),("Under 1.5",m["under15"],u15),
    ("Over 2.5",m["over25"],o25),("Under 2.5",m["under25"],u25),
    ("Over 3.5",m["over35"],o35),("Under 3.5",m["under35"],u35),
    ("BTTS Yes",m["btts"],btts_yes),("BTTS No",m["no_btts"],btts_no)
]
gdf = pd.DataFrame({
    "Market":[x[0] for x in markets],
    "Model %":[x[1]*100 for x in markets],
    "Fair odds":[fair_odds(x[1]) for x in markets],
    "Bookmaker odds":[x[2] for x in markets],
    "Implied %":[implied(x[2])*100 for x in markets],
    "Edge (percentage points)":[(x[1]-implied(x[2]))*100 for x in markets],
    "EV %":[ev(x[1],x[2])*100 for x in markets],
    "Assessment":[assessment(x[1],x[2]) for x in markets]
})
st.dataframe(gdf, use_container_width=True, hide_index=True)

st.header("3. Exact Score Probabilities")
score_df = pd.DataFrame(
    [[m["scores"][(h,a)]*100 for a in range(max_goals+1)]
     for h in range(max_goals+1)],
    index=[str(h) for h in range(max_goals+1)],
    columns=[str(a) for a in range(max_goals+1)]
)
score_df.index.name = f"{home_team} goals"
score_df.columns.name = f"{away_team} goals"
st.dataframe(score_df.style.format("{:.2f}%"), use_container_width=True)

top = m["top"][:10]
topdf = pd.DataFrame({
    "Rank":range(1,11),
    "Score":[f"{h}-{a}" for (h,a),p in top],
    "Probability %":[p*100 for (h,a),p in top],
    "Fair odds":[fair_odds(p) for (h,a),p in top]
})
st.subheader("Top 10 exact scores")
st.dataframe(topdf, use_container_width=True, hide_index=True)

st.header("4. Automatic Summary")
best_idx = max(range(3), key=lambda i: probs[i])
names = [f"{home_team} Win","Draw",f"{away_team} Win"]
st.success(f"Most likely 1X2 outcome: **{names[best_idx]} — {probs[best_idx]*100:.1f}%**")
st.write(f"**Most likely score:** {top[0][0][0]}-{top[0][0][1]} ({top[0][1]*100:.1f}%)")
best = max(markets, key=lambda x: ev(x[1],x[2]))
st.write(f"**Highest EV among entered markets:** {best[0]} at {best[2]:.2f}, model probability {best[1]*100:.1f}%, EV {ev(best[1],best[2])*100:.1f}%.")

st.header("5. Formula Reference")
st.markdown("""
**Implied probability:** `P = 1 / odds`

**Overround:** `OR = Σ(1 / odds)`

**Normalized market probability:** `Pᵢ = (1 / oddsᵢ) / OR`

**Fair odds:** `Fair odds = 1 / model probability`

**Expected value:** `EV = model probability × bookmaker odds − 1`

**Poisson:** `P(X=k) = e^(-λ) × λ^k / k!`

**Exact score:** `P(i-j) = P(Home=i) × P(Away=j)`

**BTTS Yes:** `(1 − e^(-λhome)) × (1 − e^(-λaway))`
""")

st.warning(
    "This version requires manual entry of statistics and expected goals. "
    "The next upgrade should automatically fetch team form, home/away data, xG/xGA, "
    "injuries, lineups and bookmaker odds."
)
