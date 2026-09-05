import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Football Prediction Analyzer", page_icon="⚽", layout="wide")

BASE_URL = "https://v3.football.api-sports.io"

# -----------------------------
# API-Football connection
# -----------------------------

def get_api_key() -> str:
    try:
        return st.secrets["API_FOOTBALL_KEY"]
    except Exception:
        return ""


@st.cache_data(ttl=300, show_spinner=False)
def api_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    key = get_api_key()
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY is missing from Streamlit Secrets.")

    r = requests.get(
        f"{BASE_URL}/{endpoint.lstrip('/')}",
        headers={"x-apisports-key": key},
        params=params,
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()

    if data.get("errors"):
        raise RuntimeError(str(data["errors"]))
    return data


@st.cache_data(ttl=86400, show_spinner=False)
def search_team(name: str) -> List[Dict[str, Any]]:
    data = api_get("teams", {"search": name.strip()})
    return data.get("response", [])


def pick_team(name: str) -> Tuple[int, str, str]:
    results = search_team(name)
    if not results:
        raise RuntimeError(f'No team found for "{name}".')
    # Prefer an exact name match, otherwise use the first API result.
    exact = [x for x in results if x.get("team", {}).get("name", "").lower() == name.strip().lower()]
    item = exact[0] if exact else results[0]
    team = item.get("team", {})
    return int(team["id"]), team.get("name", name), team.get("logo", "")


@st.cache_data(ttl=3600, show_spinner=False)
def find_fixture(home_id: int, away_id: int) -> Optional[Dict[str, Any]]:
    # Pull each team's upcoming fixtures and find the requested pairing.
    home_f = api_get("fixtures", {"team": home_id, "next": 20}).get("response", [])
    for f in home_f:
        h = f.get("teams", {}).get("home", {}).get("id")
        a = f.get("teams", {}).get("away", {}).get("id")
        if h == home_id and a == away_id:
            return f
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def next_fixtures(team_id: int, n: int = 10) -> List[Dict[str, Any]]:
    return api_get("fixtures", {"team": team_id, "next": n}).get("response", [])


@st.cache_data(ttl=3600, show_spinner=False)
def last_fixtures(team_id: int, n: int = 10) -> List[Dict[str, Any]]:
    return api_get("fixtures", {"team": team_id, "last": n}).get("response", [])


@st.cache_data(ttl=21600, show_spinner=False)
def team_statistics(team_id: int, league_id: int, season: int) -> Dict[str, Any]:
    data = api_get("teams/statistics", {"team": team_id, "league": league_id, "season": season})
    return data.get("response", {})


@st.cache_data(ttl=3600, show_spinner=False)
def standings(league_id: int, season: int) -> List[Dict[str, Any]]:
    data = api_get("standings", {"league": league_id, "season": season})
    rows = []
    for block in data.get("response", []):
        for table in block.get("league", {}).get("standings", []):
            rows.extend(table)
    return rows


@st.cache_data(ttl=900, show_spinner=False)
def fixture_odds(fixture_id: int) -> List[Dict[str, Any]]:
    return api_get("odds", {"fixture": fixture_id}).get("response", [])


@st.cache_data(ttl=1800, show_spinner=False)
def fixture_predictions(fixture_id: int) -> List[Dict[str, Any]]:
    return api_get("predictions", {"fixture": fixture_id}).get("response", [])


# -----------------------------
# Model mathematics
# -----------------------------

def implied(odds: float) -> float:
    return 1.0 / odds if odds and odds > 0 else 0.0


def normalize(odds: List[float]) -> List[float]:
    raw = [implied(o) for o in odds]
    total = sum(raw)
    return [x / total for x in raw] if total else [0.0] * len(raw)


def fair_odds(p: float) -> float:
    return 1.0 / p if p > 0 else float("inf")


def ev(p: float, odds: float) -> float:
    return p * odds - 1.0


def poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def model(lh: float, la: float, max_goals: int = 10) -> Dict[str, Any]:
    scores = {(h, a): poisson(h, lh) * poisson(a, la)
              for h in range(max_goals + 1) for a in range(max_goals + 1)}
    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}

    hw = sum(p for (h, a), p in scores.items() if h > a)
    draw = sum(p for (h, a), p in scores.items() if h == a)
    aw = sum(p for (h, a), p in scores.items() if h < a)
    over15 = sum(p for (h, a), p in scores.items() if h + a >= 2)
    over25 = sum(p for (h, a), p in scores.items() if h + a >= 3)
    over35 = sum(p for (h, a), p in scores.items() if h + a >= 4)
    btts = sum(p for (h, a), p in scores.items() if h >= 1 and a >= 1)
    return {
        "scores": scores, "home": hw, "draw": draw, "away": aw,
        "over15": over15, "under15": 1 - over15,
        "over25": over25, "under25": 1 - over25,
        "over35": over35, "under35": 1 - over35,
        "btts": btts, "no_btts": 1 - btts,
        "top": sorted(scores.items(), key=lambda x: x[1], reverse=True),
    }


def safe_num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def extract_stat(stats: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = stats.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            value = value.get("value")
        n = safe_num(value)
        if n is not None:
            return n
    return None


def goals_summary(stats: Dict[str, Any], side: str) -> Tuple[Optional[float], Optional[float]]:
    goals = stats.get("goals", {}) if isinstance(stats, dict) else {}
    block = goals.get(side, {}) if isinstance(goals, dict) else {}
    gf = safe_num(block.get("for", {}).get("average"))
    ga = safe_num(block.get("against", {}).get("average"))
    return gf, ga


def xg_from_stats(stats: Dict[str, Any], side: str) -> Optional[float]:
    # API coverage varies. Accept common xG locations if present.
    for root_name in ("goals", "shots", "xg", "expected_goals"):
        root = stats.get(root_name)
        if isinstance(root, dict):
            candidates = root.get(side)
            if isinstance(candidates, dict):
                for key in ("xg", "expected", "average", "value"):
                    n = safe_num(candidates.get(key))
                    if n is not None and 0 <= n <= 10:
                        return n
    return None


def form_from_fixtures(fixtures: List[Dict[str, Any]], team_id: int) -> Dict[str, Any]:
    rows = []
    gf = ga = 0
    wins = draws = losses = 0
    for f in fixtures:
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        h = teams.get("home", {}).get("id")
        a = teams.get("away", {}).get("id")
        hg, ag = goals.get("home"), goals.get("away")
        if hg is None or ag is None:
            continue
        scored = hg if h == team_id else ag
        conceded = ag if h == team_id else hg
        gf += scored; ga += conceded
        if scored > conceded: result = "W"; wins += 1
        elif scored == conceded: result = "D"; draws += 1
        else: result = "L"; losses += 1
        rows.append(result)
    n = len(rows)
    return {
        "form": "".join(rows), "n": n, "gf": gf, "ga": ga,
        "gf_avg": gf / n if n else None, "ga_avg": ga / n if n else None,
        "wins": wins, "draws": draws, "losses": losses,
    }


def home_away_from_fixtures(fixtures: List[Dict[str, Any]], team_id: int, is_home: bool) -> Dict[str, Any]:
    selected = []
    for f in fixtures:
        teams = f.get("teams", {})
        h = teams.get("home", {}).get("id")
        a = teams.get("away", {}).get("id")
        if (h == team_id) == is_home:
            selected.append(f)
    return form_from_fixtures(selected, team_id)


def estimate_lambdas(home_id: int, away_id: int, league_id: int, season: int,
                     home_stats: Dict[str, Any], away_stats: Dict[str, Any],
                     home_recent: List[Dict[str, Any]], away_recent: List[Dict[str, Any]]) -> Tuple[float, float, Dict[str, Any]]:
    # Primary estimate: recent home/away scoring and conceding, with season statistics
    # as stabilizers. xG is used if the API supplies it; otherwise the goal averages remain.
    hs_home = home_away_from_fixtures(home_recent, home_id, True)
    as_away = home_away_from_fixtures(away_recent, away_id, False)
    h_gf_season, h_ga_season = goals_summary(home_stats, "for"), goals_summary(home_stats, "against")
    a_gf_season, a_ga_season = goals_summary(away_stats, "for"), goals_summary(away_stats, "against")

    h_gf = hs_home.get("gf_avg") or (h_gf_season[0] if h_gf_season else None)
    h_ga = hs_home.get("ga_avg") or (h_ga_season[0] if h_ga_season else None)
    a_gf = as_away.get("gf_avg") or (a_gf_season[0] if a_gf_season else None)
    a_ga = as_away.get("ga_avg") or (a_ga_season[0] if a_ga_season else None)

    # Blend recent split with season split where available.
    h_attack = 0.65 * (h_gf or 1.2) + 0.35 * (h_gf_season[0] or h_gf or 1.2)
    a_def = 0.65 * (a_ga or 1.2) + 0.35 * (a_ga_season[1] or a_ga or 1.2)
    a_attack = 0.65 * (a_gf or 1.0) + 0.35 * (a_gf_season[0] or a_gf or 1.0)
    h_def = 0.65 * (h_ga or 1.0) + 0.35 * (h_ga_season[1] or h_ga or 1.0)

    lh = (h_attack + a_def) / 2.0
    la = (a_attack + h_def) / 2.0

    # Gentle xG adjustment if present in API data. Never let it dominate.
    hxg = xg_from_stats(home_stats, "for")
    axg = xg_from_stats(away_stats, "for")
    if hxg is not None:
        lh = 0.75 * lh + 0.25 * hxg
    if axg is not None:
        la = 0.75 * la + 0.25 * axg

    lh = min(max(lh, 0.20), 4.50)
    la = min(max(la, 0.20), 4.00)
    details = {"home_split": hs_home, "away_split": as_away, "home_xg": hxg, "away_xg": axg}
    return lh, la, details


def odds_from_api(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {"1X2": {}, "OU25": {}, "BTTS": {}}
    # Prefer the first bookmaker with the main/common markets.
    for bookmaker_block in rows:
        for bet in bookmaker_block.get("bookmakers", []):
            bets = bet.get("bets", [])
            for b in bets:
                name = (b.get("name") or "").lower()
                values = b.get("values", [])
                if "match winner" in name:
                    for v in values:
                        label = (v.get("value") or "").lower()
                        odd = safe_num(v.get("odd"))
                        if odd:
                            if label in ("home", "1"): out["1X2"]["home"] = odd
                            elif label in ("draw", "x"): out["1X2"]["draw"] = odd
                            elif label in ("away", "2"): out["1X2"]["away"] = odd
                elif "over/under 2.5" in name or "over/under 2.5 goals" in name:
                    for v in values:
                        label = (v.get("value") or "").lower()
                        odd = safe_num(v.get("odd"))
                        if odd:
                            if "over" in label: out["OU25"]["over"] = odd
                            if "under" in label: out["OU25"]["under"] = odd
                elif "both teams to score" in name:
                    for v in values:
                        label = (v.get("value") or "").lower()
                        odd = safe_num(v.get("odd"))
                        if odd:
                            if label == "yes": out["BTTS"]["yes"] = odd
                            if label == "no": out["BTTS"]["no"] = odd
            if len(out["1X2"]) >= 3:
                return out
    return out


def assessment(p: float, odds: Optional[float]) -> str:
    if not odds:
        return "—"
    value = ev(p, odds)
    if value >= 0.08: return "🟢 Strong value"
    if value >= 0.03: return "🟡 Possible value"
    if value >= 0: return "⚪ Tiny edge"
    return "🔴 No value"


def pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


# -----------------------------
# UI
# -----------------------------

st.title("⚽ Football Prediction Analyzer")
st.caption("API-Football data → independent model → fair odds → value → final pick")

api_key = get_api_key()
if not api_key:
    st.error("API_FOOTBALL_KEY is not available. Add it in Streamlit Community Cloud → App settings → Secrets.")
    st.stop()

with st.sidebar:
    st.header("Match")
    home_input = st.text_input("Home team", "LDU Quito")
    away_input = st.text_input("Away team", "Mushuc Runa")
    analyze = st.button("🔎 Analyze match", type="primary", use_container_width=True)
    st.divider()
    st.caption("The app finds the upcoming fixture between the two teams. If several matches exist, use exact team names.")
    max_goals = st.slider("Maximum model goals", 6, 12, 10)

if analyze or "analysis" not in st.session_state:
    try:
        with st.spinner("Finding teams and fixture..."):
            home_id, home_name, home_logo = pick_team(home_input)
            away_id, away_name, away_logo = pick_team(away_input)
            fixture = find_fixture(home_id, away_id)

        if not fixture:
            st.warning("I could not find an upcoming fixture between these two teams in the next 20 fixtures returned for the home team.")
            st.info("Check the team names, or wait until the fixture is available in API-Football.")
            st.stop()

        league = fixture.get("league", {})
        fixture_id = fixture.get("fixture", {}).get("id")
        league_id = int(league.get("id"))
        season = int(league.get("season"))

        with st.spinner("Fetching form, home/away splits, standings and odds..."):
            hs = team_statistics(home_id, league_id, season)
            a_s = team_statistics(away_id, league_id, season)
            h_recent = last_fixtures(home_id, 10)
            a_recent = last_fixtures(away_id, 10)
            odds_rows = fixture_odds(fixture_id)
            api_pred_rows = fixture_predictions(fixture_id)
            table = standings(league_id, season)

        lh, la, details = estimate_lambdas(home_id, away_id, league_id, season, hs, a_s, h_recent, a_recent)
        m = model(lh, la, max_goals)
        odds = odds_from_api(odds_rows)

        st.session_state.analysis = {
            "fixture": fixture, "home_name": home_name, "away_name": away_name,
            "home_id": home_id, "away_id": away_id, "lh": lh, "la": la,
            "model": m, "odds": odds, "details": details,
            "home_stats": hs, "away_stats": a_s, "home_recent": h_recent,
            "away_recent": a_recent, "api_pred": api_pred_rows, "table": table,
            "league": league, "fixture_id": fixture_id,
        }
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

A = st.session_state.analysis
fixture = A["fixture"]
m = A["model"]
odds = A["odds"]

kick = fixture.get("fixture", {}).get("date", "")
venue = fixture.get("fixture", {}).get("venue", {}).get("name")
league = A["league"].get("name", "")
country = A["league"].get("country", "")

st.subheader(f"{A['home_name']} vs {A['away_name']}")
st.write(f"**{league} — {country}**")
st.write(f"Kickoff: **{kick}**" + (f" · {venue}" if venue else ""))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Home λ", f"{A['lh']:.2f}")
c2.metric("Away λ", f"{A['la']:.2f}")
c3.metric("Expected goals", f"{A['lh'] + A['la']:.2f}")
c4.metric("BTTS", pct(m["btts"]))

# 1X2 analysis
st.header("1. Main prediction")
probs = [m["home"], m["draw"], m["away"]]
names = [f"{A['home_name']} win", "Draw", f"{A['away_name']} win"]
main_idx = max(range(3), key=lambda i: probs[i])
main_name = names[main_idx]
st.success(f"Most likely result: **{main_name} — {probs[main_idx]*100:.1f}%**")

rows = []
market_keys = [("Home win", m["home"], odds["1X2"].get("home")),
               ("Draw", m["draw"], odds["1X2"].get("draw")),
               ("Away win", m["away"], odds["1X2"].get("away")),
               ("Over 2.5", m["over25"], odds["OU25"].get("over")),
               ("Under 2.5", m["under25"], odds["OU25"].get("under")),
               ("BTTS Yes", m["btts"], odds["BTTS"].get("yes")),
               ("BTTS No", m["no_btts"], odds["BTTS"].get("no"))]
for name, p, o in market_keys:
    rows.append({"Market": name, "Model %": p*100, "Bookmaker odds": o if o else None,
                 "Fair odds": fair_odds(p), "EV %": ev(p, o)*100 if o else None,
                 "Assessment": assessment(p, o)})

market_df = pd.DataFrame(rows)
st.dataframe(market_df.style.format({"Model %": "{:.1f}", "Bookmaker odds": "{:.2f}", "Fair odds": "{:.2f}", "EV %": "{:.1f}"}), use_container_width=True, hide_index=True)

# Best value selection
aavailable = [(n, p, o) for n, p, o in market_keys if o]
if available:
    best = max(available, key=lambda x: ev(x[1], x[2]))
    if ev(best[1], best[2]) >= 0.03:
        st.success(f"Best value found: **{best[0]} @ {best[2]:.2f}** — model {best[1]*100:.1f}% — EV {ev(best[1], best[2])*100:.1f}%.")
    else:
        st.info(f"No strong value found in the available main markets. Best available edge: **{best[0]} @ {best[2]:.2f}** (EV {ev(best[1], best[2])*100:.1f}%).")
else:
    st.info("No usable bookmaker odds were returned for the main markets. The probability model still works.")

# Form and context
st.header("2. Data used")
form_cols = st.columns(2)
for col, team, team_id, recent, split in [
    (form_cols[0], A["home_name"], A["home_id"], A["home_recent"], A["details"]["home_split"]),
    (form_cols[1], A["away_name"], A["away_id"], A["away_recent"], A["details"]["away_split"]),
]:
    f = form_from_fixtures(recent, team_id)
    with col:
        st.markdown(f"### {team}")
        st.write(f"Last 10: **{f['form'] or '—'}**")
        st.write(f"Last 10 goals: **{f['gf']}–{f['ga']}**")
        st.write(f"Relevant split: **{split['form'] or '—'}** · {split['gf_avg']:.2f} scored / {split['ga_avg']:.2f} conceded" if split['n'] else "Relevant home/away split not available")

# Standings position
pos = {}
for row in A["table"]:
    tid = row.get("team", {}).get("id")
    if tid in (A["home_id"], A["away_id"]):
        pos[tid] = row.get("rank")
if pos:
    st.write(f"Table position: **{A['home_name']} #{pos.get(A['home_id'], '—')}** · **{A['away_name']} #{pos.get(A['away_id'], '—')}**")

# Exact scores
top = m["top"][:10]
st.header("3. Exact-score probabilities")
topdf = pd.DataFrame({
    "Rank": range(1, len(top)+1),
    "Score": [f"{h}-{a}" for (h, a), _ in top],
    "Probability %": [p*100 for _, p in top],
    "Fair odds": [fair_odds(p) for _, p in top],
})
st.dataframe(topdf.style.format({"Probability %": "{:.2f}", "Fair odds": "{:.2f}"}), use_container_width=True, hide_index=True)

# API prediction comparison
if A["api_pred"]:
    p0 = A["api_pred"][0].get("predictions", {})
    st.header("4. Independent cross-check")
    api_home = p0.get("percent", {}).get("home")
    api_draw = p0.get("percent", {}).get("draw")
    api_away = p0.get("percent", {}).get("away")
    if api_home or api_draw or api_away:
        st.write(f"API-Football prediction cross-check: Home **{api_home or '—'}**, Draw **{api_draw or '—'}**, Away **{api_away or '—'}**.")
    if p0.get("advice"):
        st.write(f"API advice: **{p0['advice']}**")

# Final compact verdict
st.header("5. Final verdict")
score = top[0][0]
verdict = f"**{main_name}** is the most likely 1X2 outcome. Most likely score: **{score[0]}-{score[1]}**."
if available and ev(best[1], best[2]) >= 0.03:
    verdict += f" The strongest value among returned markets is **{best[0]} @ {best[2]:.2f}**."
else:
    verdict += " The model does **not** find a strong betting edge in the returned main markets."
st.write(verdict)
st.caption("Probability first. Odds second. Bet last. The model is a statistical baseline, not a guarantee.")

with st.expander("Technical details"):
    st.write("λ is estimated from recent home/away scoring and conceding, season statistics, and a gentle xG adjustment when xG is available. Poisson then converts λ into score and market probabilities.")
    st.write("API-Football coverage varies by league, so missing xG, injuries, lineups or odds are handled rather than invented.")
