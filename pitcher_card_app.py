# pitcher_card_app.py
# ------------------------------------------------------------
# Streamlit app for MLB Season Pitching Dashboards
# - Start = Mar 18, 2025 (first game of season)
# - End   = yesterday (America/Los_Angeles), cannot pick beyond yesterday
# - Choose pitcher from current rosters (MLB + MiLB; pitchers only)
# - Calls mlb_pitcher_card.pitching_dashboard()
# - Spring/Exhibitions excluded inside the module
# - Graceful "no MLB data" messaging instead of errors
# ------------------------------------------------------------

import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st

from mlb_pitcher_card import pitching_dashboard  # must return a Matplotlib Figure

st.set_page_config(page_title="MLB Season Pitching Dashboard", layout="wide")
st.title("MLB Season Pitching Dashboard")

# -------------------- Roster helpers (unchanged: pitchers only) --------------------

SPORT_LEVELS = {
    1:  "MLB",
    11: "AAA",
    12: "AA",
    13: "A+",
    14: "A",
    16: "Rookie",
}
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId={sport_id}&activeStatus=Y"
ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"

@st.cache_data(show_spinner=True)
def load_pitchers_all_levels(max_workers: int = 24) -> pd.DataFrame:
    """
    Load active rosters for MLB + MiLB levels and return:
    columns = [key_mlbam, full_name, team, team_id, position, team_level]
    (Only pitchers kept.)
    """
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "FlashStats/1.0",
    })

    # 1) Fetch all active teams per level
    team_meta = []
    for sport_id, level_label in SPORT_LEVELS.items():
        url = TEAMS_URL.format(sport_id=sport_id)
        try:
            r = session.get(url, params={"fields": "teams,id,name"}, timeout=10)
            teams = (r.json() or {}).get("teams", []) or []
        except Exception:
            teams = []

        for t in teams:
            tid = t.get("id")
            tname = t.get("name")
            if tid and tname:
                team_meta.append({"id": int(tid), "name": tname, "level": level_label})

    if not team_meta:
        return pd.DataFrame(columns=["key_mlbam","full_name","team","team_id","position","team_level"])

    # 2) Fetch active rosters in parallel
    def _fetch_team_roster(team: dict) -> list[dict]:
        url = ROSTER_URL.format(team_id=team["id"])
        try:
            r = session.get(
                url,
                params={"fields": "roster,person,id,fullName,position,abbreviation"},
                timeout=10,
            )
            roster = (r.json() or {}).get("roster", []) or []
        except Exception:
            roster = []

        out = []
        for row in roster:
            pos = ((row.get("position") or {}).get("abbreviation"))
            if pos != "P":
                continue
            person = row.get("person") or {}
            pid = person.get("id")
            pname = person.get("fullName")
            if not pid or not pname:
                continue
            out.append({
                "key_mlbam": int(pid),
                "full_name": pname,
                "team": team["name"],
                "team_id": int(team["id"]),
                "position": "Pitcher",
                "team_level": team["level"],
            })
        return out

    rows: list[dict] = []
    max_workers = max(8, min(max_workers, 32))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for team_rows in ex.map(_fetch_team_roster, team_meta):
            if team_rows:
                rows.extend(team_rows)

    if not rows:
        return pd.DataFrame(columns=["key_mlbam","full_name","team","team_id","position","team_level"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["key_mlbam", "team_level"])
    level_order = pd.CategoricalDtype(categories=["MLB","AAA","AA","A+","A","Rookie"], ordered=True)
    df["team_level"] = df["team_level"].astype(level_order)
    return df.sort_values(["team_level","team","full_name"]).reset_index(drop=True)

# -------------------- Date defaults (LA time) --------------------

def _yesterday_los_angeles() -> pd.Timestamp:
    return (pd.Timestamp.now(tz="America/Los_Angeles").normalize() - pd.Timedelta(days=1))

SEASON_START = pd.Timestamp(2025, 3, 18).date()  # first game of the 2025 MLB season
YESTERDAY_LA_DATE = _yesterday_los_angeles().date()

# -------------------- Sidebar --------------------

with st.sidebar:
    st.header("Filters")

    # Dates: start fixed default = Mar 18, 2025; end default = yesterday (LA).
    start_date = st.date_input(
        "Start date",
        value=SEASON_START,
        min_value=pd.Timestamp(2015,1,1).date(),
        max_value=YESTERDAY_LA_DATE,                 # cannot exceed yesterday
        format="MM/DD/YYYY",
        help="Default start on Opening Day. Spring Training is excluded",
    )
    end_date = st.date_input(
        "End date",
        value=YESTERDAY_LA_DATE,
        max_value=YESTERDAY_LA_DATE,                 # cannot exceed yesterday
        format="MM/DD/YYYY",
        help="End date cannot be after yesterday",
    )

    # Load all current rosters (MLB + MiLB; pitchers only)
    df_pitchers = load_pitchers_all_levels()

    if df_pitchers.empty:
        st.error("Couldn’t load pitchers. Click Refresh.")
        st.stop()

    df_pitchers["key_mlbam"] = pd.to_numeric(df_pitchers["key_mlbam"], errors="coerce").astype("Int64")

    # Level selector
    level_order_all = ["MLB", "AAA", "AA", "A+", "A", "Rookie"]
    levels = [lvl for lvl in level_order_all if lvl in df_pitchers["team_level"].dropna().unique()]
    level_default_idx = levels.index("MLB") if "MLB" in levels else 0
    level = st.selectbox("Level", levels, index=level_default_idx)

    # Team selector (within chosen level)
    teams = sorted(df_pitchers.loc[df_pitchers["team_level"] == level, "team"].dropna().unique().tolist())
    team = st.selectbox("Team", teams) if teams else None

    # Pitcher selector (within chosen team)
    if team:
        pframe = (
            df_pitchers.loc[
                (df_pitchers["team_level"] == level) & (df_pitchers["team"] == team),
                ["key_mlbam", "full_name"],
            ]
            .dropna(subset=["key_mlbam", "full_name"])
            .assign(key_mlbam=lambda d: d["key_mlbam"].astype("int64"))
            .sort_values("full_name")
        )
        pitcher_options = list(pframe.apply(lambda r: (int(r["key_mlbam"]), r["full_name"]), axis=1))
    else:
        pitcher_options = []

    if not pitcher_options:
        st.info("No pitchers match the current filter.")
        pitcher_id = None
    else:
        selected_pitcher = st.selectbox(
            "Pitcher",
            options=pitcher_options,
            format_func=(lambda t: t[1]),
        )
        pitcher_id = selected_pitcher[0] if selected_pitcher else None

    # Actions
    run = st.button("Generate", type="primary")

# -------------------- Main panel --------------------

def render_dashboard(pitcher_id: int, start_str: str, end_str: str):
    try:
        with st.spinner("Building pitcher dashboard..."):
            fig = pitching_dashboard(
                pitcher_id=int(pitcher_id),
                start_dt=start_str,
                end_dt=end_str,
            )
            # If your dashboard function returns None or similar for no data:
            if fig is None:
                st.info("No MLB Statcast data found for this player in the selected date range.")
                return
            st.pyplot(fig, width='stretch')

        # optional download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button("Download PNG", buf.getvalue(), "pitcher_dashboard.png", "image/png")

    except Exception as e:
        # Show a friendly message instead of throwing
        st.info("No MLB Statcast data found for this player in the selected date range.")
        # If you want to debug occasionally, uncomment the line below
        # st.caption(f"Details: {e}")

# Validate dates and render
if run:
    if pitcher_id is None:
        st.warning("Choose a pitcher.")
    else:
        # Guardrails: enforce end_date <= yesterday (already in widget), and start <= end
        if end_date > YESTERDAY_LA_DATE:
            st.error("End date cannot be after yesterday (Los Angeles time).")
        elif start_date > end_date:
            st.error("Start date must be on or before End date.")
        else:
            # Format dates like 09/08/2025 in the UI, but pass ISO to your backend
            start_iso = pd.to_datetime(start_date).date().isoformat()
            end_iso   = pd.to_datetime(end_date).date().isoformat()
            render_dashboard(pitcher_id, start_iso, end_iso)
else:
    st.info("Pick a level → team → pitcher, set dates, then click **Generate**.")
