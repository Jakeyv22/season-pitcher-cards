# mlb_pitcher_card.py
# -----------------------------------------------------------------------------
# Build a single Matplotlib Figure for a pitcher dashboard.
# Public entry point:
#     pitching_dashboard(pitcher_id: int, start_dt: str, end_dt: str) -> Figure|None
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from PIL import Image
from io import BytesIO
import cairosvg
import pybaseball as pyb

from pathlib import Path
import os

# Your local OneDrive folder (fallback for running on your PC)
LEGACY_DATA_DIR = Path(r"C:\Users\jakev\OneDrive\Documents\FlashStats\Statcast")

def resolve_data(filename: str, override: str | Path | None = None) -> Path:
    """
    Find a data file in common locations:
    - explicit override path
    - repo ./data/<filename> (works on Streamlit Cloud)
    - current working dir ./data/<filename> (works in Streamlit run)
    - same folder as this module
    - your local OneDrive Statcast folder (legacy fallback)
    - PITCH_CARDS_DATA_DIR env var (optional)
    """
    # 1) explicit override
    if override:
        p = Path(override)
        if p.exists():
            return p

    # 2) env var directory (optional)
    env_dir = os.getenv("PITCH_CARDS_DATA_DIR")
    # build candidate list
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / filename,
        Path.cwd() / "data" / filename,
        here / filename,
        (Path(env_dir) / filename) if env_dir else None,
        LEGACY_DATA_DIR / filename,
    ]

    for p in candidates:
        if p and p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find {filename}. Looked in:\n" +
        "\n".join(str(c) for c in candidates if c)
    )

# replaces the hard-coded C:\ path
STATCAST_CSV = resolve_data("statcast_2025_grouped.csv")
df_statcast_group = pd.read_csv(STATCAST_CSV)

# ---------- Matplotlib defaults ----------
# Define font properties for general text
font_properties = {'family': 'DejaVu Sans', 'size': 12}

# Define font properties for titles
font_properties_titles = {'family': 'DejaVu Sans', 'size': 20}

# Define font properties for axes labels
font_properties_axes = {'family': 'DejaVu Sans', 'size': 16}

# Set the theme for seaborn plots
sns.set_theme(style='whitegrid', 
              palette='deep', 
              font='DejaVu Sans', 
              font_scale=1.5, 
              color_codes=True, 
              rc=None)

# Import matplotlib
import matplotlib as mpl

# Set the resolution of the figures to 300 DPI
mpl.rcParams['figure.dpi'] = 300

# ---------- Color palette & pitch dictionaries ----------
pitch_colors = {
    # Fastballs
    "FF": {"color": "#C21014", "name": "4-Seam Fastball"},
    "FA": {"color": "#C21014", "name": "Fastball"},
    "SI": {"color": "#F4B400", "name": "Sinker"},
    "FC": {"color": "#993300", "name": "Cutter"},
    # Offspeed
    "CH": {"color": "#00B386", "name": "Changeup"},
    "FS": {"color": "#66CCCC", "name": "Splitter"},
    "SC": {"color": "#33CC99", "name": "Screwball"},
    "FO": {"color": "#339966", "name": "Forkball"},
    # Sliders
    "SL": {"color": "#FFCC00", "name": "Slider"},
    "ST": {"color": "#CCCC66", "name": "Sweeper"},
    "SV": {"color": "#9999FF", "name": "Slurve"},
    # Curveballs
    "KC": {"color": "#0000CC", "name": "Knuckle Curve"},
    "CU": {"color": "#3399FF", "name": "Curveball"},
    "CS": {"color": "#66CCFF", "name": "Slow Curve"},
    # Knuckleball
    "KN": {"color": "#3333CC", "name": "Knuckleball"},
    # Others
    "EP": {"color": "#999966", "name": "Eephus"},
    "PO": {"color": "#CCCCCC", "name": "Pitchout"},
    "UN": {"color": "#9C8975", "name": "Unknown"},
}
dict_color = {k: v["color"] for k, v in pitch_colors.items()}
dict_pitch = {k: v["name"] for k, v in pitch_colors.items()}

# ---------- Small helpers ----------
def sdiv(a, b):
    """Safe divide with NaN result when denominator is 0/NaN."""
    return np.where((b == 0) | (~np.isfinite(b)), np.nan, a / b)

def fmt_series(s: pd.Series, fmt: str) -> pd.Series:
    mask = s.notna() & np.isfinite(s)
    out = s.astype(object)
    out.loc[mask] = out.loc[mask].map(lambda x: format(float(x), fmt))
    out.loc[~mask] = "—"
    return out

# ---------- Statcast processing ----------
def df_processing(df_pyb: pd.DataFrame) -> pd.DataFrame:
    df = df_pyb.copy()
    swing_code = [
        "foul_bunt","foul","hit_into_play","swinging_strike","foul_tip",
        "swinging_strike_blocked","missed_bunt","bunt_foul_tip"
    ]
    whiff_code = ["swinging_strike","foul_tip","swinging_strike_blocked"]
    df["swing"] = df["description"].isin(swing_code)
    df["whiff"] = df["description"].isin(whiff_code)
    df["in_zone"] = (pd.to_numeric(df["zone"], errors="coerce") < 10)
    df["out_zone"] = (pd.to_numeric(df["zone"], errors="coerce") > 10)
    df["chase"] = (~df["in_zone"]) & (df["swing"] == 1)
    # convert movement to inches
    df["pfx_z"] = pd.to_numeric(df["pfx_z"], errors="coerce") * 12
    df["pfx_x"] = pd.to_numeric(df["pfx_x"], errors="coerce") * 12
    return df

def _final_pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    return out[out["events"].notna()].copy()

# ---------- Rolling xwOBA ----------
def rolling_xwoba(df_pitcher: pd.DataFrame, window_pas: int = 50) -> pd.DataFrame:
    pa = df_pitcher[df_pitcher["events"].notna()].copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa = pa.sort_values(
        ["game_date","game_pk","at_bat_number","pitch_number"]
    ).reset_index(drop=True)
    pa["xwoba_pa"] = pd.to_numeric(pa["estimated_woba_using_speedangle"], errors="coerce")
    pa["xwoba_roll"] = pa["xwoba_pa"].rolling(window_pas, min_periods=1).mean()
    return pa[["game_date","game_pk","at_bat_number","xwoba_pa","xwoba_roll"]]

def plot_rolling_xwoba(pa, window_pas=50, lg_avg=None, title="", ax=None, df_statcast_group=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    if pa.empty:
        ax.set_title("No data")
        return ax

    ax.plot(pa.index, pa["xwoba_roll"], lw=2)

    if lg_avg is None and df_statcast_group is not None:
        try:
            lg_avg = (
                df_statcast_group.loc[df_statcast_group["pitch_type"] == "All", "xwoba"]
                .iloc[0]
            )
        except Exception:
            lg_avg = None
    if lg_avg is None:
        lg_avg = pa["xwoba_pa"].mean()

    ax.axhline(lg_avg, ls="--", lw=1.2, color="black")
    ax.text(pa.index[-1], lg_avg, "     LG AVG", va="center", fontsize=12)

    ax.set_ylim(0.100, 0.500)
    ax.set_yticks([0.100,0.200,0.300,0.400,0.500])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}".lstrip("0")))
    ax.set_title(title or f"{window_pas} PAs Rolling xwOBA")
    ax.set_xlabel("Plate Appearances")

    # dotted horizontals at major y-ticks
    ax.set_axisbelow(True)  # draw grid beneath data
    ax.grid(False)          # kill any seaborn/mpl default grid
    ax.yaxis.grid(True, which='major', linestyle=':', linewidth=1, color='0.7', alpha=0.7)

    pa = pa.copy()
    pa["month"] = pd.to_datetime(pa["game_date"]).dt.to_period("M")
    month_starts = pa.groupby("month").head(1)
    for _, row in month_starts.iterrows():
        ax.axvline(row.name, color="black", ls=":", lw=1.5, alpha=0.5)
        ax.text(row.name, ax.get_ylim()[0], row["game_date"].strftime("%b"),
                rotation=90, va="bottom", ha="right", fontsize=16, color="gray")
    return ax

# ---------- Bio & imagery ----------
def player_headshot(pitcher_id: int, ax: plt.Axes):
    url = (
        "https://img.mlbstatic.com/mlb-photos/image/"
        "upload/d_people:generic:headshot:67:current.png"
        f"/w_640,q_auto:best/v1/people/{pitcher_id}/headshot/silo/current.png"
    )
    r = requests.get(url, timeout=10)
    img = Image.open(BytesIO(r.content))
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1)
    ax.imshow(img, extent=[0, 1, 0, 1], origin="upper")
    ax.axis("off")

def player_bio(pitcher_id: int, start_dt: str, end_dt: str, ax: plt.Axes):
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=currentTeam"
    data = requests.get(url, timeout=10).json()
    p = (data.get("people") or [{}])[0]
    full_name = p.get("fullName", "Unknown")
    hand = ((p.get("pitchHand") or {}).get("code") or "?")
    age = p.get("currentAge", "?")
    height = p.get("height", "?")
    weight = p.get("weight", "?")

    fontsize = 56 if len(str(full_name)) < 18 else 42
    ax.text(0.5, 1, full_name, va="top", ha="center",
            fontsize=fontsize, fontweight="bold", fontfamily="DejaVu Sans")
    ax.text(0.5, 0.65, f"{hand}HP | Age: {age} | {height} | {weight} lbs",
            va="top", ha="center", fontsize=30, fontfamily="Arial")
    ax.text(0.5, 0.40, "Pitching Summary",
            va="top", ha="center", fontsize=40, fontweight="bold", fontfamily="Georgia")
    ax.text(0.5, 0.15, f"{start_dt} to {end_dt}",
            va="top", ha="center", fontsize=30, fontstyle="italic")
    ax.axis("off")

def plot_logo(batter_id: int, ax: plt.Axes, size: int = 300):
    try:
        p = requests.get(
            f"https://statsapi.mlb.com/api/v1/people?personIds={batter_id}&hydrate=currentTeam",
            timeout=10
        ).json()["people"][0]
        team_id = (p.get("currentTeam") or {}).get("id")
        if not team_id:
            ax.axis("off")
            return
        svg = requests.get(f"https://www.mlbstatic.com/team-logos/{team_id}.svg", timeout=10).content
        png_bytes = cairosvg.svg2png(bytestring=svg, output_width=size, output_height=size)
        img = Image.open(BytesIO(png_bytes))
        ax.imshow(img, interpolation="nearest", origin="upper", aspect="equal")
        ax.axis("off")
    except Exception:
        ax.axis("off")

# ---------- Break plot ----------
def break_plot(df: pd.DataFrame, ax: plt.Axes, df_statcast_group: Optional[pd.DataFrame] = None):
    throws = (df["p_throws"].iloc[0] if "p_throws" in df.columns and not df.empty else "R")
    if throws == "R":
        x = -1 * df["pfx_x"]
    else:
        x = df["pfx_x"]

    import seaborn as sns
    sns.scatterplot(
        ax=ax, x=x, y=df["pfx_z"], hue=df["pitch_type"], palette=dict_color, ec="black", alpha=1, zorder=2
    )

    if df_statcast_group is not None and "p_throws" in df.columns:
        pitcher_hand = df["p_throws"].iloc[0]
        for pitch_type in df["pitch_type"].dropna().unique():
            m = df_statcast_group[
                (df_statcast_group["pitch_type"] == pitch_type) &
                (df_statcast_group.get("p_throws","R") == pitcher_hand)
            ]
            if m.empty:
                continue
            row = m.iloc[0]
            league_x = -row["pfx_x"] if pitcher_hand == "R" else row["pfx_x"]
            league_y = row["pfx_z"]
            color = dict_color.get(pitch_type, "gray")
            ax.add_patch(Ellipse(
                xy=(league_x, league_y),
                width=7, height=7, angle=0,
                edgecolor=color, facecolor=color, alpha=0.5, lw=2, zorder=0
            ))

    ax.axhline(0, color="#808080", alpha=0.5, ls="--", zorder=1)
    ax.axvline(0, color="#808080", alpha=0.5, ls="--", zorder=1)
    ax.set_xlabel("Horizontal Break (in)")
    ax.set_ylabel("Induced Vertical Break (in)")

    # Add title and subtitle to plot
    if 'arm_angle' in df.columns:
        avg_angle = df['arm_angle'].mean()
    
        # Set plot title
        title = f"Pitch Breaks - Arm Angle: {avg_angle:.0f}°"

        # Set title with extra padding to make room for subtitle
        ax.set_title(title, fontdict=font_properties_titles, pad=25)
        
        # Additional Note: MLB average movement ellipses
        ax.text(0.5, 1.02, "Note: Ellipses = League average pitch movement",
            transform=ax.transAxes,
            ha='center',
            fontsize=11,
            style='italic',
            color='dimgray',
            zorder=4)

    ax.set_xticks(range(-20, 21, 10))
    ax.set_yticks(range(-20, 21, 10))
    ax.set_xlim((-25, 25))
    ax.set_ylim((-25, 25))
    if throws == "R":
        ax.text(-24.2, -24.2, "← Glove Side", fontstyle="italic", ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black"), fontsize=10, zorder=3)
        ax.text(24.2, -24.2, "Arm Side →", fontstyle="italic", ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black"), fontsize=10, zorder=3)
    else:
        ax.invert_xaxis()
        ax.text(24.2, -24.2, "← Arm Side", fontstyle="italic", ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black"), fontsize=10, zorder=3)
        ax.text(-24.2, -24.2, "Glove Side →", fontstyle="italic", ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black"), fontsize=10, zorder=3)
        
    # Add dashed arm angle line from the origin
    if 'arm_angle' in df.columns and not df['arm_angle'].isnull().all():
        mean_angle_deg = df['arm_angle'].mean()
        mean_angle_rad = np.deg2rad(mean_angle_deg)  # Don't subtract from π

        length = 35  # Length of the line

        # Compute end coordinates regardless of throwing hand
        x_end = length * np.cos(mean_angle_rad)
        y_end = length * np.sin(mean_angle_rad)

        ax.plot([0, x_end], [0, y_end], linestyle='--', color='black', alpha=0.7, label='Arm Angle')
  

    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: int(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: int(v)))
    ax.get_legend().remove()

# ---------- FanGraphs (season table + percentiles) ----------
def fangraphs_pitching_leaderboards(season: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    season = pd.to_datetime(end_dt).year
    url = (
        "https://www.fangraphs.com/api/leaders/major-league/data"
        f"?age=&pos=all&stats=pit&lg=all&season={season}&season1={season}"
        "&ind=0&qual=0&type=8&month=1000"
        f"&startdate={start_dt}&enddate={end_dt}&pageitems=2500"
    )
    data = requests.get(url, timeout=20).json()
    return pd.DataFrame((data or {}).get("data", []))

fangraphs_stats_dict = {
    "IP":{"table_header":"$\\bf{IP}$","format":".1f"},
    "TBF":{"table_header":"$\\bf{PA}$","format":".0f"},
    "AVG":{"table_header":"$\\bf{AVG}$","format":".3f"},
    "K/9":{"table_header":"$\\bf{K\\/9}$","format":".2f"},
    "BB/9":{"table_header":"$\\bf{BB\\/9}$","format":".2f"},
    "K/BB":{"table_header":"$\\bf{K\\/BB}$","format":".2f"},
    "HR/9":{"table_header":"$\\bf{HR\\/9}$","format":".2f"},
    "K%":{"table_header":"$\\bf{K\\%}$","format":".1%"},
    "BB%":{"table_header":"$\\bf{BB\\%}$","format":".1%"},
    "K-BB%":{"table_header":"$\\bf{K-BB\\%}$","format":".1%"},
    "WHIP":{"table_header":"$\\bf{WHIP}$","format":".2f"},
    "BABIP":{"table_header":"$\\bf{BABIP}$","format":".3f"},
    "GB%":{"table_header":"$\\bf{GB\\%}$","format":".1%"},
    "LOB%":{"table_header":"$\\bf{LOB\\%}$","format":".1%"},
    "xFIP":{"table_header":"$\\bf{xFIP}$","format":".2f"},
    "FIP":{"table_header":"$\\bf{FIP}$","format":".2f"},
    "SIERA":{"table_header":"$\\bf{SIERA}$","format":".2f"},
    "H":{"table_header":"$\\bf{H}$","format":".0f"},
    "2B":{"table_header":"$\\bf{2B}$","format":".0f"},
    "3B":{"table_header":"$\\bf{3B}$","format":".0f"},
    "R":{"table_header":"$\\bf{R}$","format":".0f"},
    "ER":{"table_header":"$\\bf{ER}$","format":".0f"},
    "HR":{"table_header":"$\\bf{HR}$","format":".0f"},
    "BB":{"table_header":"$\\bf{BB}$","format":".0f"},
    "IBB":{"table_header":"$\\bf{IBB}$","format":".0f"},
    "HBP":{"table_header":"$\\bf{HBP}$","format":".0f"},
    "SO":{"table_header":"$\\bf{SO}$","format":".0f"},
    "OBP":{"table_header":"$\\bf{OBP}$","format":".0f"},
    "SLG":{"table_header":"$\\bf{SLG}$","format":".0f"},
    "ERA":{"table_header":"$\\bf{ERA}$","format":".2f"},
    "wOBA":{"table_header":"$\\bf{wOBA}$","format":".3f"},
    "G":{"table_header":"$\\bf{G}$","format":".0f"},
    "GS":{"table_header":"$\\bf{GS}$","format":".0f"},
    "sp_stuff":{"table_header":"$\\bf{Stuff+}$","format":".0f"},
    "sp_location":{"table_header":"$\\bf{Location+}$","format":".0f"},
    "sp_pitching":{"table_header":"$\\bf{Pitching+}$","format":".0f"},
}

def fangraphs_pitcher_stats(pitcher_id: int, ax: plt.Axes, stats: list,
                            season: int, start_dt: str, end_dt: str, fontsize: int = 20,
                            df_fangraphs: Optional[pd.DataFrame] = None):
    if df_fangraphs is None:
        df_fangraphs = fangraphs_pitching_leaderboards(season, start_dt, end_dt)

    # try to find pitcher row; if not, fill with dashes
    mask = pd.to_numeric(df_fangraphs.get("xMLBAMID"), errors="coerce") == int(pitcher_id)
    row = df_fangraphs.loc[mask, stats].reset_index(drop=True)
    if row.empty:
        row = pd.DataFrame([[ '---' for _ in stats ]], columns=stats)
    row = row.astype("object")

    # format
    def _fmt(col, val):
        if val == '---':
            return '---'
        spec = fangraphs_stats_dict.get(col, {}).get("format")
        try:
            return format(float(val), spec) if spec else val
        except Exception:
            return val
    row.loc[0] = [_fmt(col, row.at[0, col]) for col in row.columns]

    table = ax.table(
        cellText=row.values,
        colLabels=stats,
        cellLoc="center",
        bbox=[0.00, 0.0, 1, 1]
    )
    table.set_fontsize(fontsize)

    # pretty headers
    headers = [fangraphs_stats_dict.get(c, {}).get("table_header", c) for c in stats]
    for i, label in enumerate(headers):
        table.get_celld()[(0, i)].get_text().set_text(label)

    ax.axis("off")

def auto_min_tbf(start_dt: str, end_dt: str) -> int:
    """Heuristic PA threshold that scales with window length (days)."""
    days = (pd.to_datetime(end_dt) - pd.to_datetime(start_dt)).days + 1
    if days <= 7:    return 12
    if days <= 14:   return 25
    if days <= 30:   return 40
    if days <= 60:   return 50
    if days <= 90:   return 75
    return 100

def plot_percentile_rankings(
    pitcher_id: int,
    season: int,
    start_dt: str,
    end_dt: str,
    ax: Optional[plt.Axes] = None,
    min_tbf: Optional[int] = None,
    df_fangraphs: Optional[pd.DataFrame] = None,
):
    if df_fangraphs is None:
        df_fangraphs = fangraphs_pitching_leaderboards(season, start_dt, end_dt)

    # choose threshold from date window if not supplied
    if min_tbf is None:
        min_tbf = auto_min_tbf(start_dt, end_dt)

    label_map = {
        "SIERA":       "SIERA",
        "pfxvFA":      "Fastball Velo",
        "EV":          "Avg Exit Velo",
        "pfxZone%":    "Zone%",
        "pfxO-Swing%": "Chase%",
        "C+SwStr%":    "CSW%",
        "K%":          "K%",
        "BB%":         "BB%",
        "K-BB%":       "K-BB%",
        "Barrel%":     "Barrel%",
        "HardHit%":    "Hard-Hit%",
        "GB%":         "GB%",
    }
    percent_like = {"pfxZone%","pfxO-Swing%","C+SwStr%","K%","BB%","K-BB%","Barrel%","HardHit%","GB%"}

    def as_pct_points(x):
        """Return value as percentage points; if 0–1, scale to 0–100; keep NaN safe."""
        try:
            v = float(x)
        except Exception:
            return np.nan
        if 0.0 <= v <= 1.0:
            v *= 100.0
        return v

    df = df_fangraphs.copy()
    df["xMLBAMID"] = pd.to_numeric(df.get("xMLBAMID"), errors="coerce")
    row = df.loc[df["xMLBAMID"] == int(pitcher_id)]
    if row.empty:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No FanGraphs row for this pitcher/date range",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return ax
    pitcher_row = row.iloc[0]

    # reliability pool for percentiles
    df["TBF"] = pd.to_numeric(df.get("TBF"), errors="coerce")
    reliable = df.loc[df["TBF"] >= min_tbf].copy()
    if reliable.empty:
        reliable = df.copy()

    # percentiles (no display scaling here, we compare on raw FG scale)
    percentiles = {}
    for col in label_map.keys():
        if col not in df.columns:
            continue
        col_series = pd.to_numeric(reliable[col], errors="coerce")
        val = pd.to_numeric(pitcher_row[col], errors="coerce")
        percentiles[col] = float((col_series < val).mean() * 100.0)

    # lower-is-better metrics to flip
    for metric in ["SIERA","EV","Barrel%","HardHit%","BB%"]:
        if metric in percentiles:
            percentiles[metric] = 100.0 - percentiles[metric]

    # build plot data with baseball-aware display formatting
    values = []
    for k in label_map.keys():
        if k not in df.columns:
            continue
        v = pitcher_row[k]
        if k in percent_like:
            v = as_pct_points(v)
            values.append(round(v, 1))
        elif k in {"EV","pfxvFA"}:         # velocities (mph)
            values.append(round(float(v), 1))
        elif k == "SIERA":
            values.append(round(float(v), 2))
        else:
            values.append(round(float(v), 2))

    plot_data = pd.DataFrame({
        "Metric":     [label_map[k] for k in label_map if k in percentiles],
        "Percentile": [round(percentiles[k], 0) for k in label_map if k in percentiles],
        "Value":      values[:len([k for k in label_map if k in percentiles])],
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 0.65 * len(plot_data)))

    plot_data["Metric"] = pd.Categorical(plot_data["Metric"],
                                         categories=plot_data["Metric"],
                                         ordered=True)
    plot_data = plot_data.sort_values("Metric", ascending=False).reset_index(drop=True)

    y_pos = np.arange(len(plot_data))
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = plt.get_cmap("coolwarm")

    meets_min = bool(pd.to_numeric(pitcher_row["TBF"], errors="coerce") >= min_tbf)

    for i, r in plot_data.iterrows():
        color = cmap(norm(r["Percentile"])) if meets_min else "lightgray"
        ax.barh(i, r["Percentile"], color=color)

        if r["Percentile"] > 89:
            px, lc, ha = r["Percentile"] - 5, ("white" if meets_min else "black"), "right"
        else:
            px, lc, ha = r["Percentile"] + 1, ("black" if meets_min else "dimgray"), "left"
        ax.text(px, i, f"{int(r['Percentile'])}", va="center", ha=ha, color=lc)

        # right-side value label (already in correct baseball units)
        # --- Change: ensure SIERA always shows two decimals (e.g., 2.20) ---
        metric_label = r["Metric"]
        if metric_label == "SIERA":
            value_str = f"{float(r['Value']):.2f}"
        else:
            value_str = f"{r['Value']}"
        ax.text(105, i, value_str, va="center", ha="left")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["Metric"], fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_title("Percentile Rankings", fontsize=20)

    if not meets_min:
        ax.set_title("Percentile Rankings", fontsize=20, pad=25)
        ax.text(0.5, 1.02, f"Note: Grayed out — fewer than {min_tbf} TBF",
                transform=ax.transAxes, ha="center", fontsize=11, style="italic", color="dimgray")
    return ax

# ---------- Pitch table ----------
pitch_stats_dict = {
    "pitch": {"table_header": "$\\bf{Count}$", "format": ".0f"},
    "release_speed": {"table_header": "$\\bf{Velocity}$", "format": ".1f"},
    "release_speed_max": {"table_header": "$\\bf{Max}$", "format": ".1f"},
    "pfx_z": {"table_header": "$\\bf{iVB}$", "format": ".1f"},
    "pfx_x": {"table_header": "$\\bf{HB}$", "format": ".1f"},
    "release_pos_x": {'table_header': '$\\bf{hRel}$', 'format': '.1f'},
    "release_pos_z": {'table_header': '$\\bf{vRel}$', 'format': '.1f'},
    "release_spin_rate": {"table_header": "$\\bf{Spin}$", "format": ".0f"},
    "release_extension": {"table_header": "$\\bf{Ext.}$", "format": ".1f"},
    "xwoba": {"table_header": "$\\bf{xwOBA}$", "format": ".3f"},
    "xwobacon": {"table_header": "$\\bf{xwOBA}$\n$\\bf{con}$", "format": ".3f"},
    "pitch_usage": {"table_header": "$\\bf{Pitch\\%}$", "format": ".1%"},
    "whiff_rate": {"table_header": "$\\bf{Whiff\\%}$", "format": ".1%"},
    "in_zone_rate": {"table_header": "$\\bf{Zone\\%}$", "format": ".1%"},
    "chase_rate": {"table_header": "$\\bf{Chase\\%}$", "format": ".1%"},
    "delta_run_exp_per_100": {"table_header": "$\\bf{RV\//100}$", "format": ".1f"},
    "hard_hit_rate": {"table_header": "$\\bf{HH\\%}$", "format": ".1%"},
    "xba": {"table_header": "$\\bf{xBA}$", "format": ".3f"},
}
table_columns = [
    "pitch_description","pitch","pitch_usage","release_speed","release_speed_max","pfx_z","pfx_x",
    "release_spin_rate","release_pos_x", "release_pos_z","release_extension","in_zone_rate","chase_rate","whiff_rate",
    "hard_hit_rate","xba","xwoba","delta_run_exp_per_100",
]
cmap_sum  = mpl.colors.LinearSegmentedColormap.from_list("", ["#648FFF","#FFFFFF","#FFB000"])
cmap_sum_r = mpl.colors.LinearSegmentedColormap.from_list("", ["#FFB000","#FFFFFF","#648FFF"])
color_stats = ["release_speed","release_extension","delta_run_exp_per_100","whiff_rate","in_zone_rate",
               "chase_rate","hard_hit_rate","xba","xwoba","xwobacon"]

def df_grouping(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    agg_map = {
        "pitch_type": ("pitch_type","count"),
        "release_speed": ("release_speed","mean"),
        "release_speed_max": ("release_speed","max"),
        "pfx_z": ("pfx_z","mean"),
        "pfx_x": ("pfx_x","mean"),
        "release_spin_rate": ("release_spin_rate","mean"),
        "release_pos_x": ('release_pos_x', 'mean'),
        "release_pos_z": ('release_pos_z', 'mean'),
        "release_extension": ("release_extension","mean"),
        "delta_run_exp": ("delta_run_exp","sum"),
        "swing": ("swing","sum"),
        "whiff": ("whiff","sum"),
        "in_zone": ("in_zone","sum"),
        "out_zone": ("out_zone","sum"),
        "chase": ("chase","sum"),
        "xwoba": ("estimated_woba_using_speedangle","mean"),
    }
    g = (df.groupby("pitch_type").agg(**agg_map)
           .rename(columns={"pitch_type":"pitch"})
           .reset_index())

    # xBA with strikeouts = 0
    xba_df = df[["pitch_type","estimated_ba_using_speedangle","events"]].copy()
    mask_k = xba_df["events"].str.contains("strikeout", case=False, na=False)
    xba_df.loc[mask_k, "estimated_ba_using_speedangle"] = 0.0
    xba_map = xba_df.groupby("pitch_type")["estimated_ba_using_speedangle"].mean()
    g["xba"] = g["pitch_type"].map(xba_map)

    # xwOBA on contact
    x_df = df.loc[df["type"].eq("X"), ["pitch_type","estimated_woba_using_speedangle"]]
    xwoba_map = x_df.groupby("pitch_type")["estimated_woba_using_speedangle"].mean()
    g["xwobacon"] = g["pitch_type"].map(xwoba_map)

    # hard-hit on contact
    hh = df.loc[df["type"].eq("X"), ["pitch_type","launch_speed"]].copy()
    hh["is_hard_hit"] = pd.to_numeric(hh["launch_speed"], errors="coerce") >= 95
    hard_hit_map = hh.groupby("pitch_type")["is_hard_hit"].mean()
    g["hard_hit_rate"] = g["pitch_type"].map(hard_hit_map)

    g["pitch_description"] = g["pitch_type"].map(dict_pitch)
    g["color"] = g["pitch_type"].map(dict_color)

    tot_pitches = g["pitch"].sum()
    g["pitch_usage"] = sdiv(g["pitch"], tot_pitches)
    g["whiff_rate"] = sdiv(g["whiff"], g["swing"])
    g["in_zone_rate"] = sdiv(g["in_zone"], g["pitch"])
    g["chase_rate"] = sdiv(g["chase"], g["out_zone"])
    g["delta_run_exp_per_100"] = -sdiv(g["delta_run_exp"], g["pitch"]) * 100

    g = g.sort_values("pitch_usage", ascending=False)
    color_list = g["color"].tolist()

    all_row = pd.DataFrame([{
        "pitch_type":"All",
        "pitch_description":"All",
        "pitch": df["pitch_type"].count(),
        "pitch_usage": 1.0,
        "release_speed": np.nan,
        "release_speed_max": np.nan,
        "pfx_z": np.nan,
        "pfx_x": np.nan,
        "release_spin_rate": np.nan,
        "release_pos_x": np.nan,
        "release_pos_z": np.nan,
        "release_extension": df["release_extension"].mean(),
        "delta_run_exp_per_100": -df["delta_run_exp"].sum() / max(df["pitch_type"].count(), 1) * 100,
        "whiff_rate": df["whiff"].sum()/max(df["swing"].sum(), 1),
        "in_zone_rate": df["in_zone"].sum()/max(df["pitch_type"].count(), 1),
        "chase_rate": df["chase"].sum()/max(df["out_zone"].sum(), 1),
        "xwoba": df["estimated_woba_using_speedangle"].mean(),
        "hard_hit_rate": float(hh["is_hard_hit"].mean()) if not hh.empty else np.nan,
        "xba": xba_df["estimated_ba_using_speedangle"].mean(),
        "color": "#FFFFFF",
    }])
    df_plot = pd.concat([g, all_row], ignore_index=True)
    return df_plot, color_list

def _make_normalizers(df_statcast_group: pd.DataFrame) -> dict:
    means = df_statcast_group.groupby("pitch_type").mean(numeric_only=True)
    norms = {}
    for pt, row in means.iterrows():
        for col in color_stats:
            if col == "release_speed":
                mu = row.get(col, np.nan); vmin, vmax = mu * 0.95, mu * 1.05
            elif col == "delta_run_exp_per_100":
                vmin, vmax = -1.5, 1.5
            elif col == "xwobacon":
                mu = row.get(col, np.nan); vmin, vmax = mu * 0.7, mu * 1.3
            else:
                mu = row.get(col, np.nan); vmin, vmax = mu * 0.7, mu * 1.3
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0
            norms[(pt, col)] = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return norms

def _to_hex(value, norm, cmap):
    if not np.isfinite(value):
        return "#ffffff"
    return mcolors.to_hex(cmap(norm(value)))

def plot_pitch_format(df_plot: pd.DataFrame) -> pd.DataFrame:
    df_fmt = df_plot[table_columns].copy()
    for col, props in pitch_stats_dict.items():
        if col in df_fmt.columns:
            df_fmt[col] = fmt_series(pd.to_numeric(df_fmt[col], errors="coerce"), props["format"])
    df_fmt["pitch_description"] = df_fmt["pitch_description"].fillna("—")
    return df_fmt

def get_cell_colors(df_group: pd.DataFrame, df_statcast_group: pd.DataFrame) -> list[list[str]]:
    norms = _make_normalizers(df_statcast_group)
    cmap_for = {c: (cmap_sum_r if c in ("xwoba","xwobacon","hard_hit_rate","xba") else cmap_sum) for c in color_stats}
    rows = []
    for _, r in df_group.iterrows():
        pt = r["pitch_type"]
        row_colors = []
        for col in table_columns:
            if col in color_stats and isinstance(r[col], (int,float,np.floating)):
                norm = norms.get((pt, col)); cmap = cmap_for.get(col, cmap_sum)
                row_colors.append(_to_hex(float(r[col]), norm, cmap) if norm else "#ffffff")
            else:
                row_colors.append("#ffffff")
        rows.append(row_colors)
    return rows

def pitch_table(df: pd.DataFrame, df_statcast_group: Optional[pd.DataFrame], ax: plt.Axes, fontsize: int = 15):
    df_group, col_first = df_grouping(df)
    if df_statcast_group is None or df_statcast_group.empty:
        cell_colors = [["#ffffff"] * len(table_columns) for _ in range(len(df_group))]
    else:
        cell_colors = get_cell_colors(df_group, df_statcast_group)
    df_display = plot_pitch_format(df_group)

    table = ax.table(
        cellText=df_display.values,
        colLabels=table_columns,
        cellLoc="center",
        bbox=[0, -0.1, 1, 1],
        colWidths=[2.5] + [1] * (len(table_columns) - 1),
        cellColours=cell_colors
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 0.5)

    nice_headers = (["$\\bf{Pitch\\ Name}$"] +
                    [pitch_stats_dict.get(c, {}).get("table_header", "---") for c in table_columns[1:]])
    for i, name in enumerate(nice_headers):
        table.get_celld()[(0, i)].get_text().set_text(name)

    for i in range(1, len(df_display) + 1):
        cell = table.get_celld()[(i, 0)]
        txt = cell.get_text().get_text()
        cell.get_text().set_fontweight("bold")
        cell.set_text_props(color="#000000" if txt in ["Splitter","Slider","Changeup","Sinker","Screwball","Forkball","Sweeper"] else "#FFFFFF")
        if i - 1 < len(col_first):
            cell.set_facecolor(col_first[i - 1])

    ax.axis("off")

# ---------- Entry point ----------
def pitching_dashboard(pitcher_id: int, start_dt: str, end_dt: str) -> Optional[plt.Figure]:
    """
    Build and return the pitcher dashboard Figure for the given date range.
    Returns None if no MLB Statcast data is available.
    """
    # 1) Pull Statcast
    df_pyb = pyb.statcast_pitcher(start_dt, end_dt, int(pitcher_id))
    if df_pyb is None or df_pyb.empty:
        return None

    # Exclude Spring/Exhibitions if column exists
    if "game_type" in df_pyb.columns:
        df_pyb = df_pyb.loc[~df_pyb["game_type"].isin(["S","E"])].copy()
        if df_pyb.empty:
            return None

    # 2) League files (optional)
    # If missing, tables still render (no colored cells / no league ellipses).
    try:
        df_statcast_group = pd.read_csv(resolve_data("statcast_2025_grouped.csv"))
    except Exception:
        df_statcast_group = None

    try:
        df_pitch_movement = pd.read_csv(resolve_data("statcast_2025_pitch_movement.csv"))
    except Exception:
        df_pitch_movement = None

    # 3) FanGraphs leaders (for the season table / percentiles)
    season = pd.to_datetime(end_dt).year
    df_fg = fangraphs_pitching_leaderboards(season, start_dt, end_dt)

    # 4) Process Statcast
    df = df_processing(df_pyb)

    # 5) Figure & layout
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(
        6, 8,
        height_ratios=[2, 19, 9, 36, 36, 7],
        width_ratios=[1, 26, 26, 22, 22, 18, 18, 1]
    )

    ax_headshot = fig.add_subplot(gs[1,1:3])
    ax_bio      = fig.add_subplot(gs[1,3:5])
    ax_logo     = fig.add_subplot(gs[1,5:7])

    ax_season_table = fig.add_subplot(gs[2,1:7])

    ax_plot_1 = fig.add_subplot(gs[3,1:3])
    ax_plot_2 = fig.add_subplot(gs[3,3:5])
    ax_plot_3 = fig.add_subplot(gs[3,5:7])

    ax_table  = fig.add_subplot(gs[4,1:7])

    ax_footer = fig.add_subplot(gs[-1,1:7])
    ax_header = fig.add_subplot(gs[0,1:7])
    ax_left   = fig.add_subplot(gs[:,0])
    ax_right  = fig.add_subplot(gs[:,-1])
    for a in (ax_footer, ax_header, ax_left, ax_right):
        a.axis("off")

    # 6) Populate sections
    stats = ["G","GS","TBF","IP","WHIP","ERA","FIP","K%","BB%","GB%","sp_stuff"]
    fangraphs_pitcher_stats(
        pitcher_id, ax=ax_season_table, stats=stats,
        season=season, start_dt=start_dt, end_dt=end_dt, fontsize=20, df_fangraphs=df_fg
    )
    pitch_table(df, df_statcast_group, ax_table, fontsize=13)

    player_headshot(pitcher_id, ax=ax_headshot)
    player_bio(pitcher_id, start_dt=start_dt, end_dt=end_dt, ax=ax_bio)
    plot_logo(pitcher_id, ax=ax_logo)

    roll_pa = rolling_xwoba(df, window_pas=100)
    plot_rolling_xwoba(roll_pa, window_pas=100, df_statcast_group=df_statcast_group, ax=ax_plot_1)
    if df_pitch_movement is not None:
        break_plot(df=df, ax=ax_plot_2, df_statcast_group=df_pitch_movement)
    else:
        ax_plot_2.text(0.5, 0.5, "League movement file not found", ha="center", va="center"); ax_plot_2.axis("off")

    plot_percentile_rankings(
        pitcher_id=pitcher_id, season=season, start_dt=start_dt, end_dt=end_dt,
        ax=ax_plot_3, df_fangraphs=df_fg
    )

    # 7) Footer
    ax_footer.text(0,   1, "By: Jake Vickroy", ha="left",  va="top", fontsize=24)
    ax_footer.text(0.5, 1, "Color Coding Compares to League Average By Pitch", ha="center", va="top", fontsize=16)
    ax_footer.text(1,   1, "Data: MLB, Fangraphs", ha="right", va="top", fontsize=24)

    fig.tight_layout()
    return fig
