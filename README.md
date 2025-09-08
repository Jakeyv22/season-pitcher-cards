# MLB Season Pitcher Cards (Streamlit)

Create season/date-range pitcher cards built on **Statcast** and **Fangraphs**. Pick a pitcher, choose a window length, and get a single dashboard: season stats, rolling xwOBA, pitch shapes, and percentile ranks — all exportable as a PNG.

## ▶️ Live App
<!-- Replace with your deployed URL -->
[Open the App!](https://your-season-pitcher-cards-app.streamlit.app/)

![Screenshot](assets/screenshot.png)

## Features
- **Date-range cards**: not just game-day — any window within the season
- **Fangraphs season table**: ERA/FIP/WHIP/K%/BB%/GB% plus Stuff+
- **Percentile rankings**: CSW%, Zone%, Chase%, EV, SIERA
- **Rolling xwOBA**: 100 PA rolling lines with dotted y-tick guides & month markers
- **Pitch movement**: scatter + league average “ellipses” by pitch
- **PNG export** button in-app

## Data sources
- **Statcast** via [`pybaseball`](https://github.com/jldbc/pybaseball)
- **Fangraphs**
  
## Repo layout
