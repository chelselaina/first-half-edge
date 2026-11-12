"""Fully fictional deterministic demo data. NEVER use these as betting recommendations."""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd


def create_demo(folder: str):
    rng=np.random.default_rng(42026)
    teams=[f'DEMO College {i:02d}' for i in range(1,65)]
    ability_o=rng.normal(0,6,len(teams)); ability_d=rng.normal(0,5,len(teams))
    rows=[]
    for season, nrounds in [(2024,35),(2025,35),(2026,35),(2027,12)]:
        start=date(season-1,11,1)
        if season==2027: start=date(2026,11,1)
        ability_o+=rng.normal(0,1.0,len(teams));ability_d+=rng.normal(0,1.0,len(teams))
        for rd in range(nrounds):
            d=start+timedelta(days=rd*(3 if season!=2027 else 1))
            participants=rng.permutation(len(teams))
            for j in range(0,len(participants),2):
                h,a=map(int,participants[j:j+2]); neutral=bool(rng.random()<.08)
                pace=rng.normal(0,2.5)
                expected_home=35+ability_o[h] - ability_d[a]*.68+pace+(0 if neutral else 1)
                expected_away=35+ability_o[a] - ability_d[h]*.68+pace
                home_points=int(np.clip(round(rng.normal(expected_home,5.2)),15,65))
                away_points=int(np.clip(round(rng.normal(expected_away,5.2)),15,65))
                rows.append({'game_id':f'DEMO_{season}_{rd}_{j//2}',
                    'season':season,'game_date':d.isoformat(),
                    'home_id':str(h+1),'away_id':str(a+1),
                    'home_team':teams[h],'away_team':teams[a],
                    'home_h1':home_points,'away_h1':away_points,'neutral':neutral})
    games=pd.DataFrame(rows)
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    games.to_csv(out/'demo_games.csv',index=False)
    # Realistic but *invented* lines for eight fictional games on following day.
    rows=[]
    ids=rng.permutation(len(teams))[:16]
    for idx in range(8):
        h,a=map(int,ids[idx*2:idx*2+2]); line=float(rng.integers(-9,10)+.5)
        for sel,pt in ((teams[h],line),(teams[a],-line)):
            rows.append({'event_id':f'DEMO_EVENT_{idx}',
                'game_date':'2026-11-13','home_team':teams[h],
                'away_team':teams[a],'selection':sel,
                'point':pt,'price':-110,'bookmaker':'DEMO BOOK - NOT REAL',
                'last_update':''})
    odds=pd.DataFrame(rows)
    odds.to_csv(out/'demo_odds.csv',index=False)
    return games,odds
