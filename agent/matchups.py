"""2026-27 current-season matchup explorer.

Uses completed first-half scores ONLY. No possessions, rosters, or sportsbook odds.
Opponent adjustment uses ridge-regularized team offense/defense components,
fit on the selected season, not historical team scoring averages.
Exploratory matchup scores are unvalidated; never classify them as bets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack, eye
from scipy.sparse.linalg import lsqr

REQUIRED = {'season', 'game_date', 'home_id', 'away_id', 'home_team', 'away_team', 'home_h1', 'away_h1', 'neutral'}


def prepare_current_season(games: pd.DataFrame, season: int = 2027, as_of: str | None = None) -> pd.DataFrame:
    """Expand each game to two team records for the specified season.
    Date cutoff is inclusive; use only finished results from input data.
    """
    missing = REQUIRED - set(games.columns)
    if missing:
        raise ValueError(f'History missing columns: {sorted(missing)}')
    df = games.loc[pd.to_numeric(games['season'], errors='coerce') == season].copy()
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    if as_of:
        df = df.loc[df['game_date'] <= pd.to_datetime(as_of)]
    df['home_h1'] = pd.to_numeric(df['home_h1'], errors='coerce')
    df['away_h1'] = pd.to_numeric(df['away_h1'], errors='coerce')
    df = df.dropna(subset=['game_date', 'home_h1', 'away_h1', 'home_id', 'away_id'])
    df = df.loc[(df.home_h1 >= 0) & (df.away_h1 >= 0) & (df.home_id.astype(str) != df.away_id.astype(str))]
    if 'game_id' in df.columns:
        df = df.drop_duplicates('game_id')
    # Be forgiving of CSV roundtrips containing strings for booleans.
    neutral = df.neutral.astype(str).str.lower().isin(['true', 't', '1', 'yes'])
    home = pd.DataFrame({
        'date': df.game_date, 'team_id': df.home_id.astype(str), 'team':df.home_team.astype(str),
        'opponent_id':df.away_id.astype(str), 'opponent':df.away_team.astype(str),
        'pf':df.home_h1.astype(float), 'pa':df.away_h1.astype(float),
        'venue':np.where(neutral, 'Neutral', 'Home'),
        'home_indicator':np.where(neutral,0.,1.),
    })
    away = pd.DataFrame({
        'date':df.game_date, 'team_id':df.away_id.astype(str), 'team':df.away_team.astype(str),
        'opponent_id':df.home_id.astype(str), 'opponent':df.home_team.astype(str),
        'pf':df.away_h1.astype(float), 'pa':df.home_h1.astype(float),
        'venue':np.where(neutral, 'Neutral', 'Away'),
        'home_indicator':np.where(neutral,0.,-1.),
    })
    return pd.concat([home, away], ignore_index=True).sort_values(['date','team_id']).reset_index(drop=True)


def fit_current_ratings(team_games: pd.DataFrame, regularization: float = 9.0) -> tuple[pd.DataFrame,dict]:
    """Estimate current-season offense (+), defensive suppression (+) in halftime pts.

    Fit pf = league_mean + off(team) - defense(opponent) + small fixed venue offset.
    Ridge shrinkage toward zero; home offset is an explicit heuristic, not
    a separately estimated advantage or validated handicap.
    Diagnostic / research estimates; not validated cover probabilities.
    """
    if team_games.empty:
        raise ValueError('No completed games for this season')
    if len(team_games) < 20:
        raise ValueError('Too few current-season games to calculate stable ratings')
    ids = sorted(set(team_games.team_id) | set(team_games.opponent_id))
    idx = {tid:i for i,tid in enumerate(ids)}
    n = len(ids); m = len(team_games)
    row_ids=np.repeat(np.arange(m),2)
    col_ids=np.empty(2*m,dtype=int)
    vals=np.empty(2*m,dtype=float)
    col_ids[0::2]=team_games.team_id.map(idx).to_numpy(dtype=int)
    col_ids[1::2]=team_games.opponent_id.map(idx).to_numpy(dtype=int)+n
    vals[0::2]=1.; vals[1::2]=-1.
    X=coo_matrix((vals,(row_ids,col_ids)),shape=(m,2*n)).tocsr()
    league=float(team_games.pf.mean())
    home_offset=1.0  # half-point per team, i.e. ~2-point halftime margin; UNVALIDATED heuristic
    target=team_games.pf.to_numpy(dtype=float)-league-team_games.home_indicator.to_numpy(dtype=float)*home_offset
    # Penalize unstable current-season team strength estimates.
    reg=eye(2*n,format='csr')*np.sqrt(regularization)
    coefficients=lsqr(vstack([X,reg],format='csr'),np.r_[target,np.zeros(2*n)],atol=1e-9,btol=1e-9)[0]
    names=(pd.concat([team_games[['team_id','team']].rename(columns={'team_id':'id','team':'name'}),
                      team_games[['opponent_id','opponent']].rename(columns={'opponent_id':'id','opponent':'name'})])
             .drop_duplicates('id',keep='last').set_index('id')['name'].to_dict())
    counts=team_games.groupby('team_id').size().to_dict()
    ratings=pd.DataFrame([{'team_id':k,'team':names.get(k,k), 'games':int(counts.get(k,0)),
                           'offense_adj':float(coefficients[i]),
                           'defense_adj':float(coefficients[i+n]),
                           'net_adj':float(coefficients[i]+coefficients[i+n])}
                          for k,i in idx.items()])
    # Schedule quality from opponents' learned components (descriptive, non-causal)
    rating_map=ratings.set_index('team_id')
    team_games=team_games.copy()
    team_games['opponent_offense_adj']=team_games.opponent_id.map(rating_map.offense_adj).fillna(0)
    team_games['opponent_defense_adj']=team_games.opponent_id.map(rating_map.defense_adj).fillna(0)
    sched=team_games.groupby('team_id').agg(
        opponent_off_strength=('opponent_offense_adj','mean'),
        opponent_def_strength=('opponent_defense_adj','mean'),
    )
    ratings=ratings.join(sched,on='team_id')
    info={'league_avg':league,'home_effect_per_team':home_offset,
          'teams':n, 'completed_games':len(team_games)//2,
          'from_date':str(team_games.date.min().date()),
          'through_date':str(team_games.date.max().date()),
          'regularization':regularization}
    return ratings.sort_values('net_adj',ascending=False).reset_index(drop=True),info


def team_summary(team_id: str, team_games: pd.DataFrame, ratings: pd.DataFrame) -> dict:
    tg=team_games.loc[team_games.team_id==str(team_id)].sort_values('date')
    if tg.empty: raise ValueError('Selected team has no current-season games')
    row=ratings.loc[ratings.team_id==str(team_id)].iloc[0]
    recent=tg.tail(5)
    return {'team':row.team,'team_id':str(team_id),'games':len(tg),
            'scored':float(tg.pf.mean()),'allowed':float(tg.pa.mean()),
            'recent_scored':float(recent.pf.mean()),'recent_allowed':float(recent.pa.mean()),
            'offense_adj':float(row.offense_adj),'defense_adj':float(row.defense_adj),
            'opponent_off_strength':float(row.opponent_off_strength),
            'opponent_def_strength':float(row.opponent_def_strength),
            'schedule':tg.sort_values('date',ascending=False)[['date','opponent','venue','pf','pa']].copy()}


def illustrative_matchup(home: dict, away: dict, info: dict, neutral: bool=False) -> dict:
    mu=info['league_avg']; hc=0. if neutral else info['home_effect_per_team']
    h=mu+home['offense_adj']-away['defense_adj']+hc
    a=mu+away['offense_adj']-home['defense_adj']-hc
    return {'home_points':round(max(0.,h),1),'away_points':round(max(0.,a),1),
            'home_margin':round(h-a,1), 'min_games':min(home['games'],away['games'])}
