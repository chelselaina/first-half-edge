"""Pregame feature engineering, strictly chronological training, and spread valuation."""
from __future__ import annotations
from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
from math import floor
import re
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LEAGUE_HALF_PTS = 35.0  # neutral cold-start PRIOR, not an observed season statistic
MIN_TEAM_GAMES = 5


def validate_games(games: pd.DataFrame) -> pd.DataFrame:
    required = {'game_id','season','game_date','home_id','away_id','home_team','away_team',
                'home_h1','away_h1','neutral'}
    missing = required - set(games.columns)
    if missing: raise ValueError(f'Historical games CSV missing: {sorted(missing)}')
    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date'],errors='coerce').dt.strftime('%Y-%m-%d')
    if df['game_date'].isna().any(): raise ValueError('Unparseable game dates in history')
    for c in ('home_id','away_id','game_id'): df[c] = df[c].astype(str)
    for c in ('home_h1','away_h1'): df[c] = pd.to_numeric(df[c],errors='raise')
    df['season'] = pd.to_numeric(df['season'],errors='raise').astype(int)
    df['neutral'] = df['neutral'].map(lambda x:str(x).strip().lower() in ('true','1','yes','t'))
    if df['game_id'].duplicated().any(): raise ValueError('Duplicate game_id values')
    if (df[['home_h1','away_h1']] < 0).any().any(): raise ValueError('Negative score')
    return df.sort_values(['season','game_date','game_id']).reset_index(drop=True)


def season_from_day(day: str) -> int:
    d = date.fromisoformat(str(day)[:10])
    return d.year+1 if d.month >= 7 else d.year


def _smoothed(values, prior=LEAGUE_HALF_PTS, k=3):
    if not values: return float(prior)
    return float((sum(values) + k*prior)/(len(values)+k))


def _snapshot(hist: list[dict], current_day: str, venue: str) -> dict:
    last10 = hist[-10:]
    last5 = hist[-5:]
    loc = [x for x in hist if x['venue'] == venue][-5:]
    n = len(hist)
    rest = min((date.fromisoformat(current_day)-date.fromisoformat(hist[-1]['date'])).days, 14) if hist else 7
    def avg(subset, key):
        return _smoothed([x[key] for x in subset], 35.0 if key not in ('margin',) else 0, 3)
    return {
        'n':min(n,35),'pf5':avg(last5,'pf'),'pa5':avg(last5,'pa'),
        'pf10':avg(last10,'pf'),'pa10':avg(last10,'pa'),
        'adjusted_pf5':avg(last5,'adj_pf'),'adjusted_pa5':avg(last5,'adj_pa'),
        'margin5':avg(last5,'margin'),
        'trend_margin':avg(last5,'margin')-avg(last10,'margin'),
        'venue_pf':avg(loc,'pf'), 'venue_pa':avg(loc,'pa'),
        'venue_count':min(len(loc),5),'rest':rest,
        'h1_total10':avg(last10,'pf')+avg(last10,'pa'),
    }


def _feature(h: dict, a: dict, neutral: bool) -> dict:
    x = {}
    for k in h:
        x[f'home_{k}']=h[k]
        x[f'away_{k}']=a[k]
    for key in ('pf5','pa5','adjusted_pf5','adjusted_pa5','margin5','trend_margin',
                'pf10','pa10','rest','h1_total10'):
        x[f'diff_{key}'] = h[key]-a[key]
    x['neutral']=float(neutral)
    x['projected_points']=0.5*(h['pf5']+a['pa5']+a['pf5']+h['pa5'])
    return x


def build_features(games: pd.DataFrame, fixtures: pd.DataFrame | None = None):
    """Generate features using ONLY games played on earlier dates in the same season.

    For safety, all games on a given day use snapshots made before that day.
    Future fixtures never update history. No pregame stats from target game.
    """
    games=validate_games(games)
    past=games.assign(_future=False)
    if fixtures is not None and len(fixtures):
        needed={'game_date','home_id','away_id','home_team','away_team','neutral'}
        if not needed.issubset(set(fixtures.columns)):
            raise ValueError(f'Fixtures missing {sorted(needed-set(fixtures.columns))}')
        upcoming=fixtures.copy()
        upcoming['_future']=True
        upcoming['game_id']=upcoming.get('game_id',pd.Series([f'fixture_{i}' for i in range(len(upcoming))],index=upcoming.index))
        upcoming['season']=upcoming['game_date'].map(season_from_day)
        for c in ('home_id','away_id','game_id'): upcoming[c]=upcoming[c].astype(str)
        allrows=pd.concat([past,upcoming],ignore_index=True,sort=False)
    else:
        allrows=past
    allrows=allrows.sort_values(['season','game_date','game_id'])
    result=[]
    for season, season_group in allrows.groupby('season',sort=True):
        hist=defaultdict(list)
        for day, batch in season_group.groupby('game_date',sort=True):
            all_teams=set(batch['home_id']).union(set(batch['away_id']))
            pre={tid:{'home':_snapshot(hist[tid],day,'home'),
                      'away':_snapshot(hist[tid],day,'away')} for tid in all_teams}
            for d in batch.to_dict('records'):
                h=pre[d['home_id']]['home']; a=pre[d['away_id']]['away']
                feat=_feature(h,a,bool(d['neutral']))
                feat.update({
                    'game_id':d['game_id'],'game_date':day,'season':season,
                    'home_id':d['home_id'],'away_id':d['away_id'],
                    'home_team':d['home_team'],'away_team':d['away_team'],
                    'is_future':bool(d['_future']),
                    'home_games_played':len(hist[d['home_id']]),
                    'away_games_played':len(hist[d['away_id']]),
                    'home_margin':np.nan if d['_future'] else float(d['home_h1'])-float(d['away_h1']),
                })
                result.append(feat)
            # Store adjusted points using opponent's PRE-DAY strength; no same-day peeking.
            for d in batch.loc[~batch['_future']].to_dict('records'):
                hi,ai=d['home_id'],d['away_id']
                hp,ap=float(d['home_h1']),float(d['away_h1'])
                hist[hi].append({'date':day,'pf':hp,'pa':ap,'margin':hp-ap,'venue':'neutral' if d['neutral'] else 'home',
                    'adj_pf':hp-pre[ai]['away']['pa10']+35.0,
                    'adj_pa':ap-pre[ai]['away']['pf10']+35.0})
                hist[ai].append({'date':day,'pf':ap,'pa':hp,'margin':ap-hp,'venue':'neutral' if d['neutral'] else 'away',
                    'adj_pf':ap-pre[hi]['home']['pa10']+35.0,
                    'adj_pa':hp-pre[hi]['home']['pf10']+35.0})
    return pd.DataFrame(result)


META={'game_id','game_date','season','home_id','away_id','home_team','away_team',
      'is_future','home_games_played','away_games_played','home_margin'}


def _new_regressor():
    return HistGradientBoostingRegressor(loss='squared_error',learning_rate=0.045,
                                         max_iter=180,max_leaf_nodes=12,max_depth=4,
                                         l2_regularization=20,random_state=42)


def train(games: pd.DataFrame) -> dict:
    """Chronological 65% fit / 15% calibration / 20% untouched test."""
    games=validate_games(games)
    ft=build_features(games)
    ft=ft.loc[~ft.is_future].sort_values(['game_date','game_id']).reset_index(drop=True)
    if len(ft)<250:
        raise ValueError(f'Need at least 250 historical games for a meaningful test; got {len(ft)}')
    feature_names=[c for c in ft.columns if c not in META]
    x=ft[feature_names].astype(float).to_numpy()
    y=ft['home_margin'].to_numpy(dtype=float)
    n=len(y); a=int(n*.65); b=int(n*.80)
    # Keep entire dates together; otherwise later games on same day can fall into training.
    def after_day(ix):
        while ix < n and ft.loc[ix,'game_date']==ft.loc[ix-1,'game_date']:
            ix += 1
        return ix
    a=after_day(a)
    b=after_day(max(b,a+1))
    if b>=n or a>=b: raise ValueError('Not enough unique dates for chronological holdouts')
    provisional=_new_regressor().fit(x[:a],y[:a])
    cal_preds=provisional.predict(x[a:b])
    calibration_error=y[a:b]-cal_preds
    sigma=float(np.clip(np.sqrt(np.mean(calibration_error**2)),7.0,20.0))
    test_model=_new_regressor().fit(x[:b],y[:b])
    test_preds=test_model.predict(x[b:])
    baseline_predictions=np.full_like(y[b:],1.5,dtype=float)
    metrics={
        'total_games':n,'training_games':a,'calibration_games':b-a,'test_games':n-b,
        'test_mae':round(float(mean_absolute_error(y[b:],test_preds)),2),
        'test_rmse':round(float(np.sqrt(mean_squared_error(y[b:],test_preds))),2),
        'baseline_mae':round(float(mean_absolute_error(y[b:],baseline_predictions)),2),
        'calibrated_sd':round(sigma,2),
        'test_80pct_interval_coverage':round(float(np.mean(abs(y[b:]-test_preds)<1.28155*sigma)),3),
        'test_from':str(ft.loc[b,'game_date']),
        'test_to':str(ft.loc[n-1,'game_date']),
        'last_history_date':str(ft['game_date'].max()),
        'model_better_than_baseline':bool(mean_absolute_error(y[b:],test_preds)<mean_absolute_error(y[b:],baseline_predictions)),
    }
    final=_new_regressor().fit(x,y)
    return {'model':final,'columns':feature_names,'sigma':sigma,
            'metrics':metrics,'schema_version':1}


def _canonical(s: str) -> str:
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',str(s).lower())).strip()


def _score_names(a: str, b: str) -> float:
    a,b=_canonical(a),_canonical(b)
    if a==b: return 1.0
    if min(len(a),len(b))>=7 and (a.startswith(b+' ') or b.startswith(a+' ')):
        return .965
    return SequenceMatcher(None,a,b).ratio()


def match_team(odds_name: str, team_id_to_name: dict[str,str], aliases: dict[str,str] | None = None) -> tuple[str | None,float]:
    if aliases and _canonical(odds_name) in aliases:
        odds_name=aliases[_canonical(odds_name)]
    options=sorted([(_score_names(odds_name,name),tid) for tid,name in team_id_to_name.items()],reverse=True)
    if not options: return None,0.0
    top=options[0]
    if top[0]<0.85 or (len(options)>1 and top[0]-options[1][0]<0.05):
        return None,float(top[0])
    return top[1],float(top[0])


def american_profit(price: int) -> float:
    price=int(price)
    if price==0 or -100<price<100:
        raise ValueError(f'American odds must be <= -100 or >= +100: got {price}')
    return (100.0/(-price)) if price<0 else (price/100.0)


def settlement_prob(mean: float, sigma: float, line: float) -> tuple[float,float,float]:
    """Probabilities for a bet on team's score margin + its signed spread.

    Integer scores imply whole-point lines can push; correct continuity correction.
    """
    threshold=-float(line)
    win_cut=floor(threshold)+.5
    pw=float(norm.sf(win_cut,loc=mean,scale=sigma))
    push=0.0
    if abs(threshold-round(threshold))<1e-8:
        push=float(norm.cdf(threshold+.5,loc=mean,scale=sigma)-
                   norm.cdf(threshold-.5,loc=mean,scale=sigma))
    return pw,push,max(0.0,1-pw-push)


def _fixtures_from_odds(odds: pd.DataFrame, games: pd.DataFrame,
                         aliases: dict[str,str] | None):
    mapping={str(k):str(v) for k,v in pd.concat([
        games[['home_id','home_team']].rename(columns={'home_id':'tid','home_team':'name'}),
        games[['away_id','away_team']].rename(columns={'away_id':'tid','away_team':'name'})
    ]).drop_duplicates('tid',keep='last')[['tid','name']].itertuples(index=False,name=None)}
    pairs=odds.drop_duplicates(['game_date','home_team','away_team']).copy()
    missing=[]; fixtures=[]
    for d in pairs.to_dict('records'):
        hi,hs=match_team(d['home_team'],mapping,aliases)
        ai,aas=match_team(d['away_team'],mapping,aliases)
        if not hi or not ai or hi==ai:
            missing.append(f"{d['game_date']}: {d['away_team']} @ {d['home_team']} (match scores {aas:.2f}/{hs:.2f})")
            continue
        fixtures.append({'game_id':f"fixture_{len(fixtures)}",'game_date':d['game_date'],
                         'home_id':hi,'away_id':ai,'home_team':d['home_team'],
                         'away_team':d['away_team'],'neutral':False})
    return pd.DataFrame(fixtures),missing


def scan(games: pd.DataFrame, odds: pd.DataFrame, trained: dict,
         min_ev: float=0.05, aliases: dict[str,str] | None=None) -> tuple[pd.DataFrame,list]:
    """Evaluate every book/side line; no automatic bet placement or staking."""
    games=validate_games(games)
    required={'game_date','home_team','away_team','selection','point','price','bookmaker'}
    if not required.issubset(odds.columns):
        raise ValueError(f'Odds file missing {sorted(required-set(odds.columns))}')
    odds=odds.copy(); odds['game_date']=pd.to_datetime(odds.game_date).dt.strftime('%Y-%m-%d')
    odds['point']=pd.to_numeric(odds['point']); odds['price']=pd.to_numeric(odds['price']).astype(int)
    odds['home_team']=odds.home_team.astype(str);odds['away_team']=odds.away_team.astype(str)
    odds['selection']=odds.selection.astype(str)
    fixtures,unmatched=_fixtures_from_odds(odds,games,aliases)
    if fixtures.empty: return pd.DataFrame(),unmatched
    future=build_features(games,fixtures)
    future=future.loc[future.is_future].copy()
    future['model_home_margin']=trained['model'].predict(future[trained['columns']].to_numpy(dtype=float))
    f_index={(r['game_date'],r['home_team'],r['away_team']):r for r in future.to_dict('records')}
    results=[]
    sigma=float(trained['sigma'])*1.15  # conservative volatility inflation
    for row in odds.to_dict('records'):
        f=f_index.get((row['game_date'],row['home_team'],row['away_team']))
        if f is None: continue
        home_selected=_canonical(row['selection'])==_canonical(row['home_team'])
        away_selected=_canonical(row['selection'])==_canonical(row['away_team'])
        if not home_selected and not away_selected: continue
        # Conservative 10% prediction shrink toward generic +1.5 home advantage.
        home_margin=float(f['model_home_margin'])*.90 + .15
        side_margin=home_margin if home_selected else -home_margin
        try: payout=american_profit(int(row['price']))
        except ValueError: continue
        pw,pp,pl=settlement_prob(side_margin,sigma,float(row['point']))
        ev=pw*payout-pl
        implied=1/(payout+1)
        fresh='CHECK LINE'
        if pd.notna(row.get('last_update')) and str(row.get('last_update')).strip():
            age=(pd.Timestamp.now(tz='UTC')-pd.to_datetime(row['last_update'],utc=True,errors='coerce')).total_seconds()/60
            if np.isfinite(age): fresh='FRESH' if -5<=age<=30 else 'STALE'
        experienced=int(f['home_games_played'])>=MIN_TEAM_GAMES and int(f['away_games_played'])>=MIN_TEAM_GAMES
        if not experienced: status='NO BET: SMALL SAMPLE'
        elif fresh=='STALE': status='NO BET: STALE ODDS'
        elif not trained['metrics']['model_better_than_baseline']: status='NO BET: UNVALIDATED MODEL'
        elif ev<min_ev: status='NO BET: EDGE TOO SMALL'
        else: status='PAPER CANDIDATE' if fresh=='FRESH' else 'RECHECK PRICE'
        results.append({
            'game_date':row['game_date'],'matchup':f"{row['away_team']} @ {row['home_team']}",
            'selection':row['selection'],'bookmaker':row['bookmaker'],
            'line':float(row['point']),'odds':int(row['price']),
            'projected_home_margin':round(float(f['model_home_margin']),2),
            'cover_prob':round(pw,4),'push_prob':round(pp,4),'break_even_prob':round(implied,4),
            'ev_roi':round(ev,4),'home_games':int(f['home_games_played']),
            'away_games':int(f['away_games_played']),
            'status':status,'price_status':fresh,
            'last_update':row.get('last_update',''),
            'model_sigma':round(sigma,2),
            'event_id':row.get('event_id',''),
        })
    output=pd.DataFrame(results)
    if not output.empty: output=output.sort_values('ev_roi',ascending=False).reset_index(drop=True)
    return output,unmatched


def grade_predictions(offers: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Grade odds offers for completed games by ID or date/team names, without guessing."""
    finished=validate_games(games)
    finished['key']=finished.game_date+'|'+finished.home_team.map(_canonical)+'|'+finished.away_team.map(_canonical)
    d=finished.set_index('key').to_dict('index')
    evaluated=[]
    for row in offers.to_dict('records'):
        key=str(row['game_date'])+'|'+_canonical(row['home_team'])+'|'+_canonical(row['away_team'])
        g=d.get(key)
        if g is None: continue
        home=_canonical(row['selection'])==_canonical(g['home_team'])
        away=_canonical(row['selection'])==_canonical(g['away_team'])
        if not home and not away: continue
        side=(g['home_h1']-g['away_h1']) if home else (g['away_h1']-g['home_h1'])
        covered=side+float(row['point'])
        profit=american_profit(int(row['price'])) if covered>0 else (-1 if covered<0 else 0)
        evaluated.append({**row,'first_half_margin_selected':side,'grade':'win' if covered>0 else ('push' if covered==0 else 'loss'),
                          'profit_per_unit':round(profit,4)})
    return pd.DataFrame(evaluated)
