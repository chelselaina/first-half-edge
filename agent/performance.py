"""Forward-only First Half Edge research audit ledger.

Runs *after* agent.auto_alert. No real bets, no wagers placed, no EV or
probability claims. Freeze each pre-tip prediction/first eligible paper-watch
quote so later results cannot rewrite a forecast. Exact-only grading matches
provider team names to their validated SportsDataverse equivalents.

Writes data/paper_predictions.csv and data/paper_watch_candidates.csv.
These files become PUBLIC when committed to a public GitHub repository.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from agent.feeds import get_history
from agent.slate import analyze_slate, fetch_upcoming_events

ET = ZoneInfo('America/New_York')
def _model_fingerprint() -> str:
    """Record the actual estimator source version, not a manually maintained label.

    The paper-ledger itself can change without making past projections look as
    though they came from a new statistical estimator.
    """
    content = b''
    for relative in ('agent/matchups.py', 'agent/slate.py'):
        source = Path(relative)
        if not source.exists():
            return 'unknown_current_season_model'
        content += relative.encode('utf-8') + b'\0' + source.read_bytes()
    return 'season_model_' + hashlib.sha256(content).hexdigest()[:12]


MODEL_VERSION = _model_fingerprint()
PRED_FIELDS = [
    'event_id', 'game_date', 'tip_utc', 'forecast_utc', 'model_version',
    'provider_home', 'provider_away', 'matched_home', 'matched_away',
    'home_games', 'away_games', 'match_confidence', 'projected_home_h1',
    'projected_away_h1', 'predicted_home_margin',
    'status', 'actual_home_h1', 'actual_away_h1', 'actual_home_margin',
    'absolute_error', 'baseline_absolute_error', 'graded_utc'
]
WATCH_FIELDS = [
    'event_id', 'game_date', 'checked_at_utc', 'quote_updated_utc',
    'provider_home', 'provider_away', 'selection', 'bookmaker',
    'market', 'signed_spread', 'american_odds', 'model_selected_margin',
    'point_gap', 'status', 'actual_selected_margin', 'paper_result',
    'paper_profit_units', 'graded_utc'
]


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns, dtype='object')
    table = pd.read_csv(path, dtype='str', keep_default_na=False)
    for col in columns:
        if col not in table:
            table[col] = ''
    return table[columns].fillna('')


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(df[columns].fillna('').to_dict('records'))
    temp.replace(path)


def _number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _key(x):
    return str(x or '').strip()


def snapshot_predictions(existing: pd.DataFrame, fixture: pd.DataFrame,
                         slate: pd.DataFrame, *, now: datetime) -> pd.DataFrame:
    """Keep the earliest pregame prediction. Never change it on later scans."""
    if existing.empty and not len(existing.columns):
        existing = pd.DataFrame(columns=PRED_FIELDS)
    if fixture.empty or slate.empty:
        return existing
    ids = set(existing['event_id'].astype(str).tolist())
    fixture_by_id = {str(f['event_id']): f for f in fixture.to_dict('records')}
    records=[]
    for r in slate.to_dict('records'):
        eid = str(r.get('Event ID', ''))
        if not eid or eid in ids or eid not in fixture_by_id:
            continue
        f = fixture_by_id[eid]
        tip = pd.to_datetime(f.get('commence_time'), utc=True, errors='coerce')
        if pd.isna(tip) or tip.to_pydatetime() <= now:
            continue
        home, away = str(f['home_team']), str(f['away_team'])
        pred = _number(r.get('Projected home margin'))
        h_games, a_games = _number(r.get('Home games')), _number(r.get('Away games'))
        match = _number(r.get('Match confidence'))
        mh, ma = _key(r.get('Matched home')), _key(r.get('Matched away'))
        if (pred is None or h_games is None or a_games is None or match is None
                or match < 0.93 or not mh or not ma or mh == ma):
            continue
        date_str = tip.tz_convert(ET).date().isoformat()
        records.append(dict(event_id=eid, game_date=date_str,
            tip_utc=tip.isoformat(), forecast_utc=now.isoformat(),
            model_version=MODEL_VERSION,
            provider_home=home, provider_away=away,
            matched_home=mh, matched_away=ma,
            home_games=int(h_games), away_games=int(a_games),
            match_confidence=round(match, 3),
            projected_home_h1=r.get('Home 1H projected', ''),
            projected_away_h1=r.get('Away 1H projected', ''),
            predicted_home_margin=pred, status='PENDING',
            actual_home_h1='', actual_away_h1='',
            actual_home_margin='', absolute_error='',
            baseline_absolute_error='', graded_utc=''))
        ids.add(eid)
    if not records:
        return existing
    return pd.concat([existing, pd.DataFrame(records)[PRED_FIELDS]], ignore_index=True)


def snapshot_watch(existing: pd.DataFrame, price_file: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    """One *first eligible* watch per event, never select using later outcomes."""
    if not price_file.exists():
        return existing
    try:
        offers = pd.read_csv(price_file, dtype='str', keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError):
        return existing
    needed = {'event_id','game_date','selection','bookmaker','spread','american_odds',
              'model_margin','point_gap','status','checked_at_utc','quote_updated_utc'}
    if offers.empty or not needed.issubset(offers.columns):
        return existing
    known = set(existing['event_id'].astype(str).tolist())
    pred_by_id = {str(r['event_id']): r for r in predictions.to_dict('records')}
    offers = offers[offers['status'].str.startswith('REVIEW', na=False)].copy()
    if offers.empty:
        return existing
    offers['gap_num'] = pd.to_numeric(offers['point_gap'], errors='coerce')
    offers = offers.sort_values(['checked_at_utc','gap_num'], ascending=[True,False], kind='stable')
    additions=[]
    for r in offers.to_dict('records'):
        eid = str(r['event_id'])
        if eid in known or eid not in pred_by_id:
            continue
        p = pred_by_id[eid]
        checked = pd.to_datetime(r.get('checked_at_utc'), utc=True, errors='coerce')
        tip = pd.to_datetime(p.get('tip_utc'), utc=True, errors='coerce')
        if pd.isna(checked) or pd.isna(tip) or checked >= tip:
            continue
        selected = _key(r.get('selection'))
        if selected not in (_key(p['provider_home']), _key(p['provider_away'])):
            continue
        spread, odds, gap = (_number(r.get(x)) for x in ('spread','american_odds','point_gap'))
        if spread is None or odds is None or gap is None or odds == 0:
            continue
        additions.append(dict(event_id=eid,game_date=str(p['game_date']),
            checked_at_utc=checked.isoformat(),quote_updated_utc=r.get('quote_updated_utc',''),
            provider_home=p['provider_home'],provider_away=p['provider_away'],
            selection=selected,bookmaker=r['bookmaker'],market='spreads_h1',
            signed_spread=spread,american_odds=int(odds),
            model_selected_margin=r['model_margin'],point_gap=gap,
            status='PAPER WATCH — NO ACTUAL WAGER',actual_selected_margin='',
            paper_result='',paper_profit_units='',graded_utc=''))
        known.add(eid)
    if not additions:
        return existing
    return pd.concat([existing, pd.DataFrame(additions)[WATCH_FIELDS]], ignore_index=True)


def grade_predictions(predictions: pd.DataFrame, history: pd.DataFrame,
                      *, now: datetime) -> pd.DataFrame:
    """Only grade exact unique date+team pairs. Reverses orientation when necessary."""
    if predictions.empty or history.empty:
        return predictions
    games=history.copy()
    games['game_date'] = games['game_date'].astype(str).str[:10]
    games['home_team'] = games['home_team'].astype(str).str.strip()
    games['away_team'] = games['away_team'].astype(str).str.strip()
    games['home_h1'] = pd.to_numeric(games['home_h1'], errors='coerce')
    games['away_h1'] = pd.to_numeric(games['away_h1'], errors='coerce')
    games=games.dropna(subset=['home_h1','away_h1'])
    games['pair_key']=games.apply(lambda r:(r['game_date'],frozenset((r['home_team'],r['away_team']))),axis=1)
    lookup={}
    for k,grp in games.groupby('pair_key'):
        if len(grp)==1:
            lookup[k]=grp.iloc[0]
    out=predictions.copy()
    for ix,p in out.iterrows():
        if str(p['status']) == 'GRADED':
            continue
        day, mh, ma = _key(p['game_date']), _key(p['matched_home']), _key(p['matched_away'])
        if not day or not mh or not ma or mh==ma:
            continue
        g=lookup.get((day,frozenset((mh,ma))))
        if g is None:
            continue
        if g['home_team'] == mh and g['away_team']==ma:
            actual_h,actual_a=float(g['home_h1']),float(g['away_h1'])
        elif g['away_team'] == mh and g['home_team']==ma:
            actual_h,actual_a=float(g['away_h1']),float(g['home_h1'])
        else:
            continue
        predicted = _number(p['predicted_home_margin'])
        if predicted is None:
            continue
        margin=actual_h-actual_a
        out.at[ix,'status']='GRADED'
        out.at[ix,'actual_home_h1']=actual_h
        out.at[ix,'actual_away_h1']=actual_a
        out.at[ix,'actual_home_margin']=margin
        out.at[ix,'absolute_error']=round(abs(predicted-margin),3)
        # This baseline is a *constant +2 home* assumption; neutral venues not known.
        out.at[ix,'baseline_absolute_error']=round(abs(2.0-margin),3)
        out.at[ix,'graded_utc']=now.isoformat()
    return out


def grade_watches(watches: pd.DataFrame, predictions: pd.DataFrame,
                  *, now: datetime) -> pd.DataFrame:
    if watches.empty or predictions.empty:
        return watches
    settled={str(r['event_id']):r for r in predictions.to_dict('records')
             if str(r.get('status',''))=='GRADED'}
    out=watches.copy()
    for ix,q in out.iterrows():
        if str(q['paper_result']) in ('WIN','LOSS','PUSH'):
            continue
        p=settled.get(str(q['event_id']))
        if p is None:
            continue
        margin=_number(p.get('actual_home_margin'))
        signed_line=_number(q.get('signed_spread'))
        price=_number(q.get('american_odds'))
        if margin is None or signed_line is None or price is None:
            continue
        if str(q['selection'])==str(p['provider_away']):
            margin=-margin
        elif str(q['selection'])!=str(p['provider_home']):
            continue
        settle=margin+signed_line
        if abs(settle)<1e-8:
            result,profit='PUSH',0.0
        elif settle>0:
            result='WIN'
            profit=price/100.0 if price>0 else 100.0/abs(price)
        else:
            result,profit='LOSS',-1.0
        out.at[ix,'actual_selected_margin']=margin
        out.at[ix,'paper_result']=result
        out.at[ix,'paper_profit_units']=round(profit,5)
        out.at[ix,'graded_utc']=now.isoformat()
    return out


def report(predictions: pd.DataFrame, watches: pd.DataFrame) -> str:
    """Report forecast error, signed bias, and independent paper line results.

    Bias is descriptive only. The app MUST NOT silently change a live forecast
    using an unvalidated correction; that requires a forward holdout comparison.
    """
    graded = predictions[predictions['status'] == 'GRADED'].copy()
    pred_m = pd.to_numeric(graded['predicted_home_margin'], errors='coerce')
    actual_m = pd.to_numeric(graded['actual_home_margin'], errors='coerce')
    home_forecast = pd.to_numeric(graded['projected_home_h1'], errors='coerce')
    away_forecast = pd.to_numeric(graded['projected_away_h1'], errors='coerce')
    home_actual = pd.to_numeric(graded['actual_home_h1'], errors='coerce')
    away_actual = pd.to_numeric(graded['actual_away_h1'], errors='coerce')
    # Positive bias = predicted higher than the actual result.
    margin_bias = (pred_m - actual_m).mean()
    home_score_bias = (home_forecast - home_actual).mean()
    away_score_bias = (away_forecast - away_actual).mean()
    mae = pd.to_numeric(graded['absolute_error'], errors='coerce').mean()
    baseline = pd.to_numeric(graded['baseline_absolute_error'], errors='coerce').mean()

    settled = watches[watches['paper_result'].isin(['WIN', 'LOSS', 'PUSH'])]
    profit = pd.to_numeric(settled['paper_profit_units'], errors='coerce').sum()
    roi = profit / len(settled) if len(settled) else None

    def fmt(v):
        return f'{v:+.2f}' if pd.notna(v) else 'not available'

    def mag(v):
        return f'{v:.2f}' if pd.notna(v) else 'not available'

    n = len(graded)
    bias_note = ('**Not enough forward-graded games for a candidate calibration rule.**'
                 if n < 50 else
                 '**Descriptive bias only; no automatic correction is enabled.** '
                 'Before applying a change, compare corrected and original '
                 'predictions on subsequent games not used to estimate bias.')
    msg = (f'### First Half Edge paper research ledger\n'
           f'- Frozen pregame predictions: **{len(predictions)}**; graded: **{n}**\n'
           f'- Halftime-margin MAE: **{mag(mae)} points**; fixed +2-home baseline: **{mag(baseline)} points**\n'
           f'- Signed predicted-minus-actual halftime margin: **{fmt(margin_bias)} points**\n'
           f'- Home first-half scoring bias: **{fmt(home_score_bias)} points**; '
           f'away scoring bias: **{fmt(away_score_bias)} points**\n'
           f'- Calibration status: {bias_note}\n'
           f'- Price-gap paper watches: **{len(watches)}**; settled: **{len(settled)}**\n'
           f'- Hypothetical 1-unit return: **{mag(profit)} units**; paper ROI: '
           f'**{f"{roi:.1%}" if roi is not None else "not available"}**\n'
           '- No actual wagers placed. This observational record does not establish '
           'profitable betting performance. Odds coverage is incomplete.\n')
    print(msg)
    path = os.environ.get('GITHUB_STEP_SUMMARY', '')
    if path:
        with open(path, 'a', encoding='utf-8') as fd:
            fd.write('\n' + msg)
    return msg


def tracker_self_test() -> str:
    """Tiny deterministic integrity test without network, keys, or real wagers.

    Fails the scheduled run if core ledger functions can't freeze, preserve,
    and grade a forecast with the correct winning/losing spread orientation.
    """
    from tempfile import TemporaryDirectory

    before = datetime.fromisoformat('2026-11-13T15:00:00+00:00')
    later = datetime.fromisoformat('2026-11-13T22:00:00+00:00')
    event = 'SELFTEST_EVENT_NEVER_REAL'
    fixture = pd.DataFrame([dict(event_id=event,
        commence_time='2026-11-13T20:00:00+00:00',
        home_team='Example Home', away_team='Example Away')])
    slate = pd.DataFrame([{'Event ID':event,
        'Projected home margin':4.5, 'Home games':3,'Away games':4,
        'Match confidence':1.0,'Matched home':'Example Home',
        'Matched away':'Example Away','Home 1H projected':40,
        'Away 1H projected':35.5}])
    p = pd.DataFrame(columns=PRED_FIELDS)
    p = snapshot_predictions(p, fixture, slate, now=before)
    if len(p) != 1 or str(p.iloc[0]['status']) != 'PENDING':
        raise AssertionError('Tracker cannot freeze pregame prediction')
    p = snapshot_predictions(p, fixture, slate, now=before)
    if len(p) != 1:
        raise AssertionError('Tracker duplicated frozen prediction')
    hist = pd.DataFrame([{'game_date':'2026-11-13',
        'home_team':'Example Home','away_team':'Example Away',
        'home_h1':42,'away_h1':36}])
    p = grade_predictions(p, hist, now=later)
    if (str(p.iloc[0]['status']) != 'GRADED'
            or abs(float(p.iloc[0]['absolute_error']) - 1.5) > 1e-6):
        raise AssertionError('Tracker grade or margin error is wrong')
    # A wager on Example Home -3.5 wins when home leads by 6.
    with TemporaryDirectory() as tmp:
        price = Path(tmp) / 'quotes.csv'
        pd.DataFrame([{'event_id':event,'game_date':'2026-11-13',
            'selection':'Example Home','bookmaker':'Example Book',
            'spread':-3.5,'american_odds':-110,'model_margin':4.5,
            'point_gap':1.0,'status':'REVIEW POINT GAP — PAPER ONLY',
            'checked_at_utc':'2026-11-13T15:01:00+00:00',
            'quote_updated_utc':'2026-11-13T15:00:00+00:00'}]).to_csv(price,index=False)
        w = snapshot_watch(pd.DataFrame(columns=WATCH_FIELDS),price,p)
        w = grade_watches(w,p,now=later)
        if (len(w) != 1 or str(w.iloc[0]['paper_result']) != 'WIN'
                or abs(float(w.iloc[0]['paper_profit_units']) - 100/110) > 1e-4):
            raise AssertionError('Tracker quote snapshot or 1H spread grading is wrong')
        ledger = Path(tmp)/'predictions.csv'
        _write_csv(ledger,p,PRED_FIELDS)
        if len(_read_csv(ledger,PRED_FIELDS)) != 1:
            raise AssertionError('Tracker failed to persist records')
    return 'Tracker integrity test PASSED (fictional data only).'


def run(*, quotes_only: bool = False):
    now=datetime.now(timezone.utc)
    local=now.astimezone(ET)
    season=local.year+1 if local.month>=7 else local.year
    root=Path(os.getenv('PAPER_DATA_DIR','data'))
    pred_path=root/'paper_predictions.csv'
    watch_path=root/'paper_watch_candidates.csv'
    pred=_read_csv(pred_path,PRED_FIELDS)
    watch=_read_csv(watch_path,WATCH_FIELDS)
    initial_pred=len(pred)
    initial_graded=(pred['status']=='GRADED').sum() if not pred.empty else 0
    key=os.getenv('ODDS_API_KEY','').strip()
    market_output=Path(os.getenv('SCAN_OUTPUT','run_artifacts/quotes.csv'))

    if quotes_only:
        # Attach sportsbook quotes without re-fetching the schedule or scores.
        # This runs after the odds checker and must NOT change past predictions.
        watch=snapshot_watch(watch, market_output, pred)
        _write_csv(watch_path,watch,WATCH_FIELDS)
        print('Quote attachment complete. Existing pregame forecasts not modified.')
        report(pred,watch)
        return

    if not key:
        raise SystemExit('Missing ODDS_API_KEY in GitHub Actions repository secrets')
    history=pd.DataFrame()
    score_date='unavailable'
    try:
        history=get_history([season],'/tmp/fhe_paper_games.csv')
        if not history.empty:
            score_date=str(history.game_date.max())
    except Exception as e:
        # Never print an exception message that might contain a URL/API key.
        print(f'History feed unavailable ({type(e).__name__}). '
              'Existing predictions remain pending; no invented outcomes.')
    # Grade forecasts from previous days independently of the odds scanner.
    pred=grade_predictions(pred,history,now=now)
    watch=grade_watches(watch,pred,now=now)
    fixtures_count=0
    projections_count=0
    eligible_count=0
    try:
        fixtures=fetch_upcoming_events(local.date().isoformat(),key,now_utc=now)
        fixtures_count=len(fixtures)
        if fixtures.empty:
            print('No pre-tip games remain today in the schedule feed.')
        elif history.empty:
            print(f'{fixtures_count} upcoming fixtures; history unavailable, '
                  'so no safe matchup predictions generated.')
        else:
            slate=analyze_slate(history,fixtures,local.date().isoformat(),season=season)
            projections_count=int(pd.to_numeric(slate.get('Projected home margin'),errors='coerce').notna().sum())
            # These match/quality checks mirror the actual snapshot requirements.
            eligible_count=sum(
                _number(r.get('Projected home margin')) is not None
                and (_number(r.get('Match confidence')) or 0) >= .93
                and _number(r.get('Home games')) is not None
                and _number(r.get('Away games')) is not None
                and _key(r.get('Matched home')) != _key(r.get('Matched away'))
                and bool(_key(r.get('Matched home')))
                and bool(_key(r.get('Matched away')))
                for r in slate.to_dict('records')
            )
            pred=snapshot_predictions(pred,fixtures,slate,now=now)
    except Exception as e:
        print(f"Today's forecast snapshot unavailable ({type(e).__name__}); "
              'prior audit records preserved. See full run logs for source status.')
    # This phase is independent of the odds lookup. Odds are attached later.
    _write_csv(pred_path,pred,PRED_FIELDS)
    _write_csv(watch_path,watch,WATCH_FIELDS)
    new_count=len(pred)-initial_pred
    newly_graded=int((pred['status']=='GRADED').sum())-int(initial_graded)
    status=(f'Scan audit: {fixtures_count} pre-tip fixtures, {projections_count} model '
            f'estimates, {eligible_count} eligible estimates, {new_count} NEW frozen '
            f'pregame predictions, {newly_graded} NEW results graded. '
            f'Score data latest date: {score_date}.')
    print(status)
    location = os.getenv('GITHUB_STEP_SUMMARY','')
    if location:
        with open(location,'a',encoding='utf-8') as fd:
            fd.write('\n' + status + '\n')
    report(pred,watch)


if __name__=='__main__':
    import sys
    if '--self-test' in sys.argv:
        print(tracker_self_test())
    else:
        run(quotes_only='--quotes-only' in sys.argv)
