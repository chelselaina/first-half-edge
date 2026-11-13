"""Read-only, current-season NCAA men's basketball daily fixture explorer.

Fixtures: The Odds API's free quota-exempt /events endpoint (no prices).
Scores: user-provided verified NCAA historical CSV (do not fetch automatically).
This module never places wagers or claims to compute a profitable market edge.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import requests

from .matchups import prepare_current_season, fit_current_ratings, team_summary, illustrative_matchup
from .model import match_team

SPORT = 'basketball_ncaab'
EVENTS_URL = f'https://api.the-odds-api.com/v4/sports/{SPORT}/events'
ET = ZoneInfo('America/New_York')
FIELDS = [
    'Game (away @ home)', 'Tip (ET)', 'Status', 'Home games', 'Away games',
    'Home 1H scored', 'Home 1H allowed', 'Away 1H scored', 'Away 1H allowed',
    'Home opp-off strength', 'Home opp-def strength',
    'Away opp-off strength', 'Away opp-def strength',
    'Home 1H projected', 'Away 1H projected', 'Projected home margin',
    'Matched home', 'Matched away', 'Match confidence', 'Source', 'Event ID'
]


def normalize_events(events: list[dict], target_date: str, *, now_utc: datetime | None = None) -> pd.DataFrame:
    """Filter live provider events by local US Eastern tip date, no assumed timezone.

    Excludes events that have already started when called, to avoid presenting
    an in-play fixture as a pregame prediction.
    """
    target = date.fromisoformat(str(target_date))
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('now_utc needs timezone')
    rows=[]
    seen=set()
    for obj in events:
        eid=str(obj.get('id','')).strip()
        home=str(obj.get('home_team','')).strip()
        away=str(obj.get('away_team','')).strip()
        if not eid or not home or not away or home==away or eid in seen:
            continue
        utc=pd.to_datetime(obj.get('commence_time'),utc=True,errors='coerce')
        if pd.isna(utc):
            continue
        local=utc.tz_convert(ET)
        if local.date()!=target or utc.to_pydatetime()<=now:
            continue
        seen.add(eid)
        rows.append({'event_id':eid,'game_date':str(target),
                     'commence_time':utc.isoformat(),
                     'tip_et':local.strftime('%I:%M %p').lstrip('0'),
                     'home_team':home,'away_team':away})
    return pd.DataFrame(rows, columns=['event_id','game_date','commence_time','tip_et','home_team','away_team']).sort_values('commence_time').reset_index(drop=True)


def fetch_upcoming_events(day: str, api_key: str, *, timeout: int = 20, session=None, now_utc=None) -> pd.DataFrame:
    """Obtain the schedule only. Never request any paid/event odds markets."""
    if not api_key:
        raise ValueError('Enter The Odds API key in the left sidebar')
    requester=session or requests
    try:
        response=requester.get(EVENTS_URL,params={'apiKey':api_key},timeout=timeout,
                            headers={'User-Agent':'FirstHalfEdgeResearch/0.3'})
    except requests.RequestException as err:
        # Error strings from requests can embed API-key query parameters.
        raise RuntimeError('Could not reach the NCAA schedule feed; check connectivity and try later.') from err
    try:
        response.raise_for_status()
    except requests.HTTPError as err:
        # Don't echo the request URL: it contains a private API key.
        raise RuntimeError(f'Schedule feed failed (HTTP {response.status_code}). Check your key and account status.') from err
    try:
        payload=response.json()
    except ValueError as err:
        raise RuntimeError('Schedule provider returned invalid JSON') from err
    if not isinstance(payload,list):
        raise RuntimeError('Schedule provider returned an unexpected response; no games substituted')
    return normalize_events(payload,day,now_utc=now_utc)


def _name_map(games: pd.DataFrame) -> dict[str,str]:
    # Full current-season names, by stable team ID (no other-season overrides).
    out={}
    for r in games[['home_id','home_team']].drop_duplicates('home_id').itertuples(index=False):
        out[str(r[0])]=str(r[1])
    for r in games[['away_id','away_team']].drop_duplicates('away_id').itertuples(index=False):
        out[str(r[0])]=str(r[1])
    return out


def analyze_slate(games: pd.DataFrame, fixtures: pd.DataFrame, day: str, *, season: int = 2027) -> pd.DataFrame:
    """Analyze upcoming events using only this season's completed games before target day.

    If an event can't be matched confidently or a team has no games, we provide
    NO ESTIMATE rather than fabricating scores. We do not calculate odds/EV.
    """
    if fixtures.empty:
        return pd.DataFrame(columns=FIELDS)
    target=date.fromisoformat(str(day))
    # Prevent leakage from any completed game on the slate date itself.
    previous_day=(pd.Timestamp(target)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    team_games=prepare_current_season(games,season=season,as_of=previous_day)
    if team_games.empty:
        raise ValueError('No current-season completed games before the selected date')
    ratings,info=fit_current_ratings(team_games)
    current_games=games[pd.to_numeric(games['season'],errors='coerce')==season].copy()
    current_games=current_games[pd.to_datetime(current_games['game_date'],errors='coerce')<pd.Timestamp(target)]
    names=_name_map(current_games)
    known=set(ratings.team_id.astype(str))
    rows=[]
    for f in fixtures.to_dict('records'):
        home=str(f['home_team']);away=str(f['away_team'])
        h_id,hconf=match_team(home,names)
        a_id,aconf=match_team(away,names)
        row={'Game (away @ home)':f'{away} @ {home}', 'Tip (ET)':f['tip_et'],
             'Status':'NO ESTIMATE — unverified team match',
             'Home games':None,'Away games':None,
             'Home 1H scored':None,'Home 1H allowed':None,
             'Away 1H scored':None,'Away 1H allowed':None,
             'Home opp-off strength':None,'Home opp-def strength':None,
             'Away opp-off strength':None,'Away opp-def strength':None,
             'Home 1H projected':None,'Away 1H projected':None,
             'Projected home margin':None,
             'Matched home':names.get(str(h_id),'') if h_id else '',
             'Matched away':names.get(str(a_id),'') if a_id else '',
             'Match confidence':round(min(hconf,aconf),3),
             'Source':'The Odds API /events (schedule only)',
             'Event ID':str(f['event_id'])}
        # Fuzzy match is allowed only with unusually strong and unique evidence.
        # Never use name matching to guess one of two similar schools.
        if (h_id and a_id and h_id != a_id and h_id in known and a_id in known
                and hconf >= .93 and aconf >= .93):
            hs=team_summary(h_id,team_games,ratings)
            aw=team_summary(a_id,team_games,ratings)
            estimate=illustrative_matchup(hs,aw,info,neutral=False)
            row.update({'Home games':hs['games'],'Away games':aw['games'],
                'Home 1H scored':round(hs['scored'],1),'Home 1H allowed':round(hs['allowed'],1),
                'Away 1H scored':round(aw['scored'],1),'Away 1H allowed':round(aw['allowed'],1),
                'Home opp-off strength':round(hs['opponent_off_strength'],1),
                'Home opp-def strength':round(hs['opponent_def_strength'],1),
                'Away opp-off strength':round(aw['opponent_off_strength'],1),
                'Away opp-def strength':round(aw['opponent_def_strength'],1),
                'Home 1H projected':estimate['home_points'],
                'Away 1H projected':estimate['away_points'],
                'Projected home margin':estimate['home_margin'],
                'Status':'RESEARCH ONLY — SMALL SAMPLE' if estimate['min_games']<5 else 'RESEARCH ONLY — NOT BACKTESTED'})
        rows.append(row)
    return pd.DataFrame(rows,columns=FIELDS)
