"""CSV history import and optional live odds feed. No API calls without explicit CLI action."""
from __future__ import annotations
import ast
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any
import pandas as pd
import requests

SCHEDULE_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_mens_college_basketball_schedules/mbb_schedule_{season}.csv"
)
API = "https://api.the-odds-api.com/v4/sports/basketball_ncaab"


def _bool(x: Any) -> bool:
    return str(x).strip().lower() in {'1', 'true', 'yes', 't'}


def _period_points(blob: Any) -> list[int]:
    """Decode per-period NCAA scores from JSON, Python, or numeric delimited text.

    The SportsDataverse CSV documentation describes linescores as delimited text;
    old versions of this app only accepted bracketed arrays and excluded every
    otherwise-valid game in some releases. Never infer a score from a game total.
    """
    if blob is None or (isinstance(blob, float) and pd.isna(blob)):
        return []
    if isinstance(blob, (list, tuple)):
        parts = list(blob)
    else:
        raw = str(blob).strip()
        if not raw or raw.lower() in {'nan', 'null', 'none', '[]'}:
            return []
        parts = None
        for decoder in (json.loads, ast.literal_eval):
            try:
                candidate = decoder(raw)
                if isinstance(candidate, (list, tuple)):
                    parts = list(candidate)
                    break
                if isinstance(candidate, dict):
                    candidate = candidate.get('linescores', candidate.get('periods'))
                    if isinstance(candidate, (list, tuple)):
                        parts = list(candidate)
                        break
            except (ValueError, TypeError, SyntaxError):
                pass
        if parts is None:
            # Examples: "38,41", "38|41", "[38 41]", "c(38, 41)".
            # Accept ONLY all-numeric delimited strings, not game-score prose.
            text = raw
            if text.startswith('c(') and text.endswith(')'):
                text = text[2:-1].strip()
            if len(text) >= 2 and text[0] in '[({' and text[-1] in '])}':
                text = text[1:-1].strip()
            if not re.fullmatch(r'\d+(?:\s*[,;|/]\s*|\s+)\d+(?:(?:\s*[,;|/]\s*|\s+)\d+)*', text):
                return []
            parts = re.split(r'\s*[,;|/]\s*|\s+', text)
    vals = []
    for item in parts:
        if isinstance(item, dict):
            item = next((item.get(k) for k in
                ('value', 'score', 'displayValue', 'display_value')
                if item.get(k) is not None), None)
        try:
            value = float(item)
            if not value.is_integer() or value < 0 or value > 120:
                return []
            vals.append(int(value))
        except (TypeError, ValueError):
            return []
    # A regulation NCAA men's game has two halves, plus possible overtime periods.
    return vals if 2 <= len(vals) <= 6 else []


def normalize_schedule(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Transform hoopR season schedules into completed, verified first-half game rows."""
    required = ('game_id','season','home_id','away_id','home_linescores','away_linescores',
                'home_score','away_score','home_display_name','away_display_name')
    missing = [k for k in required if k not in raw.columns]
    if missing:
        raise ValueError(f'Schedule is missing columns: {missing}')
    records = []
    exclusions = {}
    def skip(reason):
        exclusions[reason] = exclusions.get(reason, 0) + 1
    for d in raw.to_dict('records'):
        if 'status_type_completed' in raw.columns and not _bool(d.get('status_type_completed')):
            skip('not final'); continue
        hs = _period_points(d['home_linescores']); aa = _period_points(d['away_linescores'])
        if not hs or not aa:
            skip('missing/ambiguous linescores'); continue
        if len(hs) != len(aa):
            skip('period mismatch'); continue
        try:
            home_total, away_total = int(float(d['home_score'])), int(float(d['away_score']))
        except (ValueError, TypeError):
            skip('missing final scores'); continue
        # NCAA men's has 2 periods in regulation. Reject scores that fail reconciliation.
        if sum(hs) != home_total or sum(aa) != away_total:
            skip('scores not reconciled'); continue
        try:
            dt = pd.to_datetime(d.get('game_date') or d.get('game_date_time') or d.get('date'))
            if pd.isna(dt): raise ValueError('bad date')
            gd = str(dt.date())
        except (ValueError, TypeError, OverflowError):
            skip('missing date'); continue
        home, away = str(d['home_display_name']).strip(), str(d['away_display_name']).strip()
        if home.lower() in {'nan','none',''} or away.lower() in {'nan','none',''}:
            skip('missing team names'); continue
        records.append({
            'game_id':str(d['game_id']),'season':int(d['season']),'game_date':gd,
            'home_id':str(d['home_id']), 'away_id':str(d['away_id']),
            'home_team':home,'away_team':away,
            'home_h1':hs[0], 'away_h1':aa[0],
            'neutral':_bool(d.get('neutral_site', False)),
        })
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out = out.drop_duplicates('game_id').sort_values(['game_date','game_id']).reset_index(drop=True)
    return out, exclusions


def get_history(seasons: list[int], destination: str, timeout: int = 60) -> pd.DataFrame:
    """Download archived hoopR schedules. On some networks GitHub assets may be blocked."""
    frames = []
    for year in seasons:
        url = SCHEDULE_URL.format(season=int(year))
        try:
            response = requests.get(url, timeout=timeout, headers={'User-Agent':'FirstHalfEdgeResearch/0.1'})
            response.raise_for_status()
            from io import BytesIO
            raw = pd.read_csv(BytesIO(response.content), low_memory=False)
            clean, skipped = normalize_schedule(raw)
            if clean.empty:
                samples = raw[['home_linescores','away_linescores','home_score','away_score']].dropna().head(3).to_dict('records')
                raise ValueError(
                    f'No completed, verified first-half games. Rejection counts: {skipped}. '
                    f'Sample source score formats: {str(samples)[:700]}')
            print(f'{year}: {len(clean):,} verified first-half games; excluded: {skipped}')
            frames.append(clean)
        except Exception as e:
            raise RuntimeError(f'Could not load {url}: {e}. You can import the released CSV manually.') from e
    if not frames:
        raise ValueError('Specify at least one season')
    games = pd.concat(frames, ignore_index=True).drop_duplicates('game_id').sort_values(['game_date','game_id'])
    path = Path(destination); path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(path,index=False)
    return games


def _get_json(url: str, key: str, extra: dict | None = None) -> dict | list:
    params = {'apiKey': key}
    if extra: params.update(extra)
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        txt = response.text[:350] if response is not None else ''
        raise RuntimeError(f'Odds provider returned HTTP {response.status_code}: {txt}') from e


def get_live_h1_odds(target_day: str, api_key: str, max_events: int = 20,
                      region: str = 'us', bookmaker: str | None = None) -> pd.DataFrame:
    """Requests an additional-market event endpoint ONCE for each selected event.

    Limits event calls to control usage costs. Does not place bets.
    """
    if not api_key:
        raise ValueError('Set ODDS_API_KEY or provide --api-key')
    if max_events < 1 or max_events > 100:
        raise ValueError('--max-events must be between 1 and 100')
    date.fromisoformat(target_day)
    events = _get_json(API + '/events', api_key)
    scheduled = []
    for event in events:
        ts = pd.to_datetime(event.get('commence_time'), utc=True, errors='coerce')
        if not pd.isna(ts) and str(ts.tz_convert('America/New_York').date()) == target_day:
            scheduled.append(event)
    scheduled.sort(key=lambda x:x['commence_time'])
    if len(scheduled) > max_events:
        print(f'NOTE: {len(scheduled)} events on {target_day}; requesting first {max_events} only. '
              'Further requests consume additional API credits.')
    rows=[]
    for event in scheduled[:max_events]:
        extra={'regions': region, 'markets':'spreads_h1', 'oddsFormat':'american'}
        if bookmaker: extra['bookmakers']=bookmaker
        try:
            result=_get_json(f"{API}/events/{event['id']}/odds", api_key, extra)
        except RuntimeError as ex:
            print(f"Skipped event {event['id']}: {ex}")
            continue
        for b in result.get('bookmakers', []):
            for m in b.get('markets', []):
                if m.get('key') != 'spreads_h1': continue
                for o in m.get('outcomes', []):
                    if o.get('point') is None or o.get('price') is None: continue
                    rows.append({
                        'event_id':result['id'], 'game_date':target_day,
                        'commence_time':result.get('commence_time'),
                        'home_team':result['home_team'], 'away_team':result['away_team'],
                        'bookmaker':b.get('title',b.get('key','unknown')),
                        'selection':o['name'],'point':float(o['point']),
                        'price':int(o['price']), 'last_update':m.get('last_update'),
                    })
    return pd.DataFrame(rows, columns=['event_id','game_date','commence_time','home_team','away_team',
                                      'bookmaker','selection','point','price','last_update'])
