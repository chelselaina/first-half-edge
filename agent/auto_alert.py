"""Scheduled NCAA 1H *research* watchlist; never places bets.

Loads current-season completed scores, matches today's upcoming fixtures, checks only
near-tip-off eligible events and consumes at most MAX_EVENT_QUERIES market requests.
Creates a GitHub Issue for a sufficiently large *point gap*, not a validated edge.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from agent.feeds import get_history
from agent.slate import fetch_upcoming_events, analyze_slate

ET = ZoneInfo('America/New_York')
URL = 'https://api.the-odds-api.com/v4/sports/basketball_ncaab'
HEADERS = {'User-Agent': 'FirstHalfEdgeResearch/0.7'}
FIELDS = ['checked_at_utc', 'game_date', 'event_id', 'game', 'tip_et', 'selection',
          'bookmaker', 'market', 'spread', 'american_odds', 'model_margin',
          'point_gap', 'home_games', 'away_games', 'quote_updated_utc', 'status']


def as_float(value):
    try:
        out = float(value)
        return out if pd.notna(out) else None
    except (TypeError, ValueError):
        return None


def evaluate_quotes(slate_row: dict, odds_event: dict, now: datetime,
                    minimum_gap: float = 3.0, minimum_games: int = 3,
                    max_quote_age_minutes: int = 90) -> list[dict]:
    """Uses a *signed* sportsbook line. No probability or ROI is inferred."""
    if now.tzinfo is None:
        raise ValueError('now must include timezone')
    home = str(odds_event.get('home_team', '')).strip()
    away = str(odds_event.get('away_team', '')).strip()
    if str(odds_event.get('id', '')) != str(slate_row.get('Event ID')):
        return []
    game = str(slate_row.get('Game (away @ home)', ''))
    if game != f'{away} @ {home}':
        return []
    margin = as_float(slate_row.get('Projected home margin'))
    hgames = as_float(slate_row.get('Home games'))
    agames = as_float(slate_row.get('Away games'))
    if margin is None or hgames is None or agames is None:
        return []
    results = []
    for book in odds_event.get('bookmakers', []):
        for market in book.get('markets', []):
            if market.get('key') != 'spreads_h1':
                continue
            quote_time = pd.to_datetime(market.get('last_update'), utc=True, errors='coerce')
            if pd.isna(quote_time):
                continue
            minutes_ago = (now - quote_time.to_pydatetime()).total_seconds() / 60
            if minutes_ago < -10 or minutes_ago > max_quote_age_minutes:
                continue
            for outcome in market.get('outcomes', []):
                selection = str(outcome.get('name', '')).strip()
                spread = as_float(outcome.get('point'))
                odds = as_float(outcome.get('price'))
                if selection not in (home, away) or spread is None or odds is None:
                    continue
                if odds == 0 or -100 < odds < 100:
                    continue
                selected_margin = margin if selection == home else -margin
                gap = round(selected_margin + spread, 2)
                eligible = (min(hgames, agames) >= minimum_games and
                            gap >= minimum_gap and -150 <= odds <= 200)
                results.append({
                    'checked_at_utc': now.isoformat(),
                    'game_date': now.astimezone(ET).date().isoformat(),
                    'event_id': odds_event['id'], 'game': game,
                    'tip_et': str(slate_row.get('Tip (ET)', '')),
                    'selection': selection,
                    'bookmaker': str(book.get('title') or book.get('key') or 'Unknown'),
                    'market': 'spreads_h1', 'spread': spread,
                    'american_odds': int(odds), 'model_margin': round(selected_margin, 2),
                    'point_gap': gap, 'home_games': int(hgames), 'away_games': int(agames),
                    'quote_updated_utc': quote_time.isoformat(),
                    'status': 'REVIEW POINT GAP — PAPER ONLY' if eligible else 'NO ALERT',
                })
    return results


def select_events(fixtures: pd.DataFrame, slate: pd.DataFrame, now: datetime,
                  max_events: int = 2, min_minutes: int = 20,
                  max_minutes: int = 210) -> list[dict]:
    """Only two near-tip games per run; no odds requests for weak/no estimates."""
    if fixtures.empty or slate.empty:
        return []
    lookup = {str(row['Event ID']): row for row in slate.to_dict('records')}
    selected = []
    for f in fixtures.sort_values('commence_time').to_dict('records'):
        r = lookup.get(str(f['event_id']))
        if not r:
            continue
        game_margin = as_float(r.get('Projected home margin'))
        home_games = as_float(r.get('Home games'))
        away_games = as_float(r.get('Away games'))
        confidence = as_float(r.get('Match confidence'))
        if (game_margin is None or home_games is None or away_games is None or
                confidence is None or min(home_games, away_games) < 3 or confidence < 0.93):
            continue
        kickoff = pd.to_datetime(f['commence_time'], utc=True, errors='coerce')
        if pd.isna(kickoff):
            continue
        until = (kickoff.to_pydatetime() - now).total_seconds() / 60
        if min_minutes <= until <= max_minutes:
            selected.append({'fixture': f, 'slate': r})
        if len(selected) == max_events:
            break
    return selected


def api_event_odds(api_key: str, event_id: str, *, quota_floor: int = 30,
                   timeout: int = 20) -> tuple[dict, int | None]:
    """One market, one region. Exceptions never log private request URLs."""
    try:
        res = requests.get(f'{URL}/events/{event_id}/odds',
            params={'apiKey': api_key, 'regions': 'us', 'markets': 'spreads_h1',
                    'oddsFormat': 'american'}, timeout=timeout, headers=HEADERS)
        remaining = res.headers.get('x-requests-remaining')
        remaining = int(remaining) if remaining and remaining.isdigit() else None
        if res.status_code != 200:
            raise RuntimeError(f'Odds provider returned HTTP {res.status_code}; no quote recorded')
        data = res.json()
        if not isinstance(data, dict):
            raise RuntimeError('Odds provider returned unexpected response')
        return data, remaining
    except requests.RequestException as exc:
        raise RuntimeError('Odds provider request failed; inspect network/provider status') from None


def issue_key(row: dict) -> str:
    src = f"{row['game_date']}|{row['event_id']}|{row['selection']}"
    return hashlib.sha256(src.encode('utf-8')).hexdigest()[:18]


def github_request(method: str, path: str, token: str, repo: str, **kwargs) -> dict:
    response = requests.request(method, f'https://api.github.com/repos/{repo}/{path}',
        headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28'}, timeout=20, **kwargs)
    if response.status_code not in (200, 201):
        raise RuntimeError(f'GitHub API {method} {path.split("?")[0]} returned HTTP {response.status_code}')
    return response.json()


def issue_exists(key: str, token: str, repo: str) -> bool:
    """Check recent issues, including closed ones, to avoid repeat daily alerts."""
    for page in range(1, 6):
        issues = github_request('GET', f'issues?state=all&per_page=100&page={page}', token, repo)
        if not issues:
            return False
        for issue in issues:
            if 'pull_request' not in issue and f'FHE_KEY:{key}' in str(issue.get('body', '')):
                return True
        if len(issues) < 100:
            break
    return False


def notify_github(row: dict, token: str, repo: str) -> str:
    key = issue_key(row)
    if issue_exists(key, token, repo):
        return 'already alerted'
    title = (f"🏀 1H paper watch: {row['selection']} {row['spread']:+g} "
             f"({row['game_date']})")
    body = (f"FHE_KEY:{key}\n\n"
       f"**Research watch only — model not backtested for wagering.** No bet is advised.\n\n"
       f"- **Game:** {row['game']}\n"
       f"- **Tip (ET):** {row['tip_et']}\n"
       f"- **Bookmaker:** {row['bookmaker']}\n"
       f"- **First-half spread:** {row['selection']} {row['spread']:+g} "
       f"({row['american_odds']:+d})\n"
       f"- **Model's selected-team halftime margin:** {row['model_margin']:+.1f}\n"
       f"- **Difference:** {row['point_gap']:+.1f} points\n"
       f"- **Completed current-season games:** home {row['home_games']}, away {row['away_games']}\n"
       f"- **Quote updated:** {row['quote_updated_utc']}\n"
       f"- **Checked:** {row['checked_at_utc']}\n\n"
       "The current-season model is experimental; no calibrated cover probability or profit "
       "estimate has been established. Recheck the sportsbook before wagering. "
       "This issue is publicly visible in a public repository.\n")
    github_request('POST', 'issues', token, repo, json={'title': title, 'body': body})
    return 'issue created'


def save_scan(path: str, rows: list[dict]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def summary(message: str):
    print(message)
    location = os.getenv('GITHUB_STEP_SUMMARY')
    if location:
        with open(location, 'a', encoding='utf-8') as fh:
            fh.write(f'\n{message}\n')


def main():
    now = datetime.now(timezone.utc)
    day = now.astimezone(ET).date().isoformat()
    season = (now.astimezone(ET).year + 1 if now.astimezone(ET).month >= 7
              else now.astimezone(ET).year)
    key = os.getenv('ODDS_API_KEY', '').strip()
    token = os.getenv('GH_TOKEN', '').strip()
    repo = os.getenv('GITHUB_REPOSITORY', '').strip()
    if not key:
        raise SystemExit('ODDS_API_KEY missing from GitHub Actions secrets')
    if not token or not repo:
        raise SystemExit('GITHUB_TOKEN or GITHUB_REPOSITORY unavailable')
    # Always emit an artifact, even on a no-games/no-quotes run.
    output = os.getenv('SCAN_OUTPUT', 'run_artifacts/quotes.csv')
    save_scan(output, [])
    fixtures = fetch_upcoming_events(day, key, now_utc=now)
    if fixtures.empty:
        summary(f'No upcoming NCAA games on {day} that have not tipped off; no odds credits spent.')
        return
    # Published schedule data can lag a day or more. Missing data => no alerts.
    games = get_history([season], '/tmp/fhe_current_games.csv')
    score_date = str(games.game_date.max())
    freshness = (pd.Timestamp(day) - pd.Timestamp(score_date)).days
    if freshness > 5:
        summary(f'Scores last updated {score_date} ({freshness} days old); skip quote scanning and alerts.')
        return
    slate = analyze_slate(games, fixtures, day, season=season)
    choices = select_events(fixtures, slate, now, max_events=2)
    if not choices:
        summary(f'{len(fixtures)} future games; no nearby eligible matchups. No odds credits spent.')
        return
    rows = []
    remaining = None
    for choice in choices:
        if remaining is not None and remaining <= 30:
            summary('Monthly quota safeguard reached; stopped further paid requests.')
            break
        try:
            odds, remaining = api_event_odds(key, choice['fixture']['event_id'])
            rows.extend(evaluate_quotes(choice['slate'], odds, now))
        except RuntimeError as error:
            # Data source failures never trigger alerts.
            summary(f'Event odds unavailable for one fixture: {error}')
    save_scan(output, rows)
    flagged = [r for r in rows if r['status'].startswith('REVIEW')]
    sent = 0
    for r in sorted(flagged, key=lambda x: x['point_gap'], reverse=True):
        outcome = notify_github(r, token, repo)
        if outcome == 'issue created':
            sent += 1
        # cap notification count to avoid floods
        if sent >= 2:
            break
    summary(f'{day}: {len(fixtures)} upcoming events, {len(choices)} events queried, '
            f'{len(rows)} qualifying-market quotes logged, {len(flagged)} price gaps, '
            f'{sent} new research issues. API credits remaining: {remaining if remaining is not None else "unknown"}.')


if __name__ == '__main__':
    main()
