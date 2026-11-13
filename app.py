"""Optional local Streamlit user interface. Start: streamlit run app.py"""
from __future__ import annotations
import io
import html
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
import pandas as pd
import streamlit as st
from agent.feeds import get_live_h1_odds, get_history
from agent.model import train,scan
from agent.demo import create_demo
from agent.matchups import prepare_current_season, fit_current_ratings, team_summary, illustrative_matchup
from agent.slate import fetch_upcoming_events, analyze_slate

st.set_page_config(page_title='First Half Edge | NCAAB',page_icon='🏀',layout='wide')
st.title('🏀 First Half Edge')
st.caption('NCAA men’s basketball · Research and paper-betting scanner · No automated bet placement')
st.info('Model probabilities are estimates, not guaranteed win rates. Historical score prediction is NOT a betting-profit backtest.')

with st.sidebar:
    st.header('Data & model')
    if st.button('Load FICTIONAL example', use_container_width=True):
        games,lines=create_demo('data')
        st.session_state.games=games
        st.session_state.lines=lines
        st.session_state.demo=True
        for old_key in ('trained','outputs','unmatched'):
            st.session_state.pop(old_key,None)
    st.markdown('### Real NCAA game history')
    st.caption('Free public SportsDataverse results; downloads historical games with verified halftime scores. Available release dates can lag the current season.')
    hist_seasons=st.multiselect('Historical seasons (ending year)',[2023,2024,2025,2026,2027],default=[2027],key='history_seasons_v06',help='2027 means the current 2026–27 basketball season and is selected by default. Add past seasons only when you specifically want historical validation. Current-season matchup projections still use 2027 only.')
    if st.button('Fetch REAL NCAA halftime history',disabled=not hist_seasons,use_container_width=True):
        try:
            with st.spinner('Downloading and checking season files; this may take a while…'):
                years=sorted(hist_seasons)
                new_games=get_history(years, '/tmp/first_half_edge_real_history.csv')
            st.session_state.games=new_games
            st.session_state.demo=False
            st.session_state.history_source='SportsDataverse verified first-half scores'
            for old_key in ('trained','lines','outputs','unmatched'):
                st.session_state.pop(old_key,None)
            st.success(f'Loaded {len(new_games):,} actual completed games. Latest score date: {new_games.game_date.max()}')
        except Exception as exc:
            st.error(f'Real-history import failed. No results were substituted. Details: {exc}')
    if 'games' in st.session_state and not st.session_state.get('demo',True):
        g=st.session_state.games
        st.caption(f'Actual games loaded: {len(g):,} · Most recent: {g.game_date.max()}')
        st.download_button('Save real history CSV',g.to_csv(index=False),file_name='first_half_real_history.csv',mime='text/csv',use_container_width=True)
        if str(g.game_date.max()) < '2026-11-01':
            st.warning('This history does not include recent 2026–27 games. Do not treat any new-season estimates as current or reliable.')
    hist_file=st.file_uploader('Historical games CSV',type='csv',key='game_upload')
    if hist_file is not None:
        if st.button('Use uploaded game history', use_container_width=True):
            st.session_state.games=pd.read_csv(hist_file)
            st.session_state.demo=False
            for old_key in ('trained','lines','outputs','unmatched'):
                st.session_state.pop(old_key,None)
    lines_file=st.file_uploader('Current first-half spreads CSV',type='csv',key='odds_upload')
    if lines_file is not None and st.button('Use uploaded lines',use_container_width=True):
        st.session_state.lines=pd.read_csv(lines_file)
    st.divider()
    st.write('**Optional live odds**')
    day=st.date_input('Eastern game date')
    # Keep credentials out of the GitHub repository and out of widget state.
    # Streamlit Community Cloud secrets must contain ODDS_API_KEY = "...".
    # A missing secrets.toml raises FileNotFoundError when st.secrets is accessed.
    try:
        stored_key = str(st.secrets.get('ODDS_API_KEY', '') or '').strip()
    except (FileNotFoundError, KeyError, AttributeError):
        stored_key = ''
    stored_key = stored_key or os.environ.get('ODDS_API_KEY', '').strip()
    if stored_key:
        st.caption('✓ Odds API connected using a saved key from app settings. No need to enter it again.')
        api_key = stored_key
    else:
        api_key=st.text_input('The Odds API key (temporary; save it in Streamlit Secrets)',type='password')
        st.caption('To save it permanently, open Streamlit Community Cloud → your app → Settings → Secrets. Never commit the key to public GitHub.')
    max_events=st.slider('Max games to query (API usage)',1,50,3)
    st.caption('Fetching a first-half market makes one paid/quota-counted event request per game. Choose a small cap.')
    if st.button('Fetch first-half lines',disabled=not bool(api_key),use_container_width=True):
        with st.spinner('Loading live first-half lines…'):
            try:
                st.session_state.lines=get_live_h1_odds(str(day),api_key,max_events)
                st.session_state.demo=False if 'games' in st.session_state and not str(st.session_state.games.game_id.iloc[0]).startswith('DEMO_') else st.session_state.get('demo',False)
                st.success(f"Loaded {len(st.session_state.lines)} quoted sides; missing events have no supported 1H market.")
            except Exception as e:
                st.error(str(e))

if st.session_state.get('demo'):
    st.error('FICTIONAL DEMONSTRATION DATA. THESE ARE NOT REAL TEAMS, ODDS, OR BETTING OPPORTUNITIES.')

tab6,tab1,tab2,tab3,tab4,tab5=st.tabs(['⚡ Live 1H Cushions','Model validation','First-half scanner','How to use','2026–27 Matchups','Daily slate'])

# ---------------------------------------------------------------------------
# v0.4: readable, portrait-friendly Daily Slate cards.
# This layer changes presentation only. It does NOT validate a betting model.
# ---------------------------------------------------------------------------
def _as_num(value):
    number = pd.to_numeric(value, errors='coerce')
    return float(number) if pd.notna(number) else None



def _market_comparisons(result: pd.DataFrame, quoted: pd.DataFrame | None, target_day: str,
                        now_utc=None) -> pd.DataFrame:
    """Compare signed sportsbook FIRST HALF spreads with model margins.

    A gap is in points, NOT a cover probability, expected ROI, or bet signal.
    Match strictly by provider event id AND date; never guess a team alias.
    """
    fields=['Event ID','Game','Tip (ET)','Selection','Bookmaker','Spread','Odds',
            'Projected team margin','Point gap','Home games','Away games',
            'Quote age (min)','Review status']
    if not isinstance(result,pd.DataFrame) or result.empty or not isinstance(quoted,pd.DataFrame) or quoted.empty:
        return pd.DataFrame(columns=fields)
    required={'event_id','game_date','bookmaker','selection','point','price','home_team','away_team'}
    if not required.issubset(quoted.columns):
        return pd.DataFrame(columns=fields)
    now=pd.Timestamp.now(tz='UTC') if now_utc is None else pd.Timestamp(now_utc)
    if now.tz is None:
        now=now.tz_localize('UTC')
    else:
        now=now.tz_convert('UTC')
    by_event={str(r.get('Event ID')):r for r in result.to_dict('records')}
    comparisons=[]
    for o in quoted.to_dict('records'):
        eid=str(o.get('event_id',''))
        if str(o.get('game_date',''))!=str(target_day) or eid not in by_event:
            continue
        r=by_event[eid]
        margin=_as_num(r.get('Projected home margin'))
        line=_as_num(o.get('point')); price=_as_num(o.get('price'))
        if margin is None or line is None or price is None:
            continue
        game=str(r.get('Game (away @ home)',''))
        home=str(o.get('home_team','')).strip()
        away=str(o.get('away_team','')).strip()
        selected=str(o.get('selection','')).strip()
        # Do not infer a side from fuzzy team names or use a full-game spread.
        if selected==home and game.endswith(' @ '+home):
            projected=margin
        elif selected==away and game.startswith(away+' @ '):
            projected=-margin
        else:
            continue
        age=None
        updated=pd.to_datetime(o.get('last_update'),utc=True,errors='coerce')
        if pd.notna(updated):
            age=float((now-updated).total_seconds()/60)
        h=_as_num(r.get('Home games')); a=_as_num(r.get('Away games'))
        gap=projected+line  # selected margin plus signed spread: >0 favors selected team
        if age is None or age < -10 or age > 90:
            status='STALE OR UNDATED — RECHECK'
        elif h is None or a is None or min(h,a)<3:
            status='SMALL SAMPLE — RECHECK'
        elif gap >= 2.0:
            status='PRICE-GAP WATCH — NOT VALIDATED'
        else:
            status='NO CLEAR POINT GAP'
        comparisons.append({'Event ID':eid,'Game':game,'Tip (ET)':r.get('Tip (ET)',''),
             'Selection':selected,'Bookmaker':str(o.get('bookmaker','')),
             'Spread':line,'Odds':int(price),'Projected team margin':projected,
             'Point gap':round(gap,2),'Home games':h,'Away games':a,
             'Quote age (min)':round(age,1) if age is not None else None,
             'Review status':status})
    if not comparisons:
        return pd.DataFrame(columns=fields)
    out=pd.DataFrame(comparisons,columns=fields)
    # For each game display the most favorable quoted SPREAD and odds combination
    # for initial review, but DON'T claim it is price-adjusted best expected value.
    out=out.drop_duplicates(['Event ID','Selection','Bookmaker','Spread','Odds'])
    return out.sort_values(['Point gap','Odds'],ascending=[False,False],kind='stable').reset_index(drop=True)


def _daily_slate_cards(result: pd.DataFrame, quoted: pd.DataFrame | None, target_day: str):
    """An iPad-friendly board: shortlist at top, game cards below."""
    m=_market_comparisons(result,quoted,target_day)
    has_quotes=not m.empty
    priced_events=set(m['Event ID'].astype(str)) if has_quotes else set()
    watch=m[m['Review status'].eq('PRICE-GAP WATCH — NOT VALIDATED')] if has_quotes else m
    watch_events=set(watch['Event ID'].astype(str)) if not watch.empty else set()
    st.markdown('### Price-gap watchlist')
    x,y,z=st.columns(3)
    x.metric('Games with 1H quotes',len(priced_events))
    y.metric('Review flags',len(watch_events))
    z.metric('No 1H quote',max(0,len(result)-len(priced_events)))
    if not has_quotes:
        st.info('No usable FIRST-HALF sportsbook quotes are loaded for this slate. '
                'The scanner cannot identify a bet-vs-market discrepancy from a score projection alone. '
                'Use “Fetch first-half lines” in the sidebar closer to tip-off, or upload a 1H spreads CSV. '
                'The API may have no supported 1H market for these games.')
    elif watch.empty:
        st.info('No current quotes passed the research watchlist rules. '
                'This does NOT mean the model predicts no winners. It means no sufficiently large '
                'fresh point gap met the screening rule.')
    else:
        st.caption('Flagged when the quoted 1H spread differs by at least 2 points from the experimental '
                   'team-margin projection, with at least 3 completed games per team and a quote updated '
                   'within 90 minutes. These are NOT validated picks or profit estimates.')
        for r in watch.drop_duplicates('Event ID').head(8).to_dict('records'):
            with st.container(border=True):
                st.markdown(f"**{r['Game']}**")
                st.write(f"**Review:** {r['Selection']} {r['Spread']:+g} ({r['Odds']:+d}) "
                         f"at {r['Bookmaker']}")
                st.caption(f"Projected {r['Selection']} margin {r['Projected team margin']:+.1f}; "
                           f"point difference {r['Point gap']:+.1f}. "
                           'No bet until this estimator is backtested against actual lines.')
        st.caption('The quote with the largest point cushion is displayed first; this is NOT a '
                   'price-adjusted comparison of -110 versus -130, and actual book availability must be rechecked.')
    st.divider()
    options=['All games','Market watchlist only','Games with 1H quotes',
             'No verified 1H quotes','Needs team-data review']
    show=st.selectbox('Filter games',options,index=1 if watch_events else 0,
                      key='slate_card_filter_v05')
    sort_option=st.selectbox('Sort games by',
                 ['Scheduled tip time','Largest projected halftime lead','Largest 1H point gap'],
                 key='slate_card_sort_v05')
    view=result.copy()
    view['__margin']=pd.to_numeric(view['Projected home margin'],errors='coerce')
    ids=view['Event ID'].astype(str)
    if show=='Market watchlist only':
        view=view[ids.isin(watch_events)].copy()
    elif show=='Games with 1H quotes':
        view=view[ids.isin(priced_events)].copy()
    elif show=='No verified 1H quotes':
        view=view[~ids.isin(priced_events)].copy()
    elif show=='Needs team-data review':
        h=pd.to_numeric(view['Home games'],errors='coerce')
        a=pd.to_numeric(view['Away games'],errors='coerce')
        view=view[(view['__margin'].isna())|(h<3)|(a<3)].copy()
    if sort_option=='Largest projected halftime lead':
        view['__abs_margin']=view['__margin'].abs().fillna(-1)
        view=view.sort_values('__abs_margin',ascending=False,kind='stable')
    elif sort_option=='Largest 1H point gap':
        gap_map=m.groupby('Event ID')['Point gap'].max().to_dict() if has_quotes else {}
        view['__gap']=view['Event ID'].astype(str).map(gap_map).fillna(-999)
        view=view.sort_values('__gap',ascending=False,kind='stable')
    count=len(view)
    st.caption(f'{count} matchups shown. Model margin is NOT a sportsbook edge.')
    if not count:
        st.info('No games match this filter; choose “All games” to see the slate.')
        return
    pages=(count+9)//10
    page=st.selectbox('Page',list(range(1,pages+1)),
                      format_func=lambda p:f'{p} of {pages}',key='slate_card_page_v05')
    selected=view.iloc[(page-1)*10:page*10]
    for r in selected.to_dict('records'):
        game=str(r.get('Game (away @ home)',''))
        away,home=game.rsplit(' @ ',1) if ' @ ' in game else ('Away team','Home team')
        margin=_as_num(r.get('Projected home margin'))
        h=_as_num(r.get('Home games')); a=_as_num(r.get('Away games'))
        event_id=str(r.get('Event ID',''))
        offers=m[m['Event ID'].astype(str)==event_id] if has_quotes else pd.DataFrame()
        with st.container(border=True):
            st.markdown(f'#### {away} @ {home}')
            st.caption(f"Tip: {r.get('Tip (ET)','Time unavailable')} Eastern")
            if margin is None:
                st.warning('NO ESTIMATE — team matching or historical data insufficient.')
                st.markdown('**Status: NO BET**')
                continue
            hp=_as_num(r.get('Home 1H projected'));ap=_as_num(r.get('Away 1H projected'))
            l,rr=st.columns(2)
            l.metric('Home 1H projected',f'{hp:.1f}' if hp is not None else '—')
            rr.metric('Away 1H projected',f'{ap:.1f}' if ap is not None else '—')
            leader=home if margin>=0 else away
            st.markdown(f'**Model: {leader} by {abs(margin):.1f} at halftime**')
            if h is None or a is None or min(h,a)<5:
                st.caption(f"Early-season sample: {home} {int(h) if h is not None else '—'} "
                           f"games; {away} {int(a) if a is not None else '—'} games")
            if offers.empty:
                st.caption('Sportsbook first-half spread: NOT AVAILABLE in loaded quotes')
                st.markdown('**Status: NO MARKET COMPARISON**')
            else:
                st.markdown('**First-half sportsbook comparisons**')
                for o in offers.head(3).to_dict('records'):
                    st.write(f"{o['Bookmaker']}: {o['Selection']} {o['Spread']:+g} "
                             f"({o['Odds']:+d}) · difference from projection "
                             f"{o['Point gap']:+.1f} pts")
                is_watch=event_id in watch_events
                st.markdown('**Status: REVIEW POINT GAP — not a validated bet**' if is_watch
                            else '**Status: NO VALIDATED BET**')
            with st.expander('Current-season matchup details'):
                st.write(f"**{home}:** {r.get('Home 1H scored','—')} scored; "
                         f"{r.get('Home 1H allowed','—')} allowed")
                st.write(f"**{away}:** {r.get('Away 1H scored','—')} scored; "
                         f"{r.get('Away 1H allowed','—')} allowed")
                st.caption('No possession-based pace, injury feed, or calibrated cover '
                           'probability is attached to this estimator.')


with tab1:
    st.subheader('Train and verify the model')
    if 'games' not in st.session_state:
        st.write('Upload actual historical games or load the fictional demo in the sidebar.')
    else:
        st.write(f"Game rows loaded: **{len(st.session_state.games):,}**")
        if st.button('Train model on this history',type='primary'):
            try:
                with st.spinner('Chronological training, calibration and testing…'):
                    st.session_state.trained=train(st.session_state.games)
                st.success('Model trained; separate future test set was not used in initial fitting.')
            except Exception as e:
                st.error(str(e))
        if 'trained' in st.session_state:
            m=st.session_state.trained['metrics']
            a,b,c=st.columns(3)
            a.metric('Out-of-sample MAE',f"{m['test_mae']:.2f} pts")
            b.metric('Naïve MAE',f"{m['baseline_mae']:.2f} pts")
            c.metric('80% interval coverage',f"{m['test_80pct_interval_coverage']:.1%}")
            st.json(m)
            if not m['model_better_than_baseline']:
                st.warning('Model failed to beat simple baseline: scanner should output NO BET.')
            st.caption('Holdout metrics test HALFTIME MARGIN forecasts, not the profitability of any historical bets.')
with tab2:
    st.subheader('Compare prices against estimated half-time margins')
    threshold=st.slider('Minimum estimated expected ROI to flag',0,20,5,1)/100
    if 'trained' not in st.session_state:
        st.warning('Train a model on the first tab.')
    elif 'lines' not in st.session_state:
        st.write('Upload market odds from your sportsbook or use the optional API fetch in the sidebar.')
    else:
        if st.button('Run first-half scan',type='primary'):
            try:
                outputs,unmatched=scan(st.session_state.games,st.session_state.lines,
                                       st.session_state.trained,min_ev=threshold)
                st.session_state.outputs=outputs;st.session_state.unmatched=unmatched
            except Exception as e:
                st.error(str(e))
        if 'outputs' in st.session_state:
            results=st.session_state.outputs
            if results.empty:
                st.info('NO BET: no valid matched prices.')
            else:
                candidates=results.status.isin(['PAPER CANDIDATE','RECHECK PRICE']).sum()
                st.metric('Research candidates',int(candidates))
                order=['status','selection','bookmaker','line','odds','projected_home_margin',
                       'cover_prob','push_prob','ev_roi','home_games','away_games','price_status']
                st.dataframe(results[order],hide_index=True,use_container_width=True,
                    column_config={'cover_prob':st.column_config.NumberColumn('Cover probability',format='%.1%%'),
                                   'push_prob':st.column_config.NumberColumn('Push probability',format='%.1%%'),
                                   'ev_roi':st.column_config.NumberColumn('Expected ROI',format='%+.1%%')})
                st.download_button('Download scan CSV',results.to_csv(index=False),
                                   'edge_scan.csv','text/csv')
            if st.session_state.get('unmatched'):
                st.warning('Unmatched games: ' + ' | '.join(st.session_state.unmatched[:10]) +
                           '. Add verified name aliases, never guess the opponent.')
            st.caption('Injuries, lineup changes, and weather/travel specifics are not in version 1. Recheck all quotes before any decision.')
with tab3:
    st.markdown('''
1. Download published historical schedules with `python -m agent.cli fetch-history --seasons 2024 2025 2026 2027` or load the shipped demonstration.
2. Train and read the holdout results. Model performance alone cannot prove betting profit.
3. Load bookmaker first-half spread prices as a CSV or enter an API key and request a limited number of events.
4. Scan. **No bet** is a valid outcome. Keep a log of quotes and later grade the picks.

The odds CSV should have `game_date,home_team,away_team,selection,point,price,bookmaker` columns.
The signed `point` is from the selected team's perspective (e.g., -4.5 for a favorite).

See `README.md` for more details, sources, assumptions, and limitations.
''')

with tab4:
    st.subheader('Current-season matchup analyzer')
    st.caption('Uses ONLY the 2026–27 season for team scoring and opponent-adjusted strength. No prior-year team scoring is carried forward.')
    if 'games' not in st.session_state:
        st.info('First load real NCAA history from the sidebar, selecting the 2027 season.')
    elif st.session_state.get('demo',False):
        st.warning('Fictional history is loaded. Fetch real 2027-season game results first.')
    else:
        try:
            current = prepare_current_season(st.session_state.games, season=2027)
            if current.empty:
                st.warning('The loaded CSV does not contain games from the 2026–27 season. Re-fetch or upload the 2027 season.')
            else:
                ratings, rating_info = fit_current_ratings(current)
                st.success(f"2026–27 season: {rating_info['completed_games']:,} completed games through {rating_info['through_date']}; {rating_info['teams']} teams.")
                st.caption('Adjusted strengths use current-season first-half scoring against each team’s actual opponents, shrunk toward the current league average to limit 1–3 game overreaction.')
                lookup=dict(zip(ratings.team_id,ratings.team))
                team_options=sorted(lookup,key=lambda t: lookup[t].lower())
                left,right=st.columns(2)
                with left:
                    home_id=st.selectbox('First team (treated as home)',team_options,format_func=lambda x:lookup[x],index=0,key='current_match_home')
                with right:
                    candidate_away=[t for t in team_options if t != home_id]
                    away_id=st.selectbox('Second team (treated as away)',candidate_away,format_func=lambda x:lookup[x],index=0,key='current_match_away')
                neutral=st.checkbox('Neutral-site game',value=False,key='current_match_neutral')
                h=team_summary(home_id,current,ratings)
                a=team_summary(away_id,current,ratings)
                st.markdown('#### How these teams have performed this season')
                overview=pd.DataFrame([
                    {'Team':h['team'],'Games':h['games'],'1H scored':h['scored'],'1H allowed':h['allowed'],
                     'Last 5 scored':h['recent_scored'],'Last 5 allowed':h['recent_allowed'],
                     'Adj offense vs opponents':h['offense_adj'],'Adj defensive suppression':h['defense_adj'],
                     'Opponents’ offensive strength':h['opponent_off_strength'],'Opponents’ defensive strength':h['opponent_def_strength']},
                    {'Team':a['team'],'Games':a['games'],'1H scored':a['scored'],'1H allowed':a['allowed'],
                     'Last 5 scored':a['recent_scored'],'Last 5 allowed':a['recent_allowed'],
                     'Adj offense vs opponents':a['offense_adj'],'Adj defensive suppression':a['defense_adj'],
                     'Opponents’ offensive strength':a['opponent_off_strength'],'Opponents’ defensive strength':a['opponent_def_strength']},
                ])
                st.dataframe(overview,hide_index=True,use_container_width=True,
                  column_config={c:st.column_config.NumberColumn(format='%.1f') for c in overview.columns if c not in ('Team','Games')})
                st.caption('Adjusted values are POINTS above/below the 2026–27 league scoring average, not possessions or official KenPom ratings. Positive defensive suppression is better.')
                st.markdown('#### Recent opponents and first-half scores')
                col_h,col_a=st.columns(2)
                with col_h:
                    st.write(f"**{h['team']}**")
                    st.dataframe(h['schedule'].head(5).assign(date=lambda d:d.date.dt.strftime('%b %d')).rename(columns={'date':'Date','opponent':'Opponent','venue':'Site','pf':'Scored','pa':'Allowed'}),hide_index=True,use_container_width=True)
                with col_a:
                    st.write(f"**{a['team']}**")
                    st.dataframe(a['schedule'].head(5).assign(date=lambda d:d.date.dt.strftime('%b %d')).rename(columns={'date':'Date','opponent':'Opponent','venue':'Site','pf':'Scored','pa':'Allowed'}),hide_index=True,use_container_width=True)
                st.markdown('#### Experimental matchup estimate')
                projected=illustrative_matchup(h,a,rating_info,neutral=neutral)
                x,y,z=st.columns(3)
                x.metric(f"{h['team']} 1H pts",f"{projected['home_points']:.1f}")
                y.metric(f"{a['team']} 1H pts",f"{projected['away_points']:.1f}")
                z.metric('First-team halftime margin',f"{projected['home_margin']:+.1f}")
                if projected['min_games']<5:
                    st.warning('SMALL SAMPLE: at least one team has fewer than five games this season. Treat this estimate as highly uncertain.')
                st.error('RESEARCH ONLY — NOT A BET. This matchup estimator has not been separately backtested or calibrated against first-half sportsbook lines. Do not interpret the projected margin as a proven betting edge.')
                st.caption('Possession-based pace, efficiency per possession, lineup changes and injuries are NOT available in the uploaded halftime-score CSV; total points are not a valid replacement for pace. Those require play-by-play/box-score or roster feeds.')
        except Exception as exc:
            st.error(f'Current-season analysis could not run: {exc}')


with tab5:
    st.subheader('Daily Slate Board')
    st.caption('Matched first-half lines and price-gap watchlist at the top; game cards below. No sideways scrolling.')
    st.warning('EXPERIMENTAL RESEARCH ONLY. A predicted halftime lead is NOT a verified betting edge. No confirmed betting recommendations are issued.')
    if 'games' not in st.session_state or st.session_state.get('demo',True):
        st.info('Load real 2026–27 history in the sidebar first, or upload your saved history CSV.')
    else:
        st.caption('Uses only completed 2026–27 scores before the selected game date. It does not have verified possession pace, injuries, or player availability.')
        if st.button('Scan scheduled games for selected date',type='primary',disabled=not bool(api_key)):
            try:
                with st.spinner('Reading NCAA fixtures and generating current-season research estimates…'):
                    slate=fetch_upcoming_events(str(day),api_key)
                    if slate.empty:
                        st.session_state['slate_result']=pd.DataFrame()
                    else:
                        st.session_state['slate_result']=analyze_slate(st.session_state.games,slate,str(day))
                    st.session_state['slate_day']=str(day)
            except Exception as exc:
                st.error(f'Daily scanner could not run: {exc}')
        if not api_key:
            st.info('Enter your private Odds API key in the sidebar to retrieve upcoming fixtures. Do not put it in public GitHub.')
        if st.session_state.get('slate_day')==str(day) and 'slate_result' in st.session_state:
            r=st.session_state.slate_result
            if r.empty:
                st.info('No future events found for that Eastern date. No games were invented.')
            else:
                n=int(pd.to_numeric(r['Projected home margin'], errors='coerce').notna().sum())
                c1,c2,c3=st.columns(3)
                c1.metric('Games scheduled',len(r))
                c2.metric('Model estimates',n)
                c3.metric('Confirmed bets',0)
                st.caption('The watchlist screens for POINT gaps in quoted lines; it does not establish profitable bets or cover probabilities.')
                _daily_slate_cards(r, st.session_state.get('lines'), str(day))
                st.download_button('Export full slate CSV',r.to_csv(index=False),
                    file_name=f'first_half_slate_{day}.csv',mime='text/csv')
        else:
            st.caption('Select an Eastern game date in the sidebar and tap Scan to load the slate.')


# ---------------------------------------------------------------------------
# v1.0: usable today, no added Odds API requests.
# Load frozen forecasts from the public GitHub ledger (local fallback) and
# compare them with MANUALLY ENTERED actual FIRST-HALF sportsbook spreads.
# Prices/quotes entered by the visitor are not independently verified.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def _fhe_get_frozen_ledger():
    remote = ('https://raw.githubusercontent.com/chelselaina/'
              'first-half-edge/main/data/paper_predictions.csv')
    error = None
    try:
        request = Request(remote, headers={'User-Agent':'FirstHalfEdge-Research/1.0'})
        with urlopen(request, timeout=8) as reply:
            content=reply.read(1000000)
        return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False), 'Latest GitHub paper ledger'
    except Exception as exc:
        error = str(exc)
    local = Path('data/paper_predictions.csv')
    if local.exists():
        return pd.read_csv(local, dtype=str, keep_default_na=False), 'Local deployed ledger (GitHub fetch failed)'
    return pd.DataFrame(), f'Ledger unavailable: {error}'


def _fhe_american_odds(value):
    raw=str(value).strip().replace('−','-')
    if not raw:
        return None
    try:
        number=int(raw)
    except (TypeError,ValueError):
        return None
    if number <= -100 or number >= 100:
        return number
    return None


def _fhe_game_name(record):
    return f"{record['provider_away']} at {record['provider_home']}"


def _fhe_tip_eastern(value):
    try:
        return pd.Timestamp(value).tz_convert('America/New_York').strftime('%-I:%M %p ET')
    except Exception:
        return 'Unknown tip time'


def _fhe_is_pregame(record):
    try:
        tip=pd.Timestamp(record['tip_utc'])
        if tip.tzinfo is None:
            tip=tip.tz_localize('UTC')
        return tip > pd.Timestamp.now(tz='UTC')
    except Exception:
        return False


# ---------------------------------------------------------------------------
# v1.1: half-time scorecard presentation inspired by the user's supplied
# sports-odds dashboard reference. Keeps the v1.0 manual-market method.
# ---------------------------------------------------------------------------
_FHE_CARD_STYLES = """
<style>
.fhe-score-card {
  background: linear-gradient(130deg,#17283b 0%,#1d3347 100%);
  color: #f5f9ff; border-radius: 17px; margin: 0.65rem 0 1rem;
  border: 1px solid #354a60; overflow: hidden; box-shadow: 0 3px 10px rgba(0,0,0,.12);
}
.fhe-score-header { display:flex; align-items:center; justify-content:space-between;
  gap:.4rem; padding:.75rem 1rem; background:#132335; flex-wrap:wrap; }
.fhe-fixture {font-size: .97rem; font-weight:750; overflow-wrap:anywhere;}
.fhe-tip {font-size: .78rem; color:#b8c8d8;}
.fhe-score-heading {display:grid;grid-template-columns: minmax(0,1fr) 6.5rem 8rem;
  gap:.3rem; padding:.5rem 1rem 0; color:#a7bacd;font-size:.72rem;}
.fhe-score-row {display:grid;grid-template-columns: minmax(0,1fr) 6.5rem 8rem;
  gap:.3rem;align-items:center;padding:.35rem 1rem;}
.fhe-team {font-size: .97rem;font-weight:690; overflow-wrap:anywhere;}
.fhe-team small {display:block;font-weight:400;color:#a9bbcb;font-size:.72rem;}
.fhe-score {font-size:1.24rem;font-weight:790;text-align:center;
  border-radius:8px;background:#f9fbfe;color:#1a2b3e;padding:.33rem .2rem;}
.fhe-line {font-size:.96rem;text-align:center;font-weight:700;
  border:1px solid #3a5267;border-radius:8px;padding:.44rem .1rem;}
.fhe-line small {display:block;font-weight:450;color:#c4d0df;font-size:.72rem;}
.fhe-line-adv {background:#2b5b4a;border-color:#40886b;}
.fhe-score-footer {padding:.68rem 1rem .85rem;border-top:1px solid #3a4e62;
  display:flex;gap:.5rem;justify-content:space-between;align-items:center;flex-wrap:wrap;}
.fhe-tag {font-size:.73rem;letter-spacing:.025rem;font-weight:780;border-radius:99px;
  padding:.32rem .6rem;background:#45576c;color:#fff;}
.fhe-tag-review {background:#d5a12a;color:#102236;}
.fhe-tag-provisional {background:#68532a;color:#fce8ae;}
.fhe-claim {font-size:.82rem;color:#e3ebf2;}
.fhe-small-note {padding:.15rem 1rem .8rem;color:#abbccc;font-size:.72rem;}
@media(max-width:540px) {
  .fhe-score-heading,.fhe-score-row {grid-template-columns:minmax(0,1fr) 4.7rem 6.7rem;}
  .fhe-score-heading {font-size:.65rem;}
  .fhe-score {font-size:1.05rem;}
  .fhe-team {font-size:.87rem;}
  .fhe-line {font-size:.83rem;}
}
</style>
"""


def _fhe_safe_text(value):
    return html.escape(str(value),quote=True)


def _fhe_score_text(value):
    val=pd.to_numeric(value,errors='coerce')
    return f'{float(val):.1f}' if pd.notna(val) else '—'


def _fhe_scorecard(rec, q=None):
    """Preview model HALF-TIME projections beside manually entered 1H spreads.

    q is visitor-entered session data, NOT an independently verified market quote.
    The output must never suggest a betting edge when the market is absent.
    """
    q=q or {}
    home=str(rec.get('provider_home','Home'))
    away=str(rec.get('provider_away','Away'))
    margin=pd.to_numeric(rec.get('predicted_home_margin'),errors='coerce')
    hscore=_fhe_score_text(rec.get('projected_home_h1'))
    ascore=_fhe_score_text(rec.get('projected_away_h1'))
    line=pd.to_numeric(q.get('home_spread'),errors='coerce')
    quoted=pd.notna(line)
    home_line=f'{float(line):+g}' if quoted else '—'
    away_line=f'{-float(line):+g}' if quoted else '—'
    ph=_fhe_american_odds(q.get('home_price',''))
    pa=_fhe_american_odds(q.get('away_price',''))
    home_price=f'{ph:+d}' if ph is not None else 'No price'
    away_price=f'{pa:+d}' if pa is not None else 'No price'
    pregame=_fhe_is_pregame(rec)
    cushion=float(margin)+float(line) if quoted and pd.notna(margin) else None
    cushion_abs=abs(cushion) if cushion is not None else None
    side=home if cushion is not None and cushion >= 0 else away
    selected_price=ph if cushion is not None and cushion >= 0 else pa
    if not quoted:
        pill='NO 1H LINE ENTERED'; pill_class=''
        claim='Model score projection only · compare with an actual first-half spread'
    elif not pregame:
        pill='ALREADY STARTED / FINAL';pill_class=''
        claim='Saved line is for review only; do not treat as a new pregame opportunity'
    elif cushion_abs is not None and cushion_abs < 2.5:
        pill='NO 2.5+ POINT GAP';pill_class=''
        claim=f'{side}: {cushion_abs:.1f}-point model/market difference · pass threshold'
    elif not str(q.get('book','')).strip() or selected_price is None:
        pill='UNVERIFIED GAP';pill_class='fhe-tag-provisional'
        claim=f'{side}: {cushion_abs:.1f}-point projected cushion · add book and odds'
    elif selected_price < -120:
        pill='PRICE ABOVE LIMIT';pill_class='fhe-tag-provisional'
        claim=f'{side}: {cushion_abs:.1f}-point projection gap · price {selected_price:+d} worse than -120'
    else:
        pill='RESEARCH WATCH';pill_class='fhe-tag-review'
        claim=f'{side}: {cushion_abs:.1f}-point projected cushion · manually entered price; recheck at book'
    h_adv=' fhe-line-adv' if quoted and cushion is not None and cushion>=2.5 and pregame else ''
    a_adv=' fhe-line-adv' if quoted and cushion is not None and cushion<=-2.5 and pregame else ''
    book=_fhe_safe_text(str(q.get('book','')).strip() or 'Book not entered') if quoted else 'No sportsbook quote'
    tip=_fhe_safe_text(_fhe_tip_eastern(rec.get('tip_utc','')))
    forecast_stamp=_fhe_safe_text(str(rec.get('forecast_utc','Unknown')))
    home_n=_fhe_safe_text(rec.get('home_games','?'))
    away_n=_fhe_safe_text(rec.get('away_games','?'))
    return f"""<div class="fhe-score-card">
      <div class="fhe-score-header">
        <div class="fhe-fixture">{_fhe_safe_text(away)} @ {_fhe_safe_text(home)}</div>
        <div class="fhe-tip">{tip} · 1st HALF</div>
      </div>
      <div class="fhe-score-heading"><span>TEAM</span><span style="text-align:center">PROJECTED 1H</span><span style="text-align:center">1H SPREAD</span></div>
      <div class="fhe-score-row">
        <span class="fhe-team">{_fhe_safe_text(away)}<small>Away</small></span>
        <span class="fhe-score">{ascore}</span>
        <span class="fhe-line{a_adv}">{away_line}<small>{away_price}</small></span>
      </div>
      <div class="fhe-score-row">
        <span class="fhe-team">{_fhe_safe_text(home)}<small>Home</small></span>
        <span class="fhe-score">{hscore}</span>
        <span class="fhe-line{h_adv}">{home_line}<small>{home_price}</small></span>
      </div>
      <div class="fhe-score-footer">
        <span class="fhe-tag {pill_class}">{_fhe_safe_text(pill)}</span>
        <span class="fhe-claim">{_fhe_safe_text(claim)}</span>
      </div>
      <div class="fhe-small-note">{book} · Current-season sample: {home_n} home / {away_n} away games · Saved forecast UTC {forecast_stamp}</div>
    </div>"""


with tab6:
    st.markdown(_FHE_CARD_STYLES,unsafe_allow_html=True)
    st.subheader('🏀 First-Half Scoreboard & Cushion Checker')
    st.caption('Uses your frozen First Half Edge model projections, NOT an outside model. '
               'Manually enter an actual FIRST-HALF spread and price from your sportsbook. '
               'No extra odds-API calls, no bets placed.')
    st.warning('EXPERIMENTAL RESEARCH ONLY: a point gap is not a proven betting edge. '
               'This early-season model is based on very few games and has not shown '
               'profitable results. Check team injuries, pricing, and whether the market '
               'is specifically FIRST HALF. Never use a full-game spread here.')
    if st.button('Refresh frozen predictions', key='fhe_refresh_ledger'):
        _fhe_get_frozen_ledger.clear()
    ledger, ledger_source = _fhe_get_frozen_ledger()
    st.caption(f'Prediction source: {ledger_source}')
    required={'event_id','game_date','tip_utc','provider_home','provider_away',
              'predicted_home_margin','home_games','away_games','status','forecast_utc'}
    if ledger.empty or not required.issubset(ledger.columns):
        st.error('No usable saved forecast file found. Confirm data/paper_predictions.csv '
                 'exists in your repository, then reload the dashboard.')
    else:
        ledger=ledger.copy()
        ledger['predicted_home_margin']=pd.to_numeric(ledger['predicted_home_margin'], errors='coerce')
        ledger=ledger.dropna(subset=['predicted_home_margin'])
        dates=sorted(ledger['game_date'].dropna().astype(str).unique().tolist(), reverse=True)
        today=datetime.now(ZoneInfo('America/New_York')).date().isoformat()
        chosen_day=st.selectbox('Game date (Eastern)', dates,
                     index=dates.index(today) if today in dates else 0, key='fhe_market_date')
        slate=ledger[ledger['game_date']==chosen_day].sort_values('tip_utc').copy()
        slate=slate.drop_duplicates(subset=['event_id'])
        remain=sum(_fhe_is_pregame(r) for r in slate.to_dict('records'))
        c1,c2,c3=st.columns(3)
        c1.metric('Saved model forecasts',len(slate))
        c2.metric('Not yet tipped off',remain)
        c3.metric('Model picks verified as profitable',0)
        st.caption('The 41 saved projections are not necessarily every game on today’s schedule. '
                   'Games without valid team matching or sufficient sample may be excluded.')
        if 'fhe_saved_quotes_v1' not in st.session_state:
            st.session_state['fhe_saved_quotes_v1']={}
        quotes=st.session_state['fhe_saved_quotes_v1']
        games=slate.to_dict('records')
        labels={str(g['event_id']): f"{_fhe_tip_eastern(g['tip_utc'])} · {_fhe_game_name(g)}"
                for g in games}
        selected=st.selectbox('Select a game to compare',list(labels.keys()),
                              format_func=lambda item: labels[item],key='fhe_game_picker')
        row=next(g for g in games if str(g['event_id'])==selected)
        pred=float(row['predicted_home_margin'])
        home=row['provider_home']
        away=row['provider_away']
        st.caption('Select a matchup to see the projected halftime scores and compare an actual sportsbook line:')
        st.markdown(_fhe_scorecard(row,quotes.get(selected)),unsafe_allow_html=True)
        st.metric(f'Model projected {home} first-half margin',f'{pred:+.1f} pts')
        st.caption(f"Frozen forecast: {row['forecast_utc']} UTC · "
                   f"Current-season sample: {row['home_games']} home-team games, "
                   f"{row['away_games']} away-team games. "
                   'Match confidence measures team-name matching, not predicted accuracy.')
        current=quotes.get(selected,{})
        with st.form(key=f'fhe_quote_form_{selected}',clear_on_submit=False):
            spread=st.text_input(f'{home} FIRST-HALF spread',
                   value=str(current.get('home_spread','')),
                   placeholder='Example: -6.5 or +2.5')
            col_a,col_b=st.columns(2)
            with col_a:
                price_h=st.text_input(f'{home} price',
                   value=str(current.get('home_price','')),
                   placeholder='-110')
            with col_b:
                price_a=st.text_input(f'{away} price',
                   value=str(current.get('away_price','')),
                   placeholder='-110')
            book=st.text_input('Sportsbook name',
                  value=str(current.get('book','')),placeholder='e.g. DraftKings')
            submitted=st.form_submit_button('Save / compare this first-half line',type='primary')
        if submitted:
            try:
                value=float(spread.strip().replace('−','-'))
                if not (-60 <= value <= 60):
                    raise ValueError('spread is out of range')
                if price_h.strip() and _fhe_american_odds(price_h) is None:
                    raise ValueError('Invalid home-team American odds')
                if price_a.strip() and _fhe_american_odds(price_a) is None:
                    raise ValueError('Invalid away-team American odds')
                quotes[selected]={'home_spread':value,'home_price':price_h.strip(),
                                  'away_price':price_a.strip(),'book':book.strip(),
                                  'entered_at':datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %I:%M %p ET')}
                st.success('Line compared and added to today’s ranked watchlist below.')
            except (ValueError,TypeError) as exc:
                st.error(f'Enter the signed 1H spread (e.g. -6.5 or +2.5). {exc}')
        saved=quotes.get(selected)
        if saved:
            margin_gap=pred+float(saved['home_spread'])
            favored=home if margin_gap>=0 else away
            favored_spread=float(saved['home_spread']) if margin_gap>=0 else -float(saved['home_spread'])
            cushion=abs(margin_gap)
            chosen_price=saved['home_price'] if margin_gap>=0 else saved['away_price']
            c1,c2=st.columns(2)
            c1.metric(f'Largest model cushion: {favored}',f'{cushion:.1f} pts')
            c2.metric('Side at entered line',f'{favored} {favored_spread:+g}')
            if not _fhe_is_pregame(row):
                st.error('This game is already underway or complete. Do not treat this as a pregame opportunity.')
            elif cushion < 2.5:
                st.info('PASS: projected cushion is under the 2.5-point research threshold.')
            elif not saved['book'] or not chosen_price:
                st.warning('Provisional cushion only. Enter the named book and the price for the selected side.')
            elif _fhe_american_odds(chosen_price)<-120:
                st.warning('Large projected cushion, but the manually entered price is worse than -120.')
            else:
                st.warning('RESEARCH CANDIDATE: projected gap meets 2.5 points at the '
                           'manually entered price. This is NOT a verified profitable edge '
                           'or a betting recommendation. Independently confirm the quote.')
        st.divider()
        st.markdown('#### Your biggest manually entered first-half gaps')
        found=[]
        lookup={str(g['event_id']):g for g in games}
        for eid,q in quotes.items():
            if eid not in lookup:
                continue
            rr=lookup[eid]
            hg=float(rr['predicted_home_margin'])+float(q['home_spread'])
            is_home=hg>=0
            selected_team=rr['provider_home'] if is_home else rr['provider_away']
            selection_spread=float(q['home_spread'])*(1 if is_home else -1)
            p=q['home_price'] if is_home else q['away_price']
            found.append({'Game':_fhe_game_name(rr),'Tip ET':_fhe_tip_eastern(rr['tip_utc']),
                 'Side':f'{selected_team} {selection_spread:+g}',
                 'Model cushion':round(abs(hg),1),
                 'Book':q['book'] or 'Not entered','Price':p or 'Not entered',
                 'Pregame?': 'Yes' if _fhe_is_pregame(rr) else 'No',
                 'Quote entered ET':q.get('entered_at','')})
        if found:
            board=pd.DataFrame(found).sort_values('Model cushion',ascending=False)
            st.dataframe(board,hide_index=True,use_container_width=True)
            st.download_button('Save your comparison list as CSV',
                data=board.to_csv(index=False).encode('utf-8'),
                file_name=f'first_half_manual_cushions_{chosen_day}.csv',mime='text/csv')
        else:
            st.info('No first-half lines entered for this date yet. Choose a game above '
                    'and enter its 1H spread and price to see a ranked comparison.')
        st.caption('Manually entered lines remain in this browser session only. Download the '
                   'CSV if you want to keep your comparisons. They are NOT imported into '
                   'the audited paper-betting performance ledger.')
        st.divider()
        st.markdown('### First-half projected scorecards — full saved slate')
        st.caption('A match shows projected halftime scores even without a book quote. Highlighted '
                   'green line cells show sizable MODEL/MARKET differences, not validated bets. '
                   'Prices are manually entered and must be checked at your sportsbook.')
        include_all=st.checkbox('Show games already tipped off',value=False,key='fhe_show_finished_cards_v11')
        candidates=[g for g in games if include_all or _fhe_is_pregame(g)]
        candidates=sorted(candidates,key=lambda r:str(r.get('tip_utc','')))
        st.caption(f'{len(candidates)} forecasted matchups in this view; '
                   'this is NOT necessarily every D-I game scheduled today.')
        if not candidates:
            st.info('No unstarted model-forecasted games left. Enable finished games to review the saved forecasts.')
        else:
            count_pages=(len(candidates)+7)//8
            page=st.selectbox('Scorecard page',range(1,count_pages+1),
                format_func=lambda i:f'Page {i} of {count_pages}',key='fhe_scorecard_page_v11')
            for game_card in candidates[(page-1)*8:page*8]:
                eid=str(game_card.get('event_id',''))
                st.markdown(_fhe_scorecard(game_card,quotes.get(eid)),unsafe_allow_html=True)

