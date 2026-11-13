"""Optional local Streamlit user interface. Start: streamlit run app.py"""
from __future__ import annotations
import io
import os
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
    hist_seasons=st.multiselect('Historical seasons (ending year)',[2023,2024,2025,2026,2027],default=[2024,2025,2026],help='The 2026 season means 2025–26. The 2027 release may be delayed or not yet contain current games.')
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
    api_key=st.text_input('The Odds API key',type='password',value=os.environ.get('ODDS_API_KEY',''))
    max_events=st.slider('Max games to query (API usage)',1,50,12)
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

tab1,tab2,tab3,tab4,tab5=st.tabs(['Model validation','First-half scanner','How to use','2026–27 Matchups','Daily slate'])

# ---------------------------------------------------------------------------
# v0.4: readable, portrait-friendly Daily Slate cards.
# This layer changes presentation only. It does NOT validate a betting model.
# ---------------------------------------------------------------------------
def _as_num(value):
    number = pd.to_numeric(value, errors='coerce')
    return float(number) if pd.notna(number) else None


def _daily_slate_cards(result: pd.DataFrame, quoted: pd.DataFrame | None, target_day: str):
    """Render game-by-game research cards, never a horizontally scrolling slate."""
    show = st.radio('Games to show', ['All games', 'With projections', 'Needs review'],
                    horizontal=True, key='slate_card_filter')
    sort_option = st.selectbox('Sort games by',
                               ['Scheduled tip time', 'Largest projected halftime lead'],
                               key='slate_card_sort')
    view=result.copy()
    view['__margin']=pd.to_numeric(view['Projected home margin'], errors='coerce')
    if show=='With projections':
        view=view[view['__margin'].notna()].copy()
    elif show=='Needs review':
        view=view[(view['__margin'].isna()) |
                  (view['Status'].fillna('').str.contains('SMALL SAMPLE|NO ESTIMATE',case=False,regex=True))].copy()
    if sort_option=='Largest projected halftime lead':
        view['__abs_margin']=view['__margin'].abs().fillna(-1.0)
        view=view.sort_values('__abs_margin', ascending=False, kind='stable')
    # Scheduled tip order is the original order provided by the event feed.
    count=len(view)
    st.caption(f'{count} matchups shown. Margins describe expected halftime leads, not betting edges.')
    if not count:
        st.info('Nothing matched that filter.')
        return
    pages=(count+9)//10
    page=st.selectbox('Page', list(range(1,pages+1)),
                      format_func=lambda p:f'{p} of {pages}', key='slate_card_page')
    selected=view.iloc[(page-1)*10:page*10]
    q=quoted if isinstance(quoted,pd.DataFrame) else pd.DataFrame()
    has_quotes=not q.empty and {'event_id','game_date','bookmaker','selection','point','price'}.issubset(q.columns)
    for row in selected.to_dict('records'):
        game=str(row['Game (away @ home)'])
        if ' @ ' in game:
            away,home=game.rsplit(' @ ',1)
        else:
            away,home='Away team','Home team'
        margin=_as_num(row.get('Projected home margin'))
        n_home=_as_num(row.get('Home games'))
        n_away=_as_num(row.get('Away games'))
        low_sample=(n_home is None or n_away is None or min(n_home,n_away)<5)
        with st.container(border=True):
            st.markdown(f'#### {away} @ {home}')
            st.caption(f"Tip: {row.get('Tip (ET)','Time unavailable')} Eastern · Research only")
            if margin is None:
                st.warning('NO ESTIMATE — Team data or matching needs verification.')
                st.markdown('**Bet decision: NO BET**')
                with st.expander('Why no estimate?'):
                    st.write(str(row.get('Status','Insufficient information')))
                continue
            hp=_as_num(row.get('Home 1H projected'))
            ap=_as_num(row.get('Away 1H projected'))
            left,right=st.columns(2)
            with left:
                st.metric('Home 1H projection',f'{hp:.1f}' if hp is not None else '—')
                st.caption(home)
            with right:
                st.metric('Away 1H projection',f'{ap:.1f}' if ap is not None else '—')
                st.caption(away)
            leader=home if margin>=0 else away
            st.markdown(f'**Projected halftime leader: {leader}, by {abs(margin):.1f} pts**')
            if low_sample:
                st.caption(f"Low confidence · Games played: {home} {int(n_home) if n_home is not None else '—'}; {away} {int(n_away) if n_away is not None else '—'}." )
            else:
                st.caption('Research projection only; no independently validated cover probability.')
            event_quotes=pd.DataFrame()
            if has_quotes:
                try:
                    event_quotes=q[(q['event_id'].astype(str)==str(row['Event ID'])) &
                                   (q['game_date'].astype(str)==str(target_day))].copy()
                except (KeyError, ValueError):
                    event_quotes=pd.DataFrame()
            if event_quotes.empty:
                st.caption('Sportsbook 1H spread: not available for this game in the loaded quotes.')
            else:
                st.markdown('**Available first-half quotes**')
                # Display book prices without inventing a best available line or bet.
                event_quotes=event_quotes.drop_duplicates(['bookmaker','selection','point','price'])
                for offer in event_quotes.head(6).to_dict('records'):
                    p=_as_num(offer.get('point'))
                    price=_as_num(offer.get('price'))
                    if p is None or price is None: continue
                    st.write(f"{offer.get('bookmaker','Book')}: {offer.get('selection','Team')} {p:+g} ({int(price):+d})")
                st.caption('Quotes may become stale. Verify availability directly with the book.')
            st.markdown('**Bet decision: NO BET — model/market edge not validated**')
            with st.expander('See this season’s matchup stats'):
                st.write(f"**{home}:** {row.get('Home 1H scored','—')} 1H scored; {row.get('Home 1H allowed','—')} allowed; {int(n_home) if n_home is not None else '—'} games.")
                st.write(f"**{away}:** {row.get('Away 1H scored','—')} 1H scored; {row.get('Away 1H allowed','—')} allowed; {int(n_away) if n_away is not None else '—'} games.")
                st.caption('Opponent-adjusted numbers are experimental point-based estimates; true possession pace, lineups and injuries are not yet connected.')


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
    st.caption('Each game at a glance. No sideways scrolling and no implied bets.')
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
                st.caption('NO BET is the default until odds are present and the model is validated against betting markets.')
                _daily_slate_cards(r, st.session_state.get('lines'), str(day))
                st.download_button('Export full slate CSV',r.to_csv(index=False),
                    file_name=f'first_half_slate_{day}.csv',mime='text/csv')
        else:
            st.caption('Select an Eastern game date in the sidebar and tap Scan to load the slate.')
