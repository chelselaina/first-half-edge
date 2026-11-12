"""Optional local Streamlit user interface. Start: streamlit run app.py"""
from __future__ import annotations
import io
import os
import pandas as pd
import streamlit as st
from agent.feeds import get_live_h1_odds, get_history
from agent.model import train,scan
from agent.demo import create_demo

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

tab1,tab2,tab3=st.tabs(['Model validation','First-half scanner','How to use'])
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
