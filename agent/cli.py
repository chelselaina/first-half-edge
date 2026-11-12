"""First Half Edge Agent command-line interface. Run: python -m agent.cli --help"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import joblib
import pandas as pd
from .feeds import get_history,get_live_h1_odds
from .model import train,scan,grade_predictions
from .demo import create_demo


def main():
    p=argparse.ArgumentParser(prog='first-half-agent',description='Research-only NCAA first-half spread scanner; does not place wagers.')
    sub=p.add_subparsers(dest='action',required=True)
    demo=sub.add_parser('demo',help='Generate labeled FICTIONAL training data and odds')
    demo.add_argument('--out',default='data')
    history=sub.add_parser('fetch-history',help='Download actual hoopR historical schedule data')
    history.add_argument('--seasons',type=int,nargs='+',default=[2024,2025,2026,2027])
    history.add_argument('--out',default='data/games.csv')
    odds=sub.add_parser('fetch-odds',help='Fetch 1H spread prices; consumes API credits')
    odds.add_argument('--date',required=True)
    odds.add_argument('--max-events',type=int,default=20)
    odds.add_argument('--bookmaker',help='Optional bookmaker code from The Odds API')
    odds.add_argument('--api-key',default=None)
    odds.add_argument('--out',default='data/odds.csv')
    fit=sub.add_parser('train',help='Train model and run chronological score-validation')
    fit.add_argument('--games',required=True);fit.add_argument('--out',default='data/model.joblib')
    scanning=sub.add_parser('scan',help='Compare upcoming first-half spreads to model')
    scanning.add_argument('--games',required=True);scanning.add_argument('--odds',required=True)
    scanning.add_argument('--model',required=True)
    scanning.add_argument('--out',default='data/edge_scan.csv')
    scanning.add_argument('--min-ev',type=float,default=0.05)
    scanning.add_argument('--aliases',help='Optional CSV with odds_name,history_name')
    grade=sub.add_parser('grade',help='Grade recorded lines against completed actual results')
    grade.add_argument('--bets',required=True);grade.add_argument('--games',required=True)
    grade.add_argument('--out',default='data/graded.csv')
    args=p.parse_args()
    if args.action=='demo':
        games,prices=create_demo(args.out)
        print(f'FICTIONAL DEMO ONLY: {len(games)} games, {len(prices)} market rows -> {args.out}')
    elif args.action=='fetch-history':
        games=get_history(args.seasons,args.out)
        print(f'Saved {len(games):,} VERIFIED historical games to {args.out}')
    elif args.action=='fetch-odds':
        key=args.api_key or os.environ.get('ODDS_API_KEY','')
        data=get_live_h1_odds(args.date,key,args.max_events,bookmaker=args.bookmaker)
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        data.to_csv(args.out,index=False)
        print(f'Saved {len(data)} first-half lines to {args.out}; blank means no supported 1H prices were returned.')
    elif args.action=='train':
        games=pd.read_csv(args.games)
        obj=train(games)
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(obj,args.out)
        print('Chronological holdout validation:')
        print(json.dumps(obj['metrics'],indent=2))
        print(f'Saved model to {args.out}')
    elif args.action=='scan':
        games=pd.read_csv(args.games); prices=pd.read_csv(args.odds)
        obj=joblib.load(args.model)
        aliases=None
        if args.aliases:
            from .model import _canonical
            adf=pd.read_csv(args.aliases)
            aliases={_canonical(r['odds_name']):r['history_name'] for r in adf.to_dict('records')}
        findings,unmatched=scan(games,prices,obj,args.min_ev,aliases)
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        findings.to_csv(args.out,index=False)
        if unmatched:
            print('UNMATCHED EVENTS (not guessed):'); print('\n'.join(' - '+s for s in unmatched[:35]))
        if findings.empty:
            print('NO BET: no valid matchable first-half prices.'); return
        cols=['selection','bookmaker','line','odds','cover_prob','ev_roi','status']
        print(findings[cols].to_string(index=False))
        print(f'Saved complete result to {args.out}; scores/probabilities are MODEL ESTIMATES, not guarantees.')
    elif args.action=='grade':
        result=grade_predictions(pd.read_csv(args.bets),pd.read_csv(args.games))
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(args.out,index=False)
        if result.empty: print('No bets matched completed games; results not inferred.')
        else:
            print(result['grade'].value_counts().to_string())
            print(f'ROI per 1-unit flat wager: {result.profit_per_unit.mean():+.2%}')
        print(f'Saved to {args.out}')

if __name__=='__main__': main()
