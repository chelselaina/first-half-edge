"""Write a static, shareable report of PAPER research results."""
from __future__ import annotations
from html import escape
from pathlib import Path
import pandas as pd


def create_html_report(rows: pd.DataFrame, metrics: dict, outfile: str, fictional: bool = False):
    if fictional:
        banner='FICTIONAL DEMONSTRATION — no real games, teams, prices, predictions or bets appear here.'
    else:
        banner='RESEARCH ONLY — prices must be verified and no model can guarantee profitable results.'
    lines=[]
    for _,r in rows.iterrows():
        status=str(r['status'])
        tagclass='green' if status in ('PAPER CANDIDATE','RECHECK PRICE') else 'dim'
        ev=float(r['ev_roi'])*100
        line=float(r['line'])
        selection=escape(str(r['selection']))
        lines.append(f'<tr><td><strong>{selection}</strong><small>{escape(str(r["matchup"]))}</small></td>'
                     f'<td>{line:+.1f}</td><td>{int(r["odds"]):+d}</td>'
                     f'<td>{float(r["cover_prob"]):.1%}</td>'
                     f'<td class="num {"good" if ev>0 else "bad"}">{ev:+.1f}%</td>'
                     f'<td><span class="tag {tagclass}">{escape(status)}</span></td></tr>')
    if not lines:
        lines=['<tr><td colspan="6">NO BET: no matchable first-half prices.</td></tr>']
    html='''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>First Half Edge — Research Dashboard</title>
<style>
:root{color-scheme:dark;--bg:#0e141b;--panel:#19212b;--line:#2d3946;--sub:#a7b5c5;--fg:#f3f7fa;--green:#8be2b1;--red:#f1a5ac}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,-apple-system,Segoe UI,sans-serif}
main{width:min(1120px,92vw);margin:48px auto 80px}.eyebrow{color:var(--green);font-weight:700;letter-spacing:.13em;font-size:12px}
h1{font-size:clamp(30px,5vw,56px);letter-spacing:-.045em;margin:7px 0 10px}.subtitle{font-size:17px;color:var(--sub);margin-bottom:28px}
.banner{border:1px solid #91694a;background:#493526;color:#ffe1bf;border-radius:12px;padding:14px 18px;font-weight:650;margin-bottom:24px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
.metric b{font-size:28px;display:block;letter-spacing:-.05em}.metric span{display:block;color:var(--sub);font-size:12px;margin-bottom:10px}
.section{background:var(--panel);border:1px solid var(--line);border-radius:16px;margin-top:22px;padding:22px}
h2{font-size:19px;margin:0 0 6px}.desc{color:var(--sub);font-size:13px;margin:0 0 18px}
table{width:100%;border-collapse:collapse;text-align:left}th{color:var(--sub);text-transform:uppercase;letter-spacing:.08em;font-size:10px;padding:12px;border-bottom:1px solid var(--line)}td{padding:15px 12px;border-bottom:1px solid var(--line);vertical-align:middle}td small{display:block;color:var(--sub);margin-top:4px;font-size:11px}td.num{font-weight:700}.good{color:var(--green)}.bad{color:var(--red)}.tag{display:inline-block;border-radius:20px;border:1px solid var(--line);padding:6px 10px;font-size:10px;white-space:nowrap}.tag.green{border-color:#42735c;color:var(--green)}.tag.dim{color:var(--sub)}
.note{color:var(--sub);font-size:13px;line-height:1.65}.tablewrap{overflow:auto}footer{color:var(--sub);font-size:12px;margin-top:24px}
@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}.section{padding:12px}td{padding:11px 8px}th{padding:8px}}
</style></head><body><main>
<div class="eyebrow">NCAAB / FIRST HALF SPREADS / PAPER RESEARCH</div>
<h1>First Half Edge</h1>
<p class="subtitle">An evidence-first scanner that can say <strong>NO BET</strong>.</p>
<div class="banner">'''+escape(banner)+'''</div>
<div class="stats">
<div class="metric"><span>Historical game records</span><b>'''+f'{int(metrics.get("total_games",0)):,}'+'''</b></div>
<div class="metric"><span>Out-of-sample MAE</span><b>'''+f'{metrics.get("test_mae",0):.2f}'+''' <sup>pts</sup></b></div>
<div class="metric"><span>Naive baseline MAE</span><b>'''+f'{metrics.get("baseline_mae",0):.2f}'+''' <sup>pts</sup></b></div>
<div class="metric"><span>80% interval coverage</span><b>'''+f'{metrics.get("test_80pct_interval_coverage",0):.1%}'+'''</b></div>
</div>
<section class="section"><h2>First-half spread scanner</h2><p class="desc">Sorted by estimated return. Quoted odds and game data must match; prices need independent verification.</p>
<div class="tablewrap"><table><thead><tr><th>Selected side</th><th>1H spread</th><th>Odds</th><th>Est. cover</th><th>Est. ROI</th><th>Review status</th></tr></thead><tbody>
'''+''.join(lines)+'''</tbody></table></div></section>
<section class="section"><h2>How these numbers are built</h2>
<p class="note">The model learns from first-half points scored and allowed, adjusted opponent scoring proxies, venue, rest and recent form. It predicts a halftime score margin, then estimates the probability of covering a signed sportsbook spread, including the possibility of a push. Estimated ROI accounts for American odds payout; it is <strong>not</strong> a proven historical wagering return. It cannot see unreported injuries, personnel or late line movement. All research is local; no betting orders are sent.</p>
<p class="note">Scores from this report can be misleading if market odds are old or mismatched. Bookmakers charge a vig, and backtested prediction error is not evidence of profit. For real use, start with historical schedule downloads and record offered lines before game time for a true forward audit.</p>
</section><footer>First Half Edge v0.1 · Research-only prototype · See README.md and tests for source and assumptions.</footer></main></body></html>'''
    path=Path(outfile);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(html,encoding='utf8')
    return path
