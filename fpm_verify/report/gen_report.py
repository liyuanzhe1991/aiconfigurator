# fpm-verify 对齐报告生成器(清单驱动,自包含 HTML + Markdown)。
# 用法: python gen_report.py --manifest verify_manifest.json
# 清单格式:
# {
#   "title": "...", "scope_note": "...",
#   "sections": [{"topo": "tep4", "layer": "decode",
#                 "scores_csv": "...", "scatter_png": "...", "note": "..."}],
#   "out_html": "...", "out_md": "..."
# }
import argparse
import base64
import html as _html
import json
import os
import re as _re

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
args = ap.parse_args()
M = json.load(open(args.manifest))

def stats(csv, layer):
    df = pd.read_csv(csv)
    rows = []
    for side, label in (("old", "old DB"), ("new", "new DB")):
        g = df[df.side == side]
        if layer == "decode" and "stratum" in g:
            g = g[~g.stratum.isin(["F"])]
        if not len(g):
            continue
        rows.append(dict(side=label, n=len(g), mape=g.ape.mean() * 100,
                         p95=g.ape.quantile(.95) * 100, mx=g.ape.max() * 100,
                         gt5=(g.ape > 0.05).mean() * 100,
                         gt10=(g.ape > 0.10).mean() * 100))
    return rows

def band_table(csv):
    df = pd.read_csv(csv)
    g = df[df.side == "new"].copy()
    if "tot" not in g or not len(g):
        return None
    g["signed"] = (g.model - g.truth) / g.truth * 100
    g["band"] = pd.cut(g.tot, [0, 512, 3000, 10**9],
                       labels=["<=512", "700-3000", ">=4096"])
    return g.groupby("band", observed=True).agg(
        n=("signed", "size"), median_signed=("signed", "median"),
        mape=("ape", lambda x: x.mean() * 100)).reset_index()

md = [f"# {M['title']}", ""]
if M.get("scope_note"):
    md += [M["scope_note"], ""]
for s in M["sections"]:
    md.append(f"### {s['topo']} / {s['layer']}\n")
    if s.get("note"):
        md.append(s["note"] + "\n")
    md += ["| DB | n | MAPE | P95 | MAX | >5% | >10% |",
           "|---|---|---|---|---|---|---|"]
    for r in stats(s["scores_csv"], s["layer"]):
        md.append(f"| {r['side']} | {r['n']:,} | {r['mape']:.2f}% | {r['p95']:.2f}% "
                  f"| {r['mx']:.1f}% | {r['gt5']:.1f}% | {r['gt10']:.1f}% |")
    md.append("")
    if s["layer"] == "prefill":
        bt = band_table(s["scores_csv"])
        if bt is not None:
            md += ["| band | n | signed median | MAPE |", "|---|---|---|---|"]
            for _, b in bt.iterrows():
                md.append(f"| {b['band']} | {int(b['n'])} | {b['median_signed']:+.2f}% "
                          f"| {b['mape']:.2f}% |")
            md.append("")

parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
 "<style>html{background:#fff}body{background:#fff;font-family:-apple-system,"
 "Helvetica,Arial,sans-serif;max-width:1080px;margin:24px auto;padding:0 16px;"
 "line-height:1.55;color:#111}table{border-collapse:collapse;margin:8px 0}"
 "td,th{border:1px solid #ccc;padding:4px 10px}img{max-width:100%}"
 "h2,h3{border-bottom:2px solid #76b900;padding-bottom:4px}</style></head><body>"]
in_tbl = False
for ln in "\n".join(md).split("\n"):
    e = _html.escape(ln)
    if ln.startswith("|"):
        cells = [c.strip() for c in e.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if not in_tbl:
            parts.append("<table>")
            in_tbl = True
        tag = "th" if "MAPE" in ln and "%" not in ln else "td"
        parts.append("<tr>" + "".join(f"<{tag}>{c.replace('**','')}</{tag}>"
                                      for c in cells) + "</tr>")
        continue
    if in_tbl:
        parts.append("</table>")
        in_tbl = False
    e2 = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e)
    if ln.startswith("# "):
        parts.append(f"<h1>{e2[2:]}</h1>")
    elif ln.startswith("### "):
        parts.append(f"<h3>{e2[4:]}</h3>")
    elif ln.strip():
        parts.append(f"<p>{e2}</p>")
if in_tbl:
    parts.append("</table>")
for s in M["sections"]:
    p = s.get("scatter_png")
    if p and os.path.exists(p):
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        parts.append(f"<h3>Scatter: {s['topo']} / {s['layer']}</h3>")
        parts.append(f"<img src='data:image/png;base64,{b64}'/>")
parts.append("</body></html>")
open(M["out_html"], "w").write("\n".join(parts))
open(M["out_md"], "w").write("\n".join(md))
print("report:", M["out_html"])
