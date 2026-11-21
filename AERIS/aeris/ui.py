"""ui.py - Pure-Python HTML builders for the dashboard (no Streamlit needed, so they are testable)."""
import html

TAG_CLASS = {"SCOUT": "t-scout", "COMM": "t-comm", "ROVER": "t-rover", "MISSION": "t-mission"}


def e(x):
    return html.escape(str(x))


def bar(pct, label=""):
    pct = max(0.0, min(100.0, pct))
    cls = "ok" if pct > 35 else ("warn" if pct > 15 else "crit")
    return (f'<div class="bar"><div class="fill {cls}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="barlabel">{e(label)}</div>')


def kpi(label, value, sub="", pct=None, tone=""):
    b = bar(pct, f"{pct:.0f}%") if pct is not None else ""
    return (f'<div class="kpi {tone}"><div class="kl">{e(label)}</div><div class="kv">{e(value)}</div>'
            f'<div class="ks">{e(sub)}</div>{b}</div>')


def kpi_grid(cards, cols=5):
    return f'<div class="kpi-grid" style="grid-template-columns:repeat({cols},1fr)">' + "".join(cards) + "</div>"


def chip(text, tone=""):
    return f'<span class="chip {tone}">{e(text)}</span>'


def log_html(events, n=60):
    rows = []
    for t, tag, msg in reversed(events[-n:]):          # newest first -> shown at bottom (column-reverse)
        cls = TAG_CLASS.get(tag, "")
        rows.append(f'<div class="ln"><span class="ts">{t:6.0f}s</span>'
                    f'<span class="tag {cls}">[{e(tag)}]</span> {e(msg)}</div>')
    return '<div class="logbox">' + "".join(rows) + "</div>"


def table_html(df, hl_col=None):
    head = "".join(f"<th>{e(c)}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        body += "<tr>" + "".join(
            f'<td class="{"hl" if c == hl_col else ""}">{e(v)}</td>' for c, v in zip(df.columns, r)) + "</tr>"
    return f'<table class="dt"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def packet_table(packets, n=8):
    import pandas as pd
    rows = []
    for p in list(packets)[-n:][::-1]:
        tx = f"{p.tx_end - p.tx_start:.1f}" if p.tx_start is not None else "-"
        q = f"{p.quality:.2f}" if p.quality is not None else "-"
        rows.append((f"#{p.pid}", p.kind.replace("_", " "), f"{p.sender}>{p.receiver}", p.status, len(p.rows), tx, q))
    df = pd.DataFrame(rows, columns=["ID", "Type", "Link", "Status", "Cells", "Tx s", "Q"])
    return table_html(df, "Status") if len(df) else '<div class="muted">No packets yet.</div>'


def verdict(cmp):
    if not cmp["comparable"]:
        extra = " Percentages hidden: the two strategies ended differently."
    else:
        extra = ""
    tone = "good" if cmp["worth_it"] else "bad"
    head = "RECON WORTH ITS ENERGY" if cmp["worth_it"] else "RECON NOT WORTH ITS ENERGY"
    return f'<div class="verdict {tone}"><b>{head}</b><br>{e(cmp["verdict"])}{e(extra)}</div>'


def fmt_pct(v, ok=True):
    return f"{v:+.1f}%" if ok else "n/a"
