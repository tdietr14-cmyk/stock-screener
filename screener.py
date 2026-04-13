"""
Swing Trade Stock Screener
--------------------------
Criteria:
  - Volume > 500,000
  - Price above 180 SMA
  - MACD histogram <= 35 (user-defined threshold)
  - MACD line dropped below the 9-period Signal line (bearish cross / pullback setup)

Data Source: Finviz (free, via finviz Python library)
Output: Slack message with matching stocks
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Slack config ──────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ── Screener parameters ───────────────────────────────────────────────────────
MIN_VOLUME          = 500_000
SMA_PERIOD          = 180
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
MACD_HIST_MAX       = 35   # MACD histogram must be <= this value
HISTORY_DAYS        = "1y" # yfinance period for technical calculations


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Pull candidates from Finviz
# ─────────────────────────────────────────────────────────────────────────────

def get_finviz_candidates() -> list[str]:
    """
    Use Finviz's free screener to pre-filter stocks by:
      - Average volume >= 500k  (av_mover_u500 filter)
      - Price above SMA 200 as a rough proxy (sma200_pa)
    This narrows the universe before we apply our custom technicals.
    """
    log.info("Fetching candidates from Finviz …")
    foverview = Overview()

    # Finviz filter keys: https://finviz.com/screener.ashx
    filters = {
        "Average Volume": "Over 500K",
        "200-Day Simple Moving Average": "Price above SMA200",
    }

    foverview.set_filter(filters_dict=filters)
    df = foverview.screener_view()

    if df is None or df.empty:
        log.warning("Finviz returned no candidates.")
        return []

    tickers = df["Ticker"].tolist()
    log.info(f"Finviz returned {len(tickers)} candidates.")
    return tickers


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Calculate technicals with yfinance
# ─────────────────────────────────────────────────────────────────────────────

def compute_technicals(ticker: str) -> dict | None:
    """
    Download OHLCV history and compute:
      - 180 SMA
      - MACD line, Signal line, Histogram
    Returns a result dict or None if data is insufficient.
    """
    try:
        hist = yf.download(ticker, period=HISTORY_DAYS, progress=False, auto_adjust=True)
        if hist is None or len(hist) < SMA_PERIOD + MACD_SLOW + MACD_SIGNAL:
            return None

        close  = hist["Close"].squeeze()
        volume = hist["Volume"].squeeze()

        # ── 180 SMA ──────────────────────────────────────────────────────────
        sma180 = close.rolling(SMA_PERIOD).mean()

        # ── MACD ─────────────────────────────────────────────────────────────
        ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
        ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        histogram  = macd_line - signal_line

        # ── Latest values ─────────────────────────────────────────────────────
        latest_close   = float(close.iloc[-1])
        latest_volume  = int(volume.iloc[-1])
        latest_sma180  = float(sma180.iloc[-1])
        latest_macd    = float(macd_line.iloc[-1])
        latest_signal  = float(signal_line.iloc[-1])
        latest_hist    = float(histogram.iloc[-1])

        # Previous bar to detect the cross
        prev_macd   = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])

        return {
            "ticker":       ticker,
            "price":        round(latest_close, 2),
            "volume":       latest_volume,
            "sma180":       round(latest_sma180, 2),
            "macd":         round(latest_macd, 4),
            "signal":       round(latest_signal, 4),
            "histogram":    round(latest_hist, 4),
            "prev_macd":    round(prev_macd, 4),
            "prev_signal":  round(prev_signal, 4),
        }

    except Exception as exc:
        log.debug(f"  {ticker}: error – {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Apply your exact swing-trade filters
# ─────────────────────────────────────────────────────────────────────────────

def passes_filters(row: dict) -> bool:
    """
    Returns True when ALL criteria are met:
      1. Volume  > 500,000
      2. Price   > 180 SMA
      3. MACD histogram <= MACD_HIST_MAX (35)
      4. MACD line crossed BELOW the 9-period signal (bearish crossover / pullback)
    """
    volume_ok   = row["volume"] >= MIN_VOLUME
    above_sma   = row["price"]  >  row["sma180"]
    hist_ok     = row["histogram"] <= MACD_HIST_MAX

    # Detect the drop below signal line:
    #   Previous bar: macd > signal   →   Current bar: macd < signal
    crossed_below = (
        row["prev_macd"] >= row["prev_signal"] and
        row["macd"]      <  row["signal"]
    )

    return volume_ok and above_sma and hist_ok and crossed_below


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Build Slack message
# ─────────────────────────────────────────────────────────────────────────────

def build_slack_message(results: list[dict]) -> dict:
    """Build a Slack Block Kit message payload."""
    scan_date = datetime.now().strftime("%B %d, %Y  %I:%M %p")
    match_label = f"{len(results)} Match{'es' if len(results) != 1 else ''}"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Swing Trade Screener - {match_label}", "emoji": True}
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Scan run: *{scan_date}*"}]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Criteria:* Vol > 500K  |  Price > 180 SMA  |  "
                    f"MACD Hist <= {MACD_HIST_MAX}  |  MACD crossed below Signal(9)"
                )
            }
        },
        {"type": "divider"},
    ]

    if not results:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "_No stocks matched your criteria today._"}
        })
    else:
        for r in results:
            hist_emoji = "🔴" if r["histogram"] < 0 else "🟡"
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{r['ticker']}*  —  ${r['price']:,.2f}\n"
                        f"Vol: {r['volume']:,}  |  180 SMA: ${r['sma180']:,.2f}\n"
                        f"MACD: {r['macd']:.4f}  |  Signal: {r['signal']:.4f}  |  "
                        f"{hist_emoji} Hist: {r['histogram']:.4f}"
                    )
                }
            })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "Not financial advice. Always do your own due diligence."}]
    })

    return {"blocks": blocks}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Send Slack message
# ─────────────────────────────────────────────────────────────────────────────

def send_slack_message(payload: dict):
    """POST a JSON payload to the Slack incoming webhook."""
    if not SLACK_WEBHOOK_URL:
        log.error("SLACK_WEBHOOK_URL is not set. Cannot send notification.")
        raise ValueError("SLACK_WEBHOOK_URL environment variable is missing.")

    log.info("Sending Slack notification …")
    data = json.dumps(payload).encode("utf-8")
    req = Request(SLACK_WEBHOOK_URL, data=data, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            log.info(f"Slack responded: {resp.status} – {body}")
    except HTTPError as e:
        log.error(f"Slack HTTP error {e.code}: {e.read().decode()}")
        raise
    except URLError as e:
        log.error(f"Slack connection error: {e.reason}")
        raise
    except Exception as e:
        log.error(f"Unexpected Slack error: {e}")
        log.error(traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Swing Trade Screener Starting ===")

    # 1. Get universe from Finviz
    tickers = get_finviz_candidates()
    if not tickers:
        log.error("No tickers retrieved. Exiting.")
        return

    # 2. Compute technicals and apply filters
    results = []
    for i, ticker in enumerate(tickers, 1):
        log.info(f"  [{i}/{len(tickers)}] Analyzing {ticker} …")
        data = compute_technicals(ticker)
        if data and passes_filters(data):
            log.info(f"    ✅ {ticker} PASSED")
            results.append(data)

    log.info(f"\n=== Scan complete: {len(results)} stock(s) matched ===")
    for r in results:
        log.info(f"  {r['ticker']}  price={r['price']}  vol={r['volume']:,}  "
                 f"macd={r['macd']}  signal={r['signal']}  hist={r['histogram']}")

    # 3. Build and send Slack notification
    payload = build_slack_message(results)
    send_slack_message(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
