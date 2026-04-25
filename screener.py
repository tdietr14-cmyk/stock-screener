"""
Swing Trade Stock Screener
--------------------------
Criteria:
  - Volume > 500,000
  - Rank stocks as READY, WATCHLIST, or REVERSAL candidates
  - Favor trend alignment, pullbacks near SMA9/EMA20, RSI recovery, MACD improvement, and volume

Data Source: Finviz (free, via finviz Python library)
Output: Slack watchlist with staged swing-trade candidates
"""

import math
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
SLACK_MAX_BLOCKS = 50

# ── Screener parameters ───────────────────────────────────────────────────────
MIN_PRICE           = 2.00
MIN_VOLUME          = 500_000
SMA_FAST_PERIOD     = 9
EMA_PULLBACK_PERIOD = 20
SMA_MID_PERIOD      = 50
SMA_TREND_PERIOD    = 180
SMA_LONG_PERIOD     = 200
RSI_PERIOD          = 14
ATR_PERIOD          = 14
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
HISTORY_DAYS        = "1y" # yfinance period for technical calculations
MAX_RESULTS_BY_STAGE = {
    "READY": 20,
    "WATCHLIST": 40,
    "REVERSAL": 20,
}
STAGE_ORDER = ("READY", "WATCHLIST", "REVERSAL")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Pull candidates from Finviz
# ─────────────────────────────────────────────────────────────────────────────

def get_finviz_candidates() -> list[str]:
    """
    Use Finviz's free screener to pre-filter stocks by average volume and minimum price.
    Trend, pullback, and reversal rules are scored later with yfinance data.
    """
    log.info("Fetching candidates from Finviz …")
    foverview = Overview()

    # Finviz filter keys: https://finviz.com/screener.ashx
    filters = {
        "Average Volume": "Over 500K",
        "Price": "Over $2",
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

def as_float(value) -> float | None:
    """Convert scalar pandas/numpy values to finite floats."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def pct_change(current: float | None, prior: float | None) -> float | None:
    if current is None or prior in (None, 0):
        return None
    return ((current / prior) - 1) * 100


def compute_technicals(ticker: str) -> dict | None:
    """
    Download OHLCV history and compute swing-trade indicators.
    Returns a result dict or None if data is insufficient.
    """
    try:
        hist = yf.download(ticker, period=HISTORY_DAYS, progress=False, auto_adjust=True)
        if hist is None or len(hist) < SMA_LONG_PERIOD + 20:
            return None

        hist = hist.dropna()
        if hist.empty:
            return None

        open_ = hist["Open"].squeeze()
        high = hist["High"].squeeze()
        low = hist["Low"].squeeze()
        close  = hist["Close"].squeeze()
        volume = hist["Volume"].squeeze()

        sma9 = close.rolling(SMA_FAST_PERIOD).mean()
        ema20 = close.ewm(span=EMA_PULLBACK_PERIOD, adjust=False).mean()
        sma50 = close.rolling(SMA_MID_PERIOD).mean()
        sma180 = close.rolling(SMA_TREND_PERIOD).mean()
        sma200 = close.rolling(SMA_LONG_PERIOD).mean()
        rsi = calculate_rsi(close)
        atr = calculate_atr(high, low, close)
        avg_volume_20 = volume.rolling(20).mean()

        ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
        ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        histogram  = macd_line - signal_line

        latest_open = as_float(open_.iloc[-1])
        latest_close = as_float(close.iloc[-1])
        latest_high = as_float(high.iloc[-1])
        latest_low = as_float(low.iloc[-1])
        latest_volume = as_float(volume.iloc[-1])
        latest_sma9 = as_float(sma9.iloc[-1])
        latest_ema20 = as_float(ema20.iloc[-1])
        latest_sma50 = as_float(sma50.iloc[-1])
        latest_sma180 = as_float(sma180.iloc[-1])
        latest_sma200 = as_float(sma200.iloc[-1])
        latest_rsi = as_float(rsi.iloc[-1])
        latest_atr = as_float(atr.iloc[-1])
        latest_avg_volume_20 = as_float(avg_volume_20.iloc[-1])
        latest_macd = as_float(macd_line.iloc[-1])
        latest_signal = as_float(signal_line.iloc[-1])
        latest_hist = as_float(histogram.iloc[-1])

        prev_macd = as_float(macd_line.iloc[-2])
        prev_signal = as_float(signal_line.iloc[-2])
        prev_hist = as_float(histogram.iloc[-2])
        hist_3_bars_ago = as_float(histogram.iloc[-4])
        rsi_3_bars_ago = as_float(rsi.iloc[-4])
        sma180_20_bars_ago = as_float(sma180.iloc[-21])
        sma200_20_bars_ago = as_float(sma200.iloc[-21])

        required = [
            latest_open, latest_close, latest_high, latest_low, latest_volume,
            latest_sma9, latest_ema20, latest_sma50, latest_sma180, latest_sma200,
            latest_rsi, latest_atr, latest_avg_volume_20, latest_macd, latest_signal,
            latest_hist, prev_macd, prev_signal, prev_hist, hist_3_bars_ago,
            rsi_3_bars_ago,
        ]
        if any(value is None for value in required):
            return None

        if latest_close < MIN_PRICE:
            return None

        high_63 = as_float(close.tail(63).max())
        high_252 = as_float(close.tail(252).max())
        low_252 = as_float(close.tail(252).min())
        rel_volume = latest_volume / latest_avg_volume_20 if latest_avg_volume_20 else 0
        atr_pct = latest_atr / latest_close * 100
        distance_sma9_pct = pct_change(latest_close, latest_sma9)
        distance_ema20_pct = pct_change(latest_close, latest_ema20)
        drawdown_3m_pct = pct_change(latest_close, high_63)
        drawdown_52w_pct = pct_change(latest_close, high_252)
        rebound_from_52w_low_pct = pct_change(latest_close, low_252)
        sma180_slope_20 = pct_change(latest_sma180, sma180_20_bars_ago)
        sma200_slope_20 = pct_change(latest_sma200, sma200_20_bars_ago)

        return {
            "ticker": ticker,
            "open": latest_open,
            "price": latest_close,
            "high": latest_high,
            "low": latest_low,
            "volume": int(latest_volume),
            "avg_volume_20": int(latest_avg_volume_20),
            "rel_volume": rel_volume,
            "sma9": latest_sma9,
            "ema20": latest_ema20,
            "sma50": latest_sma50,
            "sma180": latest_sma180,
            "sma200": latest_sma200,
            "sma180_slope_20": sma180_slope_20 or 0,
            "sma200_slope_20": sma200_slope_20 or 0,
            "rsi": latest_rsi,
            "rsi_delta_3": latest_rsi - rsi_3_bars_ago,
            "atr_pct": atr_pct,
            "macd": latest_macd,
            "signal": latest_signal,
            "histogram": latest_hist,
            "prev_macd": prev_macd,
            "prev_signal": prev_signal,
            "prev_histogram": prev_hist,
            "histogram_delta_3": latest_hist - hist_3_bars_ago,
            "distance_sma9_pct": distance_sma9_pct or 0,
            "distance_ema20_pct": distance_ema20_pct or 0,
            "drawdown_3m_pct": drawdown_3m_pct or 0,
            "drawdown_52w_pct": drawdown_52w_pct or 0,
            "rebound_from_52w_low_pct": rebound_from_52w_low_pct or 0,
        }

    except Exception as exc:
        log.debug(f"  {ticker}: error – {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Score swing-trade watchlist candidates
# ─────────────────────────────────────────────────────────────────────────────

def add_reason(reasons: list[str], condition: bool, points: int, text: str) -> int:
    if condition:
        reasons.append(text)
        return points
    return 0


def stage_candidate(row: dict) -> dict | None:
    """Classify a stock as READY, WATCHLIST, or REVERSAL."""
    volume_ok = row["volume"] >= MIN_VOLUME or row["avg_volume_20"] >= MIN_VOLUME
    if not volume_ok:
        return None

    price = row["price"]
    trend_up = (
        price > row["sma180"] and
        price > row["sma200"] and
        (row["sma180_slope_20"] > 0 or row["sma200_slope_20"] > 0)
    )
    strong_trend = price > row["sma50"] > row["sma180"] and row["sma200_slope_20"] > 0
    close_above_sma9 = price > row["sma9"]
    full_body_above_sma9 = min(row["open"], price) > row["sma9"]
    near_sma9_trigger = -3.0 <= row["distance_sma9_pct"] <= 2.0
    near_ema20 = abs(row["distance_ema20_pct"]) <= 4.0
    not_extended = row["distance_ema20_pct"] <= 8.0 and row["rsi"] < 70
    rsi_pullback = 35 <= row["rsi"] <= 58
    rsi_recovering = row["rsi_delta_3"] > 0
    macd_bullish_cross = row["prev_macd"] <= row["prev_signal"] and row["macd"] > row["signal"]
    macd_above_signal = row["macd"] > row["signal"]
    histogram_improving = row["histogram"] > row["prev_histogram"] and row["histogram_delta_3"] > 0
    volume_confirming = row["rel_volume"] >= 1.0
    beatdown = row["drawdown_3m_pct"] <= -15 or row["drawdown_52w_pct"] <= -25
    reversal_recovery = close_above_sma9 and rsi_recovering and histogram_improving

    scores = {"READY": 0, "WATCHLIST": 0, "REVERSAL": 0}
    reasons = {"READY": [], "WATCHLIST": [], "REVERSAL": []}
    warnings = []

    scores["READY"] += add_reason(reasons["READY"], trend_up, 25, "above rising long-term trend")
    scores["READY"] += add_reason(reasons["READY"], strong_trend, 10, "SMA50 stacked above SMA180")
    scores["READY"] += add_reason(reasons["READY"], full_body_above_sma9, 20, "full body above SMA9")
    scores["READY"] += add_reason(reasons["READY"], macd_bullish_cross, 20, "MACD bullish cross")
    scores["READY"] += add_reason(reasons["READY"], histogram_improving and macd_above_signal, 15, "MACD momentum improving")
    scores["READY"] += add_reason(reasons["READY"], rsi_pullback and rsi_recovering, 15, "RSI recovering from pullback zone")
    scores["READY"] += add_reason(reasons["READY"], volume_confirming, 10, "volume at/above 20-day average")
    scores["READY"] += add_reason(reasons["READY"], not_extended, 10, "not extended above EMA20")

    scores["WATCHLIST"] += add_reason(reasons["WATCHLIST"], trend_up, 25, "trend still constructive")
    scores["WATCHLIST"] += add_reason(reasons["WATCHLIST"], near_sma9_trigger or near_ema20, 25, "near SMA9/EMA20 trigger area")
    scores["WATCHLIST"] += add_reason(reasons["WATCHLIST"], histogram_improving, 20, "MACD histogram improving")
    scores["WATCHLIST"] += add_reason(reasons["WATCHLIST"], rsi_pullback and rsi_recovering, 20, "RSI rising in pullback zone")
    scores["WATCHLIST"] += add_reason(reasons["WATCHLIST"], volume_confirming, 10, "healthy volume")
    scores["WATCHLIST"] += add_reason(reasons["WATCHLIST"], not_extended, 10, "not chasing extended price")

    scores["REVERSAL"] += add_reason(reasons["REVERSAL"], beatdown, 25, "meaningful beatdown")
    scores["REVERSAL"] += add_reason(reasons["REVERSAL"], close_above_sma9, 20, "reclaimed SMA9")
    scores["REVERSAL"] += add_reason(reasons["REVERSAL"], row["rsi"] < 55 and rsi_recovering, 20, "RSI recovering from lower levels")
    scores["REVERSAL"] += add_reason(reasons["REVERSAL"], macd_bullish_cross or histogram_improving, 20, "momentum reversal forming")
    scores["REVERSAL"] += add_reason(reasons["REVERSAL"], volume_confirming, 10, "volume confirms interest")

    if row["atr_pct"] > 8:
        warnings.append("wide ATR")
    if row["rsi"] >= 70:
        warnings.append("RSI overbought")
    if row["distance_ema20_pct"] > 10:
        warnings.append("extended above EMA20")
    if price < row["sma180"] or price < row["sma200"]:
        warnings.append("below long-term trend")

    eligible = []
    if scores["READY"] >= 75 and full_body_above_sma9 and (macd_bullish_cross or macd_above_signal):
        eligible.append("READY")
    if scores["WATCHLIST"] >= 70 and not full_body_above_sma9:
        eligible.append("WATCHLIST")
    if scores["REVERSAL"] >= 70 and beatdown and reversal_recovery:
        eligible.append("REVERSAL")

    if not eligible:
        return None

    stage = next(stage for stage in STAGE_ORDER if stage in eligible)
    return {
        **row,
        "stage": stage,
        "score": min(scores[stage], 100),
        "reasons": reasons[stage][:4],
        "warnings": warnings[:3],
    }


def sort_and_limit_candidates(candidates: list[dict]) -> list[dict]:
    """Sort by stage and score, then keep Slack output focused."""
    ordered = []
    for stage in STAGE_ORDER:
        stage_candidates = [candidate for candidate in candidates if candidate["stage"] == stage]
        stage_candidates.sort(key=lambda item: (-item["score"], item["atr_pct"], item["ticker"]))
        limit = MAX_RESULTS_BY_STAGE[stage]
        if len(stage_candidates) > limit:
            log.info(f"Showing top {limit} {stage} candidates out of {len(stage_candidates)}.")
        ordered.extend(stage_candidates[:limit])
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Build Slack messages
# ─────────────────────────────────────────────────────────────────────────────

def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def build_result_block(result: dict) -> dict:
    """Build a Slack section block for a watchlist candidate."""
    reason_text = "; ".join(result["reasons"])
    warning_text = f"\n_Watch:_ {', '.join(result['warnings'])}" if result["warnings"] else ""
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*{result['ticker']}* | *{result['stage']}* | Score {result['score']} | ${result['price']:,.2f}\n"
                f"RSI {result['rsi']:.1f} ({fmt_pct(result['rsi_delta_3'])} 3 bars) | "
                f"MACD hist {result['histogram']:.4f} ({fmt_pct(result['histogram_delta_3'])}) | "
                f"Rel vol {result['rel_volume']:.2f}x | ATR {result['atr_pct']:.1f}%\n"
                f"SMA9 {fmt_pct(result['distance_sma9_pct'])} | EMA20 {fmt_pct(result['distance_ema20_pct'])} | "
                f"52w high {fmt_pct(result['drawdown_52w_pct'])}\n"
                f"_Why:_ {reason_text}"
                f"{warning_text}"
            )
        }
    }


def build_stage_header(stage: str, count: int) -> dict:
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*{stage}* ({count})"}
    }


def build_content_blocks(results: list[dict]) -> list[dict]:
    blocks = []
    for stage in STAGE_ORDER:
        stage_results = [result for result in results if result["stage"] == stage]
        if not stage_results:
            continue
        blocks.append(build_stage_header(stage, len(stage_results)))
        blocks.extend(build_result_block(result) for result in stage_results)
    return blocks


def build_slack_messages(results: list[dict]) -> list[dict]:
    """Build one or more Slack Block Kit message payloads."""
    scan_date = datetime.now().strftime("%B %d, %Y  %I:%M %p")
    match_label = f"{len(results)} Candidate{'s' if len(results) != 1 else ''}"
    content_blocks = build_content_blocks(results)

    if not content_blocks:
        content_blocks = [{
            "type": "section",
            "text": {"type": "mrkdwn", "text": "_No staged swing-trade candidates matched today._"}
        }]

    footer_blocks = [
        {"type": "divider"},
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "Not financial advice. Always do your own due diligence."}]
        },
    ]

    header_block_count = 4
    max_result_blocks = SLACK_MAX_BLOCKS - header_block_count - len(footer_blocks)
    total_pages = (len(content_blocks) + max_result_blocks - 1) // max_result_blocks
    messages = []
    first_result = 0

    while first_result < len(content_blocks):
        page_number = len(messages) + 1
        page_suffix = f" (part {page_number}/{total_pages})" if total_pages > 1 else ""

        header_blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Swing Trade Watchlist - {match_label}{page_suffix}", "emoji": True}
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
                        "*Stages:* READY = trigger forming/confirmed | WATCHLIST = near trigger | "
                        "REVERSAL = higher-risk beatdown recovery"
                    )
                }
            },
            {"type": "divider"},
        ]

        next_result = first_result + max_result_blocks
        blocks = header_blocks + content_blocks[first_result:next_result] + footer_blocks
        messages.append({"blocks": blocks})
        first_result = next_result

    return messages


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

    # 2. Compute technicals and classify staged watchlist candidates
    candidates = []
    for i, ticker in enumerate(tickers, 1):
        log.info(f"  [{i}/{len(tickers)}] Analyzing {ticker} …")
        data = compute_technicals(ticker)
        candidate = stage_candidate(data) if data else None
        if candidate:
            log.info(f"    ✅ {ticker} {candidate['stage']} score={candidate['score']}")
            candidates.append(candidate)

    results = sort_and_limit_candidates(candidates)

    log.info(f"\n=== Scan complete: {len(candidates)} staged candidate(s), {len(results)} shown ===")
    for r in results:
        log.info(f"  {r['stage']}  {r['ticker']}  score={r['score']}  price={r['price']:.2f}  "
                 f"rsi={r['rsi']:.1f}  rel_vol={r['rel_volume']:.2f}  hist={r['histogram']:.4f}")

    # 3. Build and send Slack notification
    payloads = build_slack_messages(results)
    if len(payloads) > 1:
        log.info(f"Splitting Slack notification into {len(payloads)} messages to stay under the {SLACK_MAX_BLOCKS}-block cap.")

    for payload in payloads:
        send_slack_message(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
