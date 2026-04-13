"""
Swing Trade Stock Screener
--------------------------
Criteria:
  - Volume > 500,000
  - Price above 180 SMA
  - MACD histogram <= 35 (user-defined threshold)
  - MACD line dropped below the 9-period Signal line (bearish cross / pullback setup)

Data Source: Finviz (free, via finviz Python library)
Output: Email with matching stocks as an HTML table
"""

import os
import sys
import smtplib
import logging
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview

# ââ Logging ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ââ Email config (set via environment variables or edit directly) âââââââââââââ
EMAIL_SENDER   = os.getenv("EMAIL_SENDER",   "your_email@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your_app_password").replace(" ", "")   # Gmail App Password (strip spaces)
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "recipient@example.com")
SMTP_HOST      = os.getenv("SMTP_HOST",      "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT",  "587"))

# ââ Screener parameters âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
MIN_VOLUME          = 500_000
SMA_PERIOD          = 180
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
MACD_HIST_MAX       = 35   # MACD histogram must be <= this value
HISTORY_DAYS        = "1y" # yfinance period for technical calculations


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# STEP 1 â Pull candidates from Finviz
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def get_finviz_candidates() -> list[str]:
    """
    Use Finviz's free screener to pre-filter stocks by:
      - Average volume >= 500k  (av_mover_u500 filter)
      - Price above SMA 200 as a rough proxy (sma200_pa)
    This narrows the universe before we apply our custom technicals.
    """
    log.info("Fetching candidates from Finviz â¦")
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


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# STEP 2 â Calculate technicals with yfinance
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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

        # ââ 180 SMA ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        sma180 = close.rolling(SMA_PERIOD).mean()

        # ââ MACD âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
        ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        histogram  = macd_line - signal_line

        # ââ Latest values âââââââââââââââââââââââââââââââââââââââââââââââââââââ
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
        log.debug(f"  {ticker}: error â {exc}")
        return None


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# STEP 3 â Apply your exact swing-trade filters
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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
    #   Previous bar: macd > signal   â   Current bar: macd < signal
    crossed_below = (
        row["prev_macd"] >= row["prev_signal"] and
        row["macd"]      <  row["signal"]
    )

    return volume_ok and above_sma and hist_ok and crossed_below

# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# STEP 4 â Build HTML email body
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def build_html_email(results: list[dict]) -> str:
    scan_date = datetime.now().strftime("%B %d, %Y  %I:%M %p")

    if not results:
        return f"""
        <html><body>
        <h2>ð Swing Trade Screener â {scan_date}</h2>
        <p>No stocks matched your criteria today.</p>
        </body></html>
        """

    rows_html = ""
    for r in results:
        macd_color = "#c0392b" if r["macd"] < r["signal"] else "#27ae60"
        rows_html += f"""
        <tr>
          <td><b>{r['ticker']}</b></td>
          <td>${r['price']:,.2f}</td>
          <td>{r['volume']:,}</td>
          <td>${r['sma180']:,.2f}</td>
          <td style="color:{macd_color}">{r['macd']:.4f}</td>
          <td>{r['signal']:.4f}</td>
          <td style="color:#c0392b">{r['histogram']:.4f}</td>
        </tr>
        """

    return f"""
    <html>
    <head>
      <style>
        body  {{ font-family: Arial, sans-serif; background:#f4f4f4; padding:20px; }}
        h2    {{ color:#2c3e50; }}
        p     {{ color:#555; }}
        table {{ border-collapse:collapse; width:100%; background:#fff;
                 box-shadow:0 2px 4px rgba(0,0,0,.1); border-radius:8px; overflow:hidden; }}
        th    {{ background:#2c3e50; color:#fff; padding:12px 15px; text-align:left; font-size:13px; }}
        td    {{ padding:10px 15px; border-bottom:1px solid #eee; font-size:13px; color:#333; }}
        tr:last-child td {{ border-bottom:none; }}
        tr:hover td       {{ background:#f9f9f9; }}
        .badge {{ background:#e74c3c; color:#fff; padding:2px 8px;
                  border-radius:12px; font-size:11px; margin-left:8px; }}
      </style>
    </head>
    <body>
      <h2>ð Swing Trade Screener
        <span class="badge">{len(results)} Match{'es' if len(results)!=1 else ''}</span>
      </h2>
      <p>Scan run: <b>{scan_date}</b></p>
      <p>
        Criteria: Volume &gt; 500K &nbsp;|&nbsp;
        Price &gt; 180 SMA &nbsp;|&nbsp;
        MACD histogram â¤ {MACD_HIST_MAX} &nbsp;|&nbsp;
        MACD crossed below 9-period Signal
      </p>
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Price</th>
            <th>Volume</th>
            <th>180 SMA</th>
            <th>MACD</th>
            <th>Signal (9)</th>
            <th>Histogram</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
      <p style="font-size:11px;color:#aaa;margin-top:20px;">
        â ï¸ This is for informational purposes only. Not financial advice.
        Always do your own due diligence before trading.
      </p>
    </body>
    </html>
    """


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# STEP 5 â Send email
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def send_email(html_body: str, match_count: int):
    subject = (
        f"Swing Screener - {match_count} Match{'es' if match_count != 1 else ''} "
        f"| {datetime.now().strftime('%b %d, %Y')}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    log.info(f"Sending email to {EMAIL_RECEIVER} via {SMTP_HOST}:{SMTP_PORT} â¦")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.set_debuglevel(1)   # Log full SMTP conversation
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        log.info("Email sent successfully.")
    except smtplib.SMTPAuthenticationError as e:
        log.error(f"SMTP Authentication failed: {e}")
        log.error("Check that EMAIL_PASSWORD is a valid Gmail App Password.")
        raise
    except smtplib.SMTPException as e:
        log.error(f"SMTP error: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected email error: {e}")
        log.error(traceback.format_exc())
        raise


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# MAIN
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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
        log.info(f"  [{i}/{len(tickers)}] Analyzing {ticker} â¦")
        data = compute_technicals(ticker)
        if data and passes_filters(data):
            log.info(f"    â {ticker} PASSED")
            results.append(data)

    log.info(f"\n=== Scan complete: {len(results)} stock(s) matched ===")
    for r in results:
        log.info(f"  {r['ticker']}  price={r['price']}  vol={r['volume']:,}  "
                 f"macd={r['macd']}  signal={r['signal']}  hist={r['histogram']}")

    # 3. Build and send email
    html = build_html_email(results)
    send_email(html, len(results))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
