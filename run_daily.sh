#!/bin/bash
# MarketPulse Daily Scraper — Linux / cloud cron script
# Add to crontab with: crontab -e
#
# Cron line (runs Mon-Fri at 21:30 UTC = after US market close year-round):
#   30 21 * * 1-5 /path/to/files/run_daily.sh
#
# To install automatically, run:
#   bash run_daily.sh --install

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/venv/bin/python"
LOG="${SCRIPT_DIR}/logs/scheduler.log"
CRON_EXPR="30 21 * * 1-5"

# Fall back to system python if no venv
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(which python3)"
fi

# --install flag: register the cron job automatically
if [ "$1" = "--install" ]; then
    CRON_CMD="$CRON_EXPR $SCRIPT_DIR/run_daily.sh >> $LOG 2>&1"
    # Remove any existing MarketPulse cron line, then add fresh
    (crontab -l 2>/dev/null | grep -v "run_daily.sh"; echo "$CRON_CMD") | crontab -
    echo "Cron job installed: $CRON_CMD"
    echo "Check with: crontab -l"
    exit 0
fi

# --remove flag: uninstall the cron job
if [ "$1" = "--remove" ]; then
    crontab -l 2>/dev/null | grep -v "run_daily.sh" | crontab -
    echo "Cron job removed."
    exit 0
fi

# Normal run
mkdir -p "$SCRIPT_DIR/logs"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MarketPulse daily pipeline..." >> "$LOG"

cd "$SCRIPT_DIR" || exit 1
"$PYTHON" main.py >> "$LOG" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline FAILED (exit $EXIT_CODE)" >> "$LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline completed successfully." >> "$LOG"
fi
