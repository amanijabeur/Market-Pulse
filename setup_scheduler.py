"""
setup_scheduler.py — Register the MarketPulse daily scraper with
Windows Task Scheduler.

Runs main.py Monday-Friday at 22:30 local time (Tunisia, UTC+1).
This is after 9:30 PM UTC — safely past US market close in both EDT
(summer) and EST (winter).

Usage
-----
    python setup_scheduler.py          # register / update the task
    python setup_scheduler.py --remove # delete the task
    python setup_scheduler.py --status # check if the task exists
"""

import os
import subprocess
import sys

TASK_NAME  = "MarketPulse_DailyScraper"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BAT_FILE   = os.path.join(SCRIPT_DIR, "run_daily.bat")
RUN_TIME   = "22:30"   # 10:30 PM Tunisia time — after US market close year-round
DAYS       = "MON,TUE,WED,THU,FRI"


def run(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def task_exists() -> bool:
    code, _ = run(["schtasks", "/Query", "/TN", TASK_NAME])
    return code == 0


def register():
    if not os.path.exists(BAT_FILE):
        print(f"ERROR: {BAT_FILE} not found. Run this script from the project folder.")
        sys.exit(1)

    # Delete existing task first so /Create doesn't prompt
    if task_exists():
        run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
        print(f"Removed existing task '{TASK_NAME}'.")

    cmd = [
        "schtasks", "/Create",
        "/TN",  TASK_NAME,
        "/TR",  f'"{BAT_FILE}"',
        "/SC",  "WEEKLY",
        "/D",   DAYS,
        "/ST",  RUN_TIME,
        "/RL",  "HIGHEST",          # run with highest privileges
        "/F",                        # force overwrite if exists
    ]
    code, out = run(cmd)
    if code == 0:
        print(f"Task '{TASK_NAME}' registered successfully.")
        print(f"  Schedule : {DAYS} at {RUN_TIME} local time")
        print(f"  Script   : {BAT_FILE}")
        print(f"  Log      : {os.path.join(SCRIPT_DIR, 'logs', 'scheduler.log')}")
    else:
        print(f"ERROR registering task:\n{out}")
        sys.exit(1)


def remove():
    if not task_exists():
        print(f"Task '{TASK_NAME}' does not exist.")
        return
    code, out = run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
    if code == 0:
        print(f"Task '{TASK_NAME}' removed.")
    else:
        print(f"ERROR removing task:\n{out}")


def status():
    if task_exists():
        _, out = run(["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"])
        print(out)
    else:
        print(f"Task '{TASK_NAME}' is NOT registered.")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--remove":
        remove()
    elif arg == "--status":
        status()
    else:
        register()
