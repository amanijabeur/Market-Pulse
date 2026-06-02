@echo off
:: MarketPulse Daily Scraper
:: Runs after US market close. Scheduled by setup_scheduler.py.

cd /d "%~dp0"

echo [%date% %time%] Starting MarketPulse daily pipeline... >> logs\scheduler.log 2>&1

"C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe" main.py >> logs\scheduler.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] Pipeline FAILED with exit code %ERRORLEVEL% >> logs\scheduler.log 2>&1
) else (
    echo [%date% %time%] Pipeline completed successfully. >> logs\scheduler.log 2>&1
)
