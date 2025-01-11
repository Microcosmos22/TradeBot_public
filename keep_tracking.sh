@echo off
setlocal enabledelayedexpansion

:: Define the path to the Python script
set SCRIPT_PATH=tracking.py

:: Define the maximum number of retries and the delay between retries (in seconds)
set MAX_RETRIES=5
set DELAY=10

:: Counter for retries
set RETRY_COUNT=0

:retry_loop
    :: Run the Python script
    python "%SCRIPT_PATH%"
    
    :: Check if the script executed successfully (exit code 0 means success)
    if %ERRORLEVEL% equ 0 (
        echo Python script executed successfully.
        goto end
    ) else (
        :: Increment retry counter
        set /a RETRY_COUNT+=1
        echo Error encountered. Retrying (%RETRY_COUNT%/%MAX_RETRIES%)...
        
        :: Check if we've reached the maximum retries
        if %RETRY_COUNT% geq %MAX_RETRIES% (
            echo Max retries reached. The script could not be completed successfully.
            goto end
        )

        :: Wait for the specified delay (in seconds)
        timeout /t %DELAY% >nul
        goto retry_loop
    )

:end
pause