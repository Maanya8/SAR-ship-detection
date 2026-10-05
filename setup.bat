@echo off
setlocal

cd /d "%~dp0"

echo [1/3] Checking for Python...
where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo Python 3 was not found. Install Python 3.10 or newer and try again.
        exit /b 1
    )
    set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/3] Creating virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        exit /b 1
    )
) else (
    echo [2/3] Reusing existing virtual environment.
)

echo [3/3] Installing project dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install one or more dependencies.
    exit /b 1
)

echo.
echo Setup complete.
echo Activate the environment with:
echo     .venv\Scripts\activate
echo Then open one of the project notebooks or run:
echo     python s1_xview3_inference.py --help

endlocal
