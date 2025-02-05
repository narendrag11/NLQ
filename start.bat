@echo off
set venv_path=venv2
set port=8001

:: Check if the virtual environment exists
if not exist "%venv_path%\Scripts\python.exe" (
    echo "Virtual environment not found. Creating virtual environment..."
    python -m venv %venv_path%
    if errorlevel 1 (
        echo "Failed to create virtual environment. Exiting..."
        exit /b 1
    )
    echo "Virtual environment created successfully."
) else (
    echo "Virtual environment found."
)


set /p user_input="Do you want to install requirements.txt? (Y/N):"

if /I "%user_input%"=="Y" (
    echo "Installing Requirements.txt..."
    %venv_path%\Scripts\python -m pip install -r requirements.txt
) else if /I "%user_input%"=="N" (
    echo "Skipping Requirements.txt installation..."
) else (
    "Invalid input. Please enter Y or N.'
)

%venv_path%\Scripts\python mock_server.py
%venv_path%\Scripts\python -m uvicorn mock_server:app --reload --host "localhost" --port %port%