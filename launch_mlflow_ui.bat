@echo off
REM Launch MLflow UI for viewing experiment results
REM This script starts the MLflow UI server and opens it in your browser

echo ================================================================================
echo Starting MLflow UI
echo ================================================================================
echo.
echo The MLflow UI will open in your browser at: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server when you're done.
echo ================================================================================
echo.

REM Start MLflow UI
mlflow ui

REM If MLflow is not found, show installation instructions
if errorlevel 1 (
    echo.
    echo ERROR: MLflow is not installed or not found in PATH
    echo.
    echo To install MLflow, run:
    echo   pip install mlflow
    echo.
    echo Or install all requirements:
    echo   pip install -r requirements.txt
    echo.
    pause
)
