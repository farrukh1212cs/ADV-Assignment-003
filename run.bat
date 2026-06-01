@echo off
cd /d "%~dp0"
echo.
echo ========================================================
echo  ADV Assignment 3 — Visual Analytics Dashboard
echo  Course: COMP-834 Advanced Data Visualization
echo  PAK-AUSTRIA Fachhochschule  ^|  Spring 2026
echo ========================================================
echo.
echo [1/2] Installing / verifying Python dependencies...
pip install -r requirements.txt
echo.
echo [2/2] Launching Streamlit dashboard...
echo       Open your browser at:  http://localhost:8501
echo.
streamlit run app.py
pause
