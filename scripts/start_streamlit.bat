@echo off
cd /d "C:\Users\Cheshta\Documents\Codex\2026-05-04\i-want-to-build-a-research"
"C:\Users\Cheshta\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false > streamlit.log 2>&1
