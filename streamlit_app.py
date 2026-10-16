"""Entry point for Streamlit Community Cloud: the dashboard over a committed snapshot of real runs.

Read-only: no evaluation runs here, so no API key is needed and nothing is spent.

    streamlit run streamlit_app.py
"""

import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("REGRESS__STORAGE__DB_PATH", "demo/runs_demo.sqlite")  # config.yaml env override
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("REGRESS_DEMO", "1")

runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
