"""Start the Streamlit app with the models already loading in the background.
Used as the container's command: `python serve.py` (same options as `streamlit run`)."""

import os
import sys

from streamlit.web import cli

from askmyrepo import warmup

# Begins loading both models now, in this process, which is also the one Streamlit runs in.
warmup.start()

port = os.environ.get("PORT", "8501")  # Cloud Run sets PORT; 8501 otherwise
sys.argv = ["streamlit", "run", "app.py", f"--server.port={port}", "--server.address=0.0.0.0",
            "--server.headless=true", "--browser.gatherUsageStats=false",
            "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
cli.main()
