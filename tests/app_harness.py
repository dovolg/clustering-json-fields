"""Test harness for app.py's main flow.

streamlit.testing.v1.AppTest cannot drive st.file_uploader at all -- it
isn't a supported widget type in that framework, not just awkward to use --
so this harness substitutes a DataFrame read from an env var in its place
and then calls app.main directly. Nothing about the flow is reimplemented
here: app.main is the same function the real app runs, so logic changes in
it can't silently go untested.
"""
import json
import os

import pandas as pd
import streamlit as st

import app


app.init_session_state()

data = pd.DataFrame(json.loads(os.environ["APP_HARNESS_DATA_JSON"]))
ready_data = app.main(data, download_name="test")

# test-only: AppTest can't read st.download_button's payload
if ready_data is not None:
    st.session_state.test_ready_data = ready_data
