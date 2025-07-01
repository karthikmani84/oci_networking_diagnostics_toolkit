#!/bin/bash
echo "Launching OCI Diagnostics Toolkit..."
cd "$(dirname "$0")"
streamlit run oci_diagnostics_gui.py
