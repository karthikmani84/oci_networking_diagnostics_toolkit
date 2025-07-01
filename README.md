# 🧰 OCI Networking Diagnostics Toolkit – Web Edition

A cross-platform GUI toolkit for system and network diagnostics — built with Streamlit for ease of use by admins and engineers from **any background**.

## 🎯 Why This Toolkit?
Designed to help troubleshoot system and network issues without relying on complex CLI commands. Ideal for hybrid cloud environments, on-prem servers, or cloud compute nodes.

## 🛠 Features

- 🌐 **URL Latency Checker** – Test up to 100 URLs and flag slow responses
- 📶 **Network Tests**:
  - DNS resolution
  - ICMP ping
  - TCP port connectivity
  - Traceroute / Tracepath
  - Bandwidth test via iPerf3 (TCP/UDP)
- 🔓 **Open Ports Scanner** – View listening ports and associated processes
- 🧠 **System Diagnostics** – CPU, disk, memory, uptime, network I/O, and more

## 📦 Requirements

- Python 3.7+
- Streamlit, psutil, requests

```bash
pip install -r requirements.txt
