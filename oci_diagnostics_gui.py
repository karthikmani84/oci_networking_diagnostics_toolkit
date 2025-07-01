import streamlit as st
import subprocess
import platform
import socket
import shutil
import time
import csv
import json
import requests
import psutil
from datetime import datetime, timedelta
from collections import Counter
import os
import re
import pandas as pd

st.set_page_config(page_title="OCI Diagnostics Toolkit", layout="wide")
st.title("🔧 OCI Diagnostics Toolkit – Web Edition")

# Utility Functions
def check_dependency(command):
    return shutil.which(command) is not None

def install_instruction(package, os_type):
    if os_type == "Windows":
        return f"Please install {package} manually via its installer or package manager."
    elif os_type == "Linux":
        return f"Run: `sudo apt install {package}`"
    elif os_type == "Darwin":
        return f"Run: `brew install {package}`"
    return "Unsupported OS."

def export_csv(data, headers, filename_prefix):
    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    return filename

def safe_table(df):
    try:
        st.dataframe(df.reset_index(drop=True), use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)

# Sidebar Prerequisite Check
with st.sidebar.expander("⚙️ Check Requirements"):
    os_type = platform.system()
    st.write(f"Detected OS: `{os_type}`")

    requirements = {
        "iperf3": check_dependency("iperf3"),
        "psutil": True,
        "requests": True,
    }

    for tool, is_available in requirements.items():
        if is_available:
            st.success(f"✅ {tool} available")
        else:
            st.error(f"❌ {tool} not found – {install_instruction(tool, os_type)}")

    st.markdown("---")
    st.markdown("### 🧰 Installer Guide")
    st.code("""
    # Windows
    pip install streamlit requests psutil

    # Mac/Linux
    brew install iperf3  # or sudo apt install iperf3
    pip install streamlit requests psutil
    """, language="bash")

# Sidebar Tool Selection
tool = st.sidebar.selectbox("Choose a Tool", [
    "System Diagnostics",
    "URL Latency Checker",
    "Open Ports Checker",
    "Network Latency & Bandwidth Test"
])

# URL Latency Checker Tool Logic
if tool == "URL Latency Checker":
    st.subheader("🌐 URL Latency Checker")
    urls_input = st.text_area("Enter up to 100 URLs (one per line):", height=200, key="url_input")
    threshold = st.slider("Set threshold for slow response (seconds):", 0.5, 10.0, 1.0, key="threshold_slider")

    if st.button("Run Test"):
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()][:100]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            rows = []
            progress = st.progress(0)
            for i, url in enumerate(urls):
                try:
                    start = time.time()
                    response = requests.get(url, timeout=10)
                    elapsed = round(time.time() - start, 3)
                    code = response.status_code
                except Exception:
                    elapsed = "N/A"
                    code = "Error"
                rows.append((url, elapsed, code))
                progress.progress((i + 1) / len(urls))

            st.write("### 📋 Results")
            df = pd.DataFrame(rows, columns=["URL", "Response Time (s)", "HTTP Code"])
            safe_table(df)

            csv_file = export_csv(rows, ["URL", "Response Time (s)", "HTTP Code"], "url_latency")
            with open(csv_file, "rb") as f:
                st.download_button("⬇️ Download CSV", f, file_name=csv_file)

# Network Latency & Bandwidth Test with independent toggle & sequential execution
if tool == "Network Latency & Bandwidth Test":
    st.subheader("📶 Network Latency & Bandwidth Test")
    mode = st.radio("Select Mode:", ["Client", "Server"], horizontal=True)

    if mode == "Client":
        target = st.text_input("Enter Target IP or Host:")
        port = st.number_input("Target Port:", min_value=1, max_value=65535, value=5201)
        use_udp = st.checkbox("Use UDP")
        parallel = st.number_input("Parallel Streams (-P):", min_value=1, max_value=10, value=1)
        duration = st.number_input("Test Duration (seconds):", min_value=1, max_value=60, value=10)

        st.markdown("---")
        st.markdown("### Select Tests to Run")
        run_dns = st.checkbox("🔍 DNS Resolution")
        run_ping = st.checkbox("📶 ICMP Ping")
        run_tcp_check = st.checkbox("🔗 TCP Port Connectivity Check")
        run_traceroute = st.checkbox("🛣️ Traceroute / Path Discovery")
        run_bandwidth = st.checkbox("📡 Bandwidth Test (iPerf3)")

        if st.button("🚀 Run Selected Tests", key="run_selected"):
            if not target:
                st.warning("Please enter a target host or IP.")
            else:
                progress = st.progress(0)
                step = 0
                total = sum([run_dns, run_ping, run_tcp_check, run_traceroute, run_bandwidth])

                if run_dns:
                    st.markdown("#### 🔍 DNS Resolution")
                    try:
                        resolved_ips = socket.getaddrinfo(target, None)
                        resolved_list = list(set([item[4][0] for item in resolved_ips]))
                        df = pd.DataFrame([[ip] for ip in resolved_list], columns=["Resolved IPs"])
                        safe_table(df)
                        csv_file = export_csv(df.values.tolist(), ["Resolved IPs"], "dns_resolution")
                        with open(csv_file, "rb") as f:
                            st.download_button("⬇️ Download DNS Resolution", f, file_name=csv_file)
                    except Exception as e:
                        st.error(f"DNS Resolution failed: {e}")
                    step += 1
                    progress.progress(step / total)

                if run_ping:
                    st.markdown("#### 📶 ICMP Ping")
                    ping_cmd = ["ping", "-c", "3", target] if os_type != "Windows" else ["ping", target, "-n", "3"]
                    try:
                        result = subprocess.check_output(ping_cmd).decode()
                        st.code(result)
                        csv_file = export_csv([[line] for line in result.splitlines()], ["Ping Output"], "icmp_ping")
                        with open(csv_file, "rb") as f:
                            st.download_button("⬇️ Download Ping Output", f, file_name=csv_file)
                    except Exception as e:
                        st.warning(f"ICMP Ping failed: {e}")
                    step += 1
                    progress.progress(step / total)

                if run_tcp_check:
                    st.markdown("#### 🔗 TCP Port Connectivity Check")
                    try:
                        s = socket.create_connection((target, port), timeout=5)
                        s.close()
                        result = f"TCP connection to {target}:{port} succeeded."
                        st.success(result)
                    except Exception as e:
                        result = f"TCP connection to {target}:{port} failed: {e}"
                        st.error(result)
                    csv_file = export_csv([[result]], ["TCP Check Result"], "tcp_connectivity")
                    with open(csv_file, "rb") as f:
                        st.download_button("⬇️ Download TCP Check", f, file_name=csv_file)
                    step += 1
                    progress.progress(step / total)

                if run_traceroute:
                    st.markdown("#### 🛣️ Traceroute / Path Discovery")
                    if os_type == "Windows":
                        traceroute_cmd = ["tracert", target]
                    elif check_dependency("traceroute"):
                        traceroute_cmd = ["traceroute", target]
                    elif check_dependency("tracepath"):
                        traceroute_cmd = ["tracepath", target]
                    else:
                        traceroute_cmd = None

                    if traceroute_cmd:
                        try:
                            trace_result = subprocess.check_output(traceroute_cmd).decode()
                            st.code(trace_result)
                            csv_file = export_csv([[line] for line in trace_result.splitlines()], ["Traceroute"], "traceroute")
                            with open(csv_file, "rb") as f:
                                st.download_button("⬇️ Download Traceroute", f, file_name=csv_file)
                        except Exception as e:
                            st.warning(f"Traceroute failed: {e}")
                    else:
                        st.error("No supported traceroute utility found on this platform.")
                    step += 1
                    progress.progress(step / total)

                if run_bandwidth and check_dependency("iperf3"):
                    st.markdown("#### 📡 Bandwidth Test (iPerf3)")
                    try:
                        cmd = ["iperf3", "-c", target, "-p", str(port), "-t", str(duration), "-P", str(parallel)]
                        if use_udp:
                            cmd.append("-u")
                        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
                        st.code(result)
                        csv_file = export_csv([[line] for line in result.splitlines()], ["iPerf3 Output"], "bandwidth_test")
                        with open(csv_file, "rb") as f:
                            st.download_button("⬇️ Download Bandwidth Result", f, file_name=csv_file)
                    except Exception as e:
                        st.error(f"Error running iperf3: {e}")
                    step += 1
                    progress.progress(step / total)

    else:
        port = st.number_input("Listening Port:", min_value=1, max_value=65535, value=5201, key="server_port")

        if "iperf_server_process" not in st.session_state:
            st.session_state.iperf_server_process = None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start iPerf3 Server", key="start_server"):
                try:
                    proc = subprocess.Popen(["iperf3", "-s", "-p", str(port)])
                    st.session_state.iperf_server_process = proc
                    st.success(f"iPerf3 server started on port {port}.")
                except Exception as e:
                    st.error(f"Error starting iperf3 server: {e}")

        with col2:
            if st.button("Stop iPerf3 Server", key="stop_server"):
                proc = st.session_state.get("iperf_server_process")
                if proc and proc.poll() is None:
                    proc.terminate()
                    st.session_state.iperf_server_process = None
                    st.success("iPerf3 server stopped.")
                else:
                    st.warning("No active iPerf3 server process found.")

# Open Ports Checker
if tool == "Open Ports Checker":
    st.subheader("🔓 Open Ports Checker")
    if st.button("Scan Open Ports", key="scan_ports_btn"):
        open_ports = []
        system = os_type.lower()
        try:
            if system == "windows":
                result = subprocess.check_output(["netstat", "-ano"], stderr=subprocess.DEVNULL).decode()
                for line in result.splitlines():
                    if "LISTENING" in line or "UDP" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            proto = parts[0]
                            local_address = parts[1]
                            state = parts[3] if proto.lower() == "tcp" else "OPEN"
                            pid = parts[4] if proto.lower() == "tcp" else "N/A"
                            port = local_address.split(":")[-1]
                            try:
                                service_name = socket.getservbyport(int(port)) if port.isdigit() else "Unknown"
                            except:
                                service_name = "Unknown"
                            open_ports.append([proto.upper(), f"Port {port}", state, f"PID {pid}", service_name])
            elif system == "linux" and check_dependency("ss"):
                result = subprocess.check_output(["ss", "-tunlp"], stderr=subprocess.DEVNULL).decode()
                st.code(result, language="text")
                for line in result.splitlines()[1:]:
                    parts = re.split(r'\s+', line, maxsplit=5)
                    if len(parts) >= 6:
                        proto = parts[0].upper()
                        local_address = parts[4]
                        port = local_address.split(":")[-1]
                        pid_match = re.search(r'pid=(\d+)', parts[5])
                        pid = pid_match.group(1) if pid_match else "N/A"
                        try:
                            service_name = socket.getservbyport(int(port)) if port.isdigit() else "Unknown"
                        except:
                            service_name = "Unknown"
                        open_ports.append([proto, f"Port {port}", "OPEN", f"PID {pid}", service_name])
            elif system == "darwin":
                result = subprocess.check_output("lsof -i -n -P", shell=True).decode()
                for line in result.splitlines()[1:]:
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 9:
                        command = parts[0]
                        pid = parts[1]
                        protocol = parts[7].upper()
                        if protocol in ["TCP", "UDP"]:
                            port = parts[8].split(":")[-1]
                            try:
                                service_name = socket.getservbyport(int(port)) if port.isdigit() else "Unknown"
                            except:
                                service_name = "Unknown"
                            open_ports.append([protocol, f"Port {port}", "OPEN", f"{command} (PID {pid})", service_name])
        except Exception as e:
            st.error(f"Error: {e}")

        st.write("### 📋 Listening Ports")
        df = pd.DataFrame(open_ports, columns=["Service", "Port Number", "State", "Process ID", "Process Name"])
        st.dataframe(df.reset_index(drop=True), use_container_width=True)

        if open_ports:
            csv_file = export_csv(open_ports, ["Service", "Port Number", "State", "Process ID", "Process Name"], "open_ports")
            with open(csv_file, "rb") as f:
                st.download_button("⬇️ Download CSV", f, file_name=csv_file, key="dl_open_ports")


# --- System Diagnostics ---
if tool == "System Diagnostics":
    st.subheader("🧠 System Diagnostics")
    if st.button("Run Diagnostics"):
        diagnostics_data = []

        # OS Info
        uname = platform.uname()
        os_info = [
            ["OS", uname.system],
            ["Node Name", uname.node],
            ["Release", uname.release],
            ["Version", uname.version],
            ["Machine", uname.machine],
            ["Processor", uname.processor],
        ]
        diagnostics_data += os_info
        with st.expander("🧩 OS & System Info", expanded=True):
            df = pd.DataFrame(os_info, columns=["Detail", "Value"])
            safe_table(df)
            with open(export_csv(os_info, ["Detail", "Value"], "os_info"), "rb") as f:
                st.download_button("⬇️ Download OS Info", f, file_name=f.name)

        # Disk Usage
        total, used, free = shutil.disk_usage("/")
        disk_info = [
            ["Total (GB)", total // (2**30)],
            ["Used (GB)", used // (2**30)],
            ["Free (GB)", free // (2**30)],
        ]
        diagnostics_data += [[f"Disk {x[0]}", x[1]] for x in disk_info]
        with st.expander("🧱 Disk Usage", expanded=True):
            df = pd.DataFrame(disk_info, columns=["Metric", "Value"])
            safe_table(df)
            with open(export_csv(disk_info, ["Metric", "Value"], "disk_usage"), "rb") as f:
                st.download_button("⬇️ Download Disk Usage", f, file_name=f.name)

        # Top Memory Processes
        memory_processes = []
        for p in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                info = p.info
                mem = info['memory_info'].rss / (1024 * 1024)
                memory_processes.append([info['pid'], info['name'], round(mem, 2)])
            except:
                continue
        memory_processes.sort(key=lambda x: x[2], reverse=True)
        diagnostics_data.append(["Top Memory Processes", "See separate table"])
        mem_df = pd.DataFrame(memory_processes[:10], columns=["PID", "Process Name", "Memory (MB)"])
        with st.expander("🧠 Top Memory Processes", expanded=True):
            st.dataframe(mem_df, use_container_width=True, hide_index=True)
            with open(export_csv(memory_processes[:10], ["PID", "Process Name", "Memory (MB)"], "top_memory_processes"), "rb") as f:
                st.download_button("⬇️ Download Top Memory Processes", f, file_name=f.name)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1)
        diagnostics_data.append(["CPU Usage (%)", cpu_percent])
        with st.expander("⚙️ CPU Load", expanded=True):
            st.metric(label="CPU Usage (%)", value=cpu_percent)

        # Uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        diagnostics_data.append(["System Boot Time", boot_time.strftime('%Y-%m-%d %H:%M:%S')])
        diagnostics_data.append(["Uptime", str(timedelta(seconds=uptime.total_seconds())).split('.')[0]])
        with st.expander("🕐 Uptime", expanded=True):
            st.write(f"System booted on: `{boot_time.strftime('%Y-%m-%d %H:%M:%S')}`")
            st.write(f"Uptime: `{str(timedelta(seconds=uptime.total_seconds())).split('.')[0]}`")

        # Network Connections
        net_stats = {}
        for conn in psutil.net_connections(kind='inet'):
            if conn.raddr and conn.status == 'ESTABLISHED':
                ip = conn.raddr.ip
                pid = conn.pid
                if pid:
                    try:
                        proc = psutil.Process(pid)
                        io_counters = proc.io_counters()
                        total_bytes = io_counters.write_bytes + io_counters.read_bytes
                        net_stats[ip] = net_stats.get(ip, 0) + total_bytes
                    except:
                        continue
        top_network = sorted(net_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        net_df = pd.DataFrame([(ip, round(total / (1024 * 1024), 2)) for ip, total in top_network], columns=["Remote IP", "Total MB"])
        diagnostics_data.append(["Top Network Connections", "See separate table"])
        with st.expander("🔌 Top Network Connections by Bandwidth", expanded=True):
            st.dataframe(net_df, use_container_width=True, hide_index=True)
            with open(export_csv(net_df.values.tolist(), ["Remote IP", "Total MB"], "top_network_connections"), "rb") as f:
                st.download_button("⬇️ Download Top Network Connections", f, file_name=f.name)

        # NIC Bandwidth
        net_io = psutil.net_io_counters(pernic=True)
        iface_data = [(iface, round(stats.bytes_sent / (1024*1024), 2), round(stats.bytes_recv / (1024*1024), 2)) for iface, stats in net_io.items()]
        iface_df = pd.DataFrame(iface_data, columns=["Interface", "Sent (MB)", "Recv (MB)"])
        diagnostics_data.append(["NIC Bandwidth", "See separate table"])
        with st.expander("🌐 Interface Bandwidth (per NIC)", expanded=True):
            st.dataframe(iface_df, use_container_width=True, hide_index=True)
            with open(export_csv(iface_df.values.tolist(), ["Interface", "Sent (MB)", "Recv (MB)"], "nic_bandwidth"), "rb") as f:
                st.download_button("⬇️ Download NIC Bandwidth", f, file_name=f.name)

        # Installed Software
        installed = []
        try:
            if os_type == "Windows":
                output = subprocess.check_output("wmic product get name,version", shell=True).decode(errors="ignore")
                for line in output.splitlines()[1:11]:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        name = " ".join(parts[:-1])
                        version = parts[-1]
                        installed.append((name, version))
                patch_output = subprocess.check_output("wmic qfe get HotFixID,InstalledOn", shell=True).decode(errors="ignore")
                for line in patch_output.splitlines()[1:6]:
                    parts = line.strip().split()
                    if parts:
                        installed.append((parts[0], " ".join(parts[1:])))
            elif os_type == "Linux":
                if shutil.which("dpkg"):
                    output = subprocess.check_output("dpkg-query -W -f='${Package} ${Version}\n' | head -n 10", shell=True).decode()
                    for line in output.strip().splitlines():
                        parts = line.split()
                        if len(parts) >= 2:
                            installed.append((parts[0], parts[1]))
                elif shutil.which("rpm"):
                    output = subprocess.check_output("rpm -qa --qf '%{NAME} %{VERSION}\n' | head -n 10", shell=True).decode()
                    for line in output.strip().splitlines():
                        parts = line.split()
                        if len(parts) >= 2:
                            installed.append((parts[0], parts[1]))
            elif os_type == "Darwin":
                output = subprocess.check_output("system_profiler SPApplicationsDataType | grep -E 'Location|Version' | head -n 20", shell=True).decode()
                app_name = None
                for line in output.splitlines():
                    line = line.strip()
                    if line.startswith("Location:"):
                        app_name = line.split(":", 1)[1].strip()
                    elif line.startswith("Version:") and app_name:
                        version = line.split(":", 1)[1].strip()
                        installed.append((app_name, version))
                        app_name = None
        except Exception as e:
            installed.append(("Error", str(e)))

        installed_df = pd.DataFrame(installed, columns=["Software / Patch", "Version / Date"])
        diagnostics_data.append(["Installed Software", "See separate table"])
        with st.expander("🧩 Installed Software & Patches", expanded=True):
            st.dataframe(installed_df, use_container_width=True, hide_index=True)
            with open(export_csv(installed, ["Software / Patch", "Version / Date"], "installed_software"), "rb") as f:
                st.download_button("⬇️ Download Installed Software List", f, file_name=f.name)

