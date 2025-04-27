# -*- coding: utf-8 -*-
import curses
import psutil
import time
import datetime
import os
from collections import deque
import locale
import socket
import platform
import subprocess

# --- Configuration / Thresholds for Highlighting ---
WARN_CPU_PERCENT = 60.0
CRIT_CPU_PERCENT = 80.0
WARN_MEM_PERCENT = 70.0
CRIT_MEM_PERCENT = 85.0
WARN_SWAP_PERCENT = 30.0
CRIT_SWAP_PERCENT = 60.0
WARN_DISK_PERCENT = 75.0
CRIT_DISK_PERCENT = 90.0
WARN_CPU_TEMP = 65.0
CRIT_CPU_TEMP = 75.0 # Adjust based on your Pi model and cooling
WARN_LOAD_MULT = 0.7 # Load avg > 70% of core count
CRIT_LOAD_MULT = 1.0 # Load avg > 100% of core count

# History for graphs
cpu_history = deque(maxlen=60)
mem_history = deque(maxlen=60)

# --- Data Fetching Functions ---
# (get_system_info, get_cpu_info, get_memory_info, get_cpu_temp, get_rpi_gpu_info remain the same)
# (get_disk_info, get_network_info, get_top_processes also remain structurally similar)
def get_system_info():
    """Fetch basic system information"""
    try:
        uname = platform.uname()
        return {
            "hostname": uname.node,
            "os": f"{uname.system} {uname.release}",
            "kernel": uname.version.split(" ")[0],
            "arch": uname.machine
        }
    except Exception:
        return {"hostname": "N/A", "os": "N/A", "kernel": "N/A", "arch": "N/A"}

def get_cpu_info():
    """Fetch CPU information including per-core usage"""
    cpu_percent_overall = psutil.cpu_percent(interval=None)
    cpu_history.append(cpu_percent_overall)
    try: cpu_percent_per_core = psutil.cpu_percent(interval=None, percpu=True)
    except Exception: cpu_percent_per_core = []
    cpu_freq = psutil.cpu_freq()
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    return {
        "percent": cpu_percent_overall,
        "per_core_percent": cpu_percent_per_core,
        "freq_current": cpu_freq.current if cpu_freq else 0,
        "freq_max": cpu_freq.max if cpu_freq and hasattr(cpu_freq, 'max') else 0,
        "cores_physical": cpu_count_physical,
        "cores_logical": cpu_count_logical
    }

def get_memory_info():
    """Fetch Memory information"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    mem_history.append(mem.percent)
    return {
        "mem_percent": mem.percent,
        "mem_total_gb": round(mem.total / (1024**3), 1),
        "mem_used_gb": round(mem.used / (1024**3), 1),
        "mem_free_gb": round(mem.available / (1024**3), 1),
        "swap_percent": swap.percent,
        "swap_total_gb": round(swap.total / (1024**3), 1),
        "swap_used_gb": round(swap.used / (1024**3), 1)
    }

def get_cpu_temp():
    """Fetch CPU temperature"""
    if not hasattr(psutil, "sensors_temperatures"): return "N/A"
    try: temps = psutil.sensors_temperatures()
    except Exception: return "N/A"
    if not temps: return "N/A"
    cpu_temp_keys = ['cpu_thermal', 'coretemp', 'k10temp', 'zenpower']
    for key in cpu_temp_keys:
        if key in temps:
            for entry in temps[key]:
                label_lower = entry.label.lower()
                if entry.label == '' or 'package' in label_lower or 'cpu' in label_lower or 'tdie' in label_lower:
                    if isinstance(entry.current, (int, float)): return f"{entry.current:.1f}"
                    else: continue
            if temps[key]:
                 first_valid = next((e for e in temps[key] if isinstance(e.current, (int, float))), None)
                 if first_valid: return f"{first_valid.current:.1f}"
    for key in temps:
        if temps[key]:
             first_valid = next((e for e in temps[key] if isinstance(e.current, (int, float))), None)
             if first_valid: return f"{first_valid.current:.1f}"
    return "N/A"

def get_rpi_gpu_info():
    """Fetch Raspberry Pi GPU temperature and memory using vcgencmd"""
    gpu_info = {"temp": "N/A", "mem": "N/A"}
    try:
        vcgencmd_path = subprocess.check_output(['which', 'vcgencmd']).strip().decode('utf-8')
        if not vcgencmd_path: return gpu_info
        temp_output = subprocess.check_output([vcgencmd_path, 'measure_temp'], timeout=1).decode('utf-8')
        temp_val = temp_output.split('=')[1].split("'")[0]
        gpu_info["temp"] = f"{float(temp_val):.1f}"
        mem_output = subprocess.check_output([vcgencmd_path, 'get_mem', 'gpu'], timeout=1).decode('utf-8')
        mem_val = mem_output.split('=')[1].strip()
        gpu_info["mem"] = mem_val
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, ValueError, OSError):
        pass
    return gpu_info

def get_disk_partitions():
    """Fetch Disk partition information (partitions only)"""
    partitions = []
    ignore_fstypes = ['squashfs', 'tmpfs', 'devtmpfs', 'iso9660']
    ignore_opts = ['cdrom']
    for partition in psutil.disk_partitions(all=False):
        if partition.fstype in ignore_fstypes or any(opt in partition.opts for opt in ignore_opts):
             continue
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            partitions.append({
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "fstype": partition.fstype,
                "total_gb": round(usage.total / (1024**3), 1),
                "used_gb": round(usage.used / (1024**3), 1),
                "free_gb": round(usage.free / (1024**3), 1),
                "percent": usage.percent
            })
        except (PermissionError, FileNotFoundError, OSError):
            continue
    return partitions

def get_network_connection_count():
    """Fetch Network connection count"""
    try: net_connections = len(psutil.net_connections(kind='inet'))
    except psutil.AccessDenied: net_connections = "N/A"
    except Exception: net_connections = "Error"
    return net_connections

def get_top_processes(n=10):
    """Get top processes by CPU usage"""
    processes = []
    attrs = ['pid', 'name', 'username', 'cpu_percent', 'memory_percent']
    for proc in psutil.process_iter(attrs=attrs, ad_value=None):
        try:
            pinfo = proc.info
            cpu_p = pinfo.get('cpu_percent')
            mem_p = pinfo.get('memory_percent')
            if pinfo.get('username') is not None and \
               ((cpu_p is not None and cpu_p > 0.0) or \
                (mem_p is not None and mem_p > 0.0)):
                 processes.append(pinfo)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, Exception): pass
    try:
        key_func = lambda x: (float(x.get('cpu_percent', 0) or 0), float(x.get('memory_percent', 0) or 0))
        return sorted(processes, key=key_func, reverse=True)[:n]
    except (TypeError, ValueError): return []

# --- Drawing Functions ---
def draw_meter(stdscr, y, x, value, width=20, title="", warn_threshold=70.0, crit_threshold=85.0):
    """Draws a horizontal meter bar with percentage and highlighting"""
    max_y_win, max_x_win = stdscr.getmaxyx()
    has_colors = curses.has_colors()

    if not isinstance(value, (int, float)): value = 0
    value = max(0.0, min(100.0, value))
    filled_width = int(width * value / 100)

    # Determine color and attribute based on thresholds
    color_pair_idx = 1 # Default Green
    attribute = curses.A_NORMAL
    if value >= crit_threshold:
        color_pair_idx = 3 # Red
        attribute = curses.A_BOLD # Add bold for critical
    elif value >= warn_threshold:
        color_pair_idx = 2 # Yellow

    color_pair = curses.color_pair(color_pair_idx) if has_colors else 0

    # Draw title (safely, check bounds)
    title_str = f"{title}: {value:.1f}%"
    try:
        if y < max_y_win and x + len(title_str) < max_x_win:
             # Apply attribute to title if critical
             if attribute != curses.A_NORMAL: stdscr.attron(attribute | color_pair)
             stdscr.addstr(y, x, title_str)
             if attribute != curses.A_NORMAL: stdscr.attroff(attribute | color_pair)
    except curses.error: pass

    # Draw meter bar (safely, check bounds)
    meter_y = y + 1
    if meter_y < max_y_win:
        try:
            if x < max_x_win: stdscr.addch(meter_y, x, "[")
            if x + width + 1 < max_x_win: stdscr.addch(meter_y, x + width + 1, "]")
            fill_char = '■'; empty_char = ' '
            bar_x_start = x + 1
            for i in range(width):
                 current_x = bar_x_start + i
                 if current_x < max_x_win:
                     if i < filled_width:
                         stdscr.attron(color_pair | attribute) # Apply attribute here too
                         try: stdscr.addch(meter_y, current_x, fill_char)
                         except: stdscr.addch(meter_y, current_x, '#')
                         stdscr.attroff(color_pair | attribute)
                     else:
                         stdscr.addch(meter_y, current_x, empty_char)
                 else: break
        except curses.error: pass

def draw_sparkline(stdscr, y, x, data, width=60, height=5, title="", warn_threshold=70.0, crit_threshold=85.0):
    """Draws a simple sparkline graph with highlighting"""
    max_y_win, max_x_win = stdscr.getmaxyx()
    has_colors = curses.has_colors()
    label_width = 7

    if y + height >= max_y_win or x + width + label_width >= max_x_win or height < 2:
         try:
             if y < max_y_win and x < max_x_win: stdscr.addstr(y, x, f"{title}: Window too small")
         except curses.error: pass
         return
    if not data or len(data) == 0:
         try:
             if y < max_y_win and x < max_x_win: stdscr.addstr(y, x, f"{title}: No data")
         except curses.error: pass
         return
    try:
        if y < max_y_win and x + len(title) < max_x_win: stdscr.addstr(y, x, title)
    except curses.error: pass

    numeric_data = [d for d in list(data) if isinstance(d, (int, float))]
    if not numeric_data: return

    max_val = max(numeric_data) if numeric_data else 100.0
    min_val = min(numeric_data) if numeric_data else 0.0
    if max_val == min_val: max_val += 1.0
    range_val = max(1.0, max_val - min_val)
    scale = (height - 1) / range_val
    label_x = x + label_width - 1
    try:
        if y + 1 < max_y_win and label_x < max_x_win: stdscr.addstr(y + 1, x, f"{max_val:3.0f}%".rjust(label_width-1))
        if y + height < max_y_win and label_x < max_x_win: stdscr.addstr(y + height, x, f"{min_val:3.0f}%".rjust(label_width-1))
    except curses.error: pass

    display_data = numeric_data[-width:]
    graph_x_start = x + label_width
    for i, val in enumerate(display_data):
        current_graph_x = graph_x_start + i
        if current_graph_x >= max_x_win: break
        scaled_val = (val - min_val) * scale
        bar_height = max(0, min(height - 1, int(scaled_val)))
        pos_y = y + height - bar_height

        # Determine color and attribute based on value
        color_pair_idx = 1; attribute = curses.A_NORMAL
        if val >= crit_threshold: color_pair_idx = 3; attribute = curses.A_BOLD
        elif val >= warn_threshold: color_pair_idx = 2
        color_pair = curses.color_pair(color_pair_idx) if has_colors else 0

        if pos_y >= 0 and pos_y < max_y_win:
             try:
                 stdscr.attron(color_pair | attribute)
                 try: stdscr.addch(pos_y, current_graph_x, '•')
                 except: stdscr.addch(pos_y, current_graph_x, '*')
                 stdscr.attroff(color_pair | attribute)
             except curses.error: pass


# --- Main Application Function ---
def main(stdscr):
    # Curses settings
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(1000) # Update interval

    # Color setup
    has_colors = curses.has_colors()
    if has_colors:
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)   # Normal
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Warning
            curses.init_pair(3, curses.COLOR_RED, -1)     # Critical
            curses.init_pair(4, curses.COLOR_CYAN, -1)    # Headers
            curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE) # Selected Tab
            # Optional: Add more pairs if needed for highlighting, e.g., reversed colors
            # curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_RED) # Critical Background
        except curses.error: has_colors = False
    def safe_attron(attr):
        if has_colors: stdscr.attron(attr)
    def safe_attroff(attr):
        if has_colors: stdscr.attroff(attr)
    def safe_color_pair(pair_num):
         return curses.color_pair(pair_num) if has_colors else 0

    # --- State variables ---
    last_time = time.time()
    # For rate calculations - Initialize outside loop
    last_net_io_pernic = psutil.net_io_counters(pernic=True)
    last_disk_io_perdisk = psutil.disk_io_counters(perdisk=True)
    net_speeds = {}
    disk_speeds = {}
    # Cached data for inactive tabs
    disk_partitions_cache = []
    net_connection_count_cache = "N/A"
    processes_data_cache = []

    system_info = get_system_info() # Static info
    logical_core_count = psutil.cpu_count(logical=True) # For load avg threshold

    active_tab = 0
    process_scroll = 0
    tab_names = ["Overview (1)", "Processes (2)", "Disks (3)", "Network (4)"]
    max_tabs = len(tab_names)
    is_paused = False # Pause state

    # --- Main Loop ---
    while True:
        max_y, max_x = stdscr.getmaxyx()
        try: key = stdscr.getch()
        except: key = -1

        # --- Input Handling ---
        if key == ord('q'): break
        elif key == ord(' '): is_paused = not is_paused # Toggle pause state
        elif key == ord('1'): active_tab = 0
        elif key == ord('2'): active_tab = 1; process_scroll = 0
        elif key == ord('3'): active_tab = 2
        elif key == ord('4'): active_tab = 3
        elif key == curses.KEY_RESIZE:
             # Basic resize handling: clear screen and reset scroll
             stdscr.erase()
             process_scroll = 0
             # A better implementation would recalculate layout here
        elif key == ord('\t') or key == curses.KEY_RIGHT:
            active_tab = (active_tab + 1) % max_tabs
            if active_tab == 1: process_scroll = 0
        elif key == curses.KEY_LEFT:
             active_tab = (active_tab - 1 + max_tabs) % max_tabs
             if active_tab == 1: process_scroll = 0
        elif active_tab == 1: # Process scrolling (only if not paused)
             if not is_paused:
                 # (Scrolling logic remains the same)
                 visible_process_lines = max(1, max_y - 8)
                 num_processes = len(processes_data_cache) # Use cached data length
                 if key == curses.KEY_DOWN:
                      if num_processes > visible_process_lines and process_scroll < (num_processes - visible_process_lines):
                           process_scroll += 1
                 elif key == curses.KEY_UP:
                      process_scroll = max(0, process_scroll - 1)
                 elif key == curses.KEY_NPAGE:
                      process_scroll = min(num_processes - visible_process_lines, process_scroll + visible_process_lines); process_scroll = max(0, process_scroll)
                 elif key == curses.KEY_PPAGE:
                      process_scroll = max(0, process_scroll - visible_process_lines)

        # --- Data Fetching & Calculations (Skip if paused) ---
        if not is_paused:
            current_time = time.time()
            time_diff = max(0.1, current_time - last_time)

            # Always fetch overview data
            cpu_info = get_cpu_info()
            mem_info = get_memory_info()
            cpu_temp = get_cpu_temp()
            gpu_info = get_rpi_gpu_info()
            load_avg = psutil.getloadavg()

            # Fetch data only for the active tab (Optimization)
            if active_tab == 1:
                processes_data_cache = get_top_processes(n=max(20, max_y * 2))
            elif active_tab == 2:
                disk_partitions_cache = get_disk_partitions()
                # Calculate Per-Disk I/O
                current_disk_io_perdisk = psutil.disk_io_counters(perdisk=True)
                disk_speeds.clear()
                for disk, current_io in current_disk_io_perdisk.items():
                    if disk.startswith('loop'): continue
                    if disk in last_disk_io_perdisk:
                        last_io = last_disk_io_perdisk[disk]
                        read_bytes_diff = max(0.0, (current_io.read_bytes or 0) - (last_io.read_bytes or 0))
                        write_bytes_diff = max(0.0, (current_io.write_bytes or 0) - (last_io.write_bytes or 0))
                        disk_speeds[disk] = {'read': read_bytes_diff / time_diff / 1024, 'write': write_bytes_diff / time_diff / 1024}
                    else: disk_speeds[disk] = {'read': 0.0, 'write': 0.0}
                last_disk_io_perdisk = current_disk_io_perdisk
            elif active_tab == 3:
                net_connection_count_cache = get_network_connection_count()
                # Calculate Per-Interface Network Speed
                current_net_io_pernic = psutil.net_io_counters(pernic=True)
                net_speeds.clear()
                for iface, current_io in current_net_io_pernic.items():
                    if iface in last_net_io_pernic:
                        last_io = last_net_io_pernic[iface]
                        net_speeds[iface] = {
                            'down': max(0.0, (current_io.bytes_recv - last_io.bytes_recv)) / time_diff / 1024,
                            'up': max(0.0, (current_io.bytes_sent - last_io.bytes_sent)) / time_diff / 1024
                        }
                    else: net_speeds[iface] = {'down': 0.0, 'up': 0.0}
                last_net_io_pernic = current_net_io_pernic

            last_time = current_time # Update time only when not paused

        # --- Drawing ---
        stdscr.erase()
        if max_y < 10 or max_x < 60:
            try: stdscr.addstr(0, 0, "Window too small...")
            except curses.error: pass
            stdscr.refresh(); continue

        # Draw Header (Title, Uptime, Pause Indicator)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = f"--- PI Dashboard ({now_str}) ---"
        pause_indicator = " [PAUSED]" if is_paused else ""
        try:
             boot_time_sec = psutil.boot_time(); uptime_delta = datetime.timedelta(seconds=int(time.time() - boot_time_sec)); uptime_str = f"Uptime: {str(uptime_delta).split('.')[0]}"
        except Exception: uptime_str = "Uptime: N/A"
        try:
            safe_attron(curses.A_BOLD)
            stdscr.addstr(0, max(0, (max_x - len(header_text) - len(pause_indicator)) // 2), header_text)
            if is_paused: stdscr.addstr(0, max(0, (max_x + len(header_text) - len(pause_indicator)) // 2), pause_indicator, safe_color_pair(2) | curses.A_BOLD) # Yellow Bold for Paused
            safe_attroff(curses.A_BOLD)
            stdscr.addstr(1, max(0, max_x - len(uptime_str) - 1), uptime_str)
        except curses.error: pass

        # Draw Tabs (code unchanged)
        tab_y = 2; current_tab_x = 2
        for i, name in enumerate(tab_names):
            display_name = f" {name} "
            try:
                if current_tab_x + len(display_name) >= max_x: break
                if i == active_tab: safe_attron(safe_color_pair(5) | curses.A_BOLD); stdscr.addstr(tab_y, current_tab_x, display_name); safe_attroff(safe_color_pair(5) | curses.A_BOLD)
                else: stdscr.addstr(tab_y, current_tab_x, display_name)
                current_tab_x += len(display_name) + 1
            except curses.error: break

        # --- Draw Content Area ---
        content_y_start = 4
        try:
            # --- Overview Tab ---
            if active_tab == 0:
                row_y = content_y_start
                col1_x = 2
                col2_x = max(col1_x + 35, max_x // 2)

                # Col 1: CPU / RAM / Per-Core
                meter_width = min(30, max(10, (col2_x - col1_x) - 5))
                draw_meter(stdscr, row_y, col1_x, cpu_info["percent"], width=meter_width, title="CPU Overall", warn_threshold=WARN_CPU_PERCENT, crit_threshold=CRIT_CPU_PERCENT)
                row_y += 3
                draw_meter(stdscr, row_y, col1_x, mem_info["mem_percent"], width=meter_width, title="RAM", warn_threshold=WARN_MEM_PERCENT, crit_threshold=CRIT_MEM_PERCENT)
                row_y += 3
                max_cores_to_show = max(1, max_y - row_y - 2)
                stdscr.addstr(row_y, col1_x, "CPU Core Usage:")
                row_y += 1
                core_meter_width = min(15, max(5, meter_width // 2))
                per_core_pct = cpu_info.get("per_core_percent", [])
                for i, core_pct in enumerate(per_core_pct):
                    if i >= max_cores_to_show:
                         if i == max_cores_to_show: stdscr.addstr(row_y, col1_x + 1, f"  ({len(per_core_pct) - i} more...)")
                         break
                    # Use CPU thresholds for core meters
                    draw_meter(stdscr, row_y, col1_x + 1, core_pct, width=core_meter_width, title=f" Core {i}", warn_threshold=WARN_CPU_PERCENT, crit_threshold=CRIT_CPU_PERCENT)
                    row_y += 2
                    if row_y >= max_y -1: break

                # Col 2: Details
                if col2_x < max_x - 25:
                    row_y_col2 = content_y_start
                    # Load Average Highlighting
                    load_str = "Load Avg: " + " ".join([f"{l:.2f}" for l in load_avg])
                    load_attr = curses.A_NORMAL
                    if logical_core_count > 0:
                        if load_avg[0] > logical_core_count * CRIT_LOAD_MULT: load_attr = safe_color_pair(3) | curses.A_BOLD # Red Bold
                        elif load_avg[0] > logical_core_count * WARN_LOAD_MULT: load_attr = safe_color_pair(2) # Yellow
                    safe_attron(load_attr); stdscr.addstr(row_y_col2, col2_x, load_str); safe_attroff(load_attr); row_y_col2 += 1

                    stdscr.addstr(row_y_col2, col2_x, f"CPU Freq: {cpu_info.get('freq_current', 0):.0f} MHz"); row_y_col2 += 1
                    # CPU Temp Highlighting
                    temp_val_str = cpu_temp.replace("°C","") if isinstance(cpu_temp, str) else "N/A"
                    temp_attr = curses.A_NORMAL
                    if temp_val_str != "N/A":
                         try:
                             temp_f = float(temp_val_str)
                             if temp_f >= CRIT_CPU_TEMP: temp_attr = safe_color_pair(3) | curses.A_BOLD
                             elif temp_f >= WARN_CPU_TEMP: temp_attr = safe_color_pair(2)
                         except ValueError: pass # Ignore if conversion fails
                    temp_str = f"CPU Temp: {cpu_temp}°C" if cpu_temp != "N/A" else "CPU Temp: N/A"
                    safe_attron(temp_attr); stdscr.addstr(row_y_col2, col2_x, temp_str); safe_attroff(temp_attr); row_y_col2 += 1

                    # GPU Info (No specific highlighting added here, could be done similarly)
                    gpu_temp_str = f"GPU Temp: {gpu_info['temp']}°C" if gpu_info['temp'] != "N/A" else "GPU Temp: N/A"
                    stdscr.addstr(row_y_col2, col2_x, gpu_temp_str); row_y_col2 += 1
                    gpu_mem_str = f"GPU Mem:  {gpu_info['mem']}" if gpu_info['mem'] != "N/A" else "GPU Mem: N/A"
                    stdscr.addstr(row_y_col2, col_x, gpu_mem_str); row_y_col2 += 2 # Fixed position from col_x to col2_x

                    stdscr.addstr(row_y_col2, col2_x, "RAM Details:"); row_y_col2 += 1
                    stdscr.addstr(row_y_col2, col2_x+1, f"Used: {mem_info.get('mem_used_gb', 0):.1f} / {mem_info.get('mem_total_gb', 0):.1f} GB"); row_y_col2 += 1
                    stdscr.addstr(row_y_col2, col2_x+1, f"Free: {mem_info.get('mem_free_gb', 0):.1f} GB"); row_y_col2 += 1
                    # SWAP Highlighting
                    swap_pct = mem_info.get('swap_percent', 0) or 0
                    swap_attr = curses.A_NORMAL
                    if swap_pct >= CRIT_SWAP_PERCENT: swap_attr = safe_color_pair(3) | curses.A_BOLD
                    elif swap_pct >= WARN_SWAP_PERCENT: swap_attr = safe_color_pair(2)
                    safe_attron(swap_attr); stdscr.addstr(row_y_col2, col2_x+1, f"SWAP: {mem_info.get('swap_used_gb', 0):.1f}/{mem_info.get('swap_total_gb', 0):.1f} GB ({swap_pct}%)"); safe_attroff(swap_attr); row_y_col2 += 2

                    stdscr.addstr(row_y_col2, col2_x, "System Info:"); row_y_col2 += 1
                    stdscr.addstr(row_y_col2, col2_x+1, f"Host: {system_info.get('hostname', 'N/A')}"); row_y_col2 += 1
                    stdscr.addstr(row_y_col2, col2_x+1, f"OS:   {system_info.get('os', 'N/A')}"); row_y_col2 += 1
                    kernel_str = system_info.get('kernel', 'N/A'); max_kernel_len = max_x - col2_x - 8; kernel_disp = kernel_str[:max_kernel_len] + ('...' if len(kernel_str) > max_kernel_len else '')
                    stdscr.addstr(row_y_col2, col2_x+1, f"Kernel: {kernel_disp}"); row_y_col2 += 1
                    stdscr.addstr(row_y_col2, col2_x+1, f"Arch: {system_info.get('arch', 'N/A')}"); row_y_col2 += 1

            # --- Processes Tab ---
            elif active_tab == 1:
                # Use cached process data for drawing
                processes_to_draw = processes_data_cache
                # (Drawing logic remains the same, now uses processes_to_draw)
                pid_w, cpu_w, mem_w, user_w = 8, 8, 8, 12
                name_w = max(15, max_x - pid_w - cpu_w - mem_w - user_w - 6)
                header_y = content_y_start
                safe_attron(safe_color_pair(4) | curses.A_BOLD); col_x = 2
                stdscr.addstr(header_y, col_x, "PID".ljust(pid_w)); col_x += pid_w; stdscr.addstr(header_y, col_x, "CPU%".ljust(cpu_w)); col_x += cpu_w
                stdscr.addstr(header_y, col_x, "MEM%".ljust(mem_w)); col_x += mem_w; stdscr.addstr(header_y, col_x, "User".ljust(user_w)); col_x += user_w
                stdscr.addstr(header_y, col_x, "Process Name".ljust(name_w)); safe_attroff(safe_color_pair(4) | curses.A_BOLD)
                visible_lines = max(1, max_y - header_y - 3); start_idx = process_scroll; end_idx = start_idx + visible_lines
                for i, proc in enumerate(processes_to_draw[start_idx:end_idx]):
                    row_y = header_y + 1 + i
                    if row_y >= max_y - 1: break
                    cpu_pct = float(proc.get('cpu_percent', 0) or 0)
                    if cpu_pct >= CRIT_CPU_PERCENT: color_idx = 3; attr = curses.A_BOLD
                    elif cpu_pct >= WARN_CPU_PERCENT: color_idx = 2; attr = curses.A_NORMAL
                    else: color_idx = 1; attr = curses.A_NORMAL
                    row_color = safe_color_pair(color_idx) | attr
                    col_x = 2; stdscr.addstr(row_y, col_x, str(proc.get('pid', '-')).ljust(pid_w)[:pid_w]); col_x += pid_w
                    safe_attron(row_color); stdscr.addstr(row_y, col_x, f"{cpu_pct:6.1f}%".ljust(cpu_w)[:cpu_w]); safe_attroff(row_color); col_x += cpu_w # Highlight CPU%
                    mem_pct = float(proc.get('memory_percent', 0) or 0)
                    stdscr.addstr(row_y, col_x, f"{mem_pct:6.1f}%".ljust(mem_w)[:mem_w]); col_x += mem_w # Can highlight mem too if needed
                    user = str(proc.get('username', '-'))[:user_w]; stdscr.addstr(row_y, col_x, user.ljust(user_w)); col_x += user_w
                    name = str(proc.get('name', '-'))[:name_w]; stdscr.addstr(row_y, col_x, name.ljust(name_w))
                if len(processes_to_draw) > visible_lines:
                     current_pos = start_idx + min(visible_lines, len(processes_to_draw) - start_idx)
                     scroll_perc = int(100 * current_pos / len(processes_to_draw)) if len(processes_to_draw) > 0 else 0
                     scroll_text = f"Scroll: {scroll_perc}%"
                     try:
                          if max_y-2 >= 0: stdscr.addstr(max_y - 2, max(2, max_x - len(scroll_text) - 2), scroll_text)
                     except: pass

            # --- Disks Tab ---
            elif active_tab == 2:
                # Use cached partition data for drawing
                partitions_to_draw = disk_partitions_cache
                row_y = content_y_start
                stdscr.addstr(row_y, 2, "Physical Disk I/O (KB/s):"); row_y += 1
                disk_io_lines = 0
                # Use current disk_speeds (calculated if tab active, otherwise stale)
                for disk, speeds in disk_speeds.items():
                    if row_y >= max_y - 1: break
                    read_kb = speeds.get('read', 0.0); write_kb = speeds.get('write', 0.0)
                    stdscr.addstr(row_y, 4, f"{disk:<10} R: {read_kb: >6.1f}, W: {write_kb: >6.1f}")
                    row_y += 1; disk_io_lines += 1
                if disk_io_lines == 0: stdscr.addstr(row_y, 4, "No disk I/O detected or supported."); row_y += 1
                row_y += 1

                header_y = row_y
                if header_y >= max_y -2: stdscr.addstr(header_y, 2, "Not enough space for partitions.")
                else:
                    dev_w, tot_w, use_w, free_w, perc_w = 16, 10, 10, 10, 9; mount_w = max(15, max_x - dev_w - tot_w - use_w - free_w - perc_w - 8)
                    safe_attron(safe_color_pair(4) | curses.A_BOLD); col_x = 2
                    stdscr.addstr(header_y, col_x, "Device".ljust(dev_w)); col_x += dev_w; stdscr.addstr(header_y, col_x, "Total".ljust(tot_w)); col_x += tot_w
                    stdscr.addstr(header_y, col_x, "Used".ljust(use_w)); col_x += use_w; stdscr.addstr(header_y, col_x, "Free".ljust(free_w)); col_x += free_w
                    stdscr.addstr(header_y, col_x, "Usage%".ljust(perc_w)); col_x += perc_w; stdscr.addstr(header_y, col_x, "Mountpoint".ljust(mount_w)); safe_attroff(safe_color_pair(4) | curses.A_BOLD)
                    for i, part in enumerate(partitions_to_draw): # Use cached data
                        row_y = header_y + 1 + i
                        if row_y >= max_y - 1: break
                        percent = float(part.get("percent", 0) or 0)
                        # Highlight Disk Usage %
                        attr = curses.A_NORMAL; color_idx = 1
                        if percent >= CRIT_DISK_PERCENT: color_idx = 3; attr = curses.A_BOLD
                        elif percent >= WARN_DISK_PERCENT: color_idx = 2
                        row_color = safe_color_pair(color_idx)
                        col_x = 2; stdscr.addstr(row_y, col_x, str(part.get("device", "-"))[:dev_w].ljust(dev_w)); col_x += dev_w
                        stdscr.addstr(row_y, col_x, f"{part.get('total_gb','?'):.1f}G".ljust(tot_w)[:tot_w]); col_x += tot_w
                        stdscr.addstr(row_y, col_x, f"{part.get('used_gb','?'):.1f}G".ljust(use_w)[:use_w]); col_x += use_w
                        stdscr.addstr(row_y, col_x, f"{part.get('free_gb','?'):.1f}G".ljust(free_w)[:free_w]); col_x += free_w
                        safe_attron(row_color | attr); stdscr.addstr(row_y, col_x, f"{percent:.1f}%".ljust(perc_w)[:perc_w]); safe_attroff(row_color | attr); col_x += perc_w # Apply highlight
                        mount = str(part.get("mountpoint", "-"))[:mount_w]; stdscr.addstr(row_y, col_x, mount.ljust(mount_w))

            # --- Network Tab ---
            elif active_tab == 3:
                # Use cached connection count
                net_conn_count = net_connection_count_cache
                row_y = content_y_start
                stdscr.addstr(row_y, 2, f"Active Connections: {net_conn_count}"); row_y += 2
                header_y = row_y
                safe_attron(safe_color_pair(4) | curses.A_BOLD); stdscr.addstr(header_y, 2, "Network Interfaces (Rates in KB/s):"); safe_attroff(safe_color_pair(4) | curses.A_BOLD); row_y +=1
                interface_row_y = row_y
                try:
                    addresses = psutil.net_if_addrs()
                    stats = psutil.net_if_stats()
                    for interface, addrs in addresses.items():
                        if interface_row_y >= max_y - 1: break
                        status_symbol = " ?" ; color_idx = 0
                        if interface in stats:
                            if stats[interface].isup: status_symbol = " ▲"; color_idx = 1
                            else: status_symbol = " ▼"; color_idx = 3
                        status_color = safe_color_pair(color_idx)
                        # Use current net_speeds (calculated if tab active, otherwise stale)
                        if_speeds = net_speeds.get(interface, {'down': 0.0, 'up': 0.0})
                        down_kb = if_speeds['down']; up_kb = if_speeds['up']
                        safe_attron(status_color); stdscr.addstr(interface_row_y, 2, f"{status_symbol} {interface}"); safe_attroff(status_color)
                        speed_str = f" D: {down_kb: >6.1f}, U: {up_kb: >6.1f}"
                        speed_x = max(20, max_x - len(speed_str) - 2)
                        stdscr.addstr(interface_row_y, speed_x, speed_str)
                        addr_indent_x = 4; addr_count = 0
                        for addr in addrs:
                             addr_str = None; addr_type = ""
                             if addr.family == socket.AF_INET: addr_type = "IPv4:"; addr_str = addr.address
                             elif addr.family == socket.AF_INET6:
                                 addr_type = "IPv6:"; addr_str = str(addr.address)
                                 if '%' in addr_str: addr_str = addr_str.split('%')[0]
                                 if len(addr_str) > max_x - addr_indent_x - 10: addr_str = addr_str[:max_x - addr_indent_x - 13] + "..."
                             elif hasattr(socket, 'AF_LINK') and addr.family == socket.AF_LINK: addr_type = "MAC: "; addr_str = addr.address
                             if addr_str:
                                 current_addr_row = interface_row_y + 1 + addr_count
                                 if current_addr_row >= max_y - 1: break
                                 try:
                                     full_addr_line = f"{addr_type.ljust(5)} {addr_str}"
                                     stdscr.addstr(current_addr_row, addr_indent_x, full_addr_line[:max_x - addr_indent_x - 1])
                                     addr_count += 1
                                 except curses.error: pass
                        interface_row_y += (addr_count + 2)
                except Exception as e:
                    if interface_row_y < max_y - 1:
                        error_str = f"Error fetching interfaces: {str(e)}"
                        stdscr.addstr(interface_row_y, 2, error_str[:max_x - 3])

        except Exception as e:
             try:
                  error_y = max_y - 2
                  if error_y >= 0: stdscr.addstr(error_y, 2, f"Draw Error: {str(e)}"[:max_x - 3])
             except curses.error: pass

        # Draw Help Footer Line
        try:
            help_text = "[q] Exit | [Spc] Pause | [Tab/←→] Tabs | [↑↓ PgUp/PgDn] Scroll" # Updated help text
            help_y = max_y - 1
            if help_y >= 0: stdscr.addstr(help_y, 2, help_text[:max_x - 3])
        except curses.error: pass

        stdscr.refresh()

# --- Main execution ---
if __name__ == "__main__":
    try: locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try: locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            try: locale.setlocale(locale.LC_ALL, '')
            except locale.Error: pass
    curses.wrapper(main)
