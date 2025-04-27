# PI Dashboard

A terminal-based system monitoring dashboard written in Python using `curses` and `psutil`. Primarily designed for Raspberry Pi monitoring but should work on other Linux systems as well.

**(Optional: Add a Screenshot/GIF Here!)**
*It's highly recommended to add a screenshot or an animated GIF showing the dashboard in action. Replace the line below with your actual image.*
## Features

PI Dashboard provides a real-time overview of various system metrics across multiple tabs:

**Overview Tab (Tab 1):**
* **CPU:** Overall usage (meter, history graph), Per-core usage (meters), Current frequency, Core count (Physical/Logical), Temperature (uses `psutil.sensors_temperatures`, includes RPi-specific checks).
* **RAM:** Usage percentage (meter, history graph), Used/Total/Free amounts.
* **SWAP:** Usage percentage, Used/Total amounts.
* **Load Average:** 1, 5, and 15-minute system load averages.
* **GPU (Raspberry Pi):** GPU Temperature and Memory usage (requires `vcgencmd`).
* **System Info:** Hostname, OS Version, Kernel Version, Architecture.
* **Uptime:** System uptime since boot.

**Processes Tab (Tab 2):**
* Scrollable list of top processes.
* Sorted primarily by CPU usage, then Memory usage.
* Displays PID, CPU%, MEM%, User, and Process Name.

**Disks Tab (Tab 3):**
* Real-time I/O rates (Read/Write KB/s) per physical disk (attempts to exclude loop devices).
* List of mounted partitions (excluding certain types like squashfs, tmpfs).
* Displays Device, Total size, Used, Free, Usage Percentage, and Mountpoint for each partition.

**Network Tab (Tab 4):**
* Count of active `inet` connections.
* List of network interfaces with status (Up/Down).
* Real-time network I/O rates (Download/Upload KB/s) per interface.
* Displays IPv4, IPv6, and MAC addresses for each interface.

**General Features:**
* Tabbed interface for easy navigation.
* Color highlighting for Warning/Critical levels (CPU, RAM, Disk Usage, Temp, Load, SWAP).
* Pause/Resume functionality for updates.
* Basic terminal resize handling (clears screen).
* Optimized data fetching (details for Disks/Network tabs are fetched only when the tab is active).

## Requirements

* **Python 3.x**
* **`psutil` library:** Provides system information.
* **(Raspberry Pi - Optional):** `vcgencmd` utility for fetching GPU temperature and memory. This is usually pre-installed on Raspberry Pi OS.
* A terminal emulator that supports:
    * Colors (for highlighting).
    * Unicode (UTF-8 recommended for special characters used in meters/graphs).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/emphyri0/pi_dashboard
    cd pi-dashboard 
    ```
2.  **Install dependencies:**
    ```bash
    pip install psutil
    # or
    pip3 install psutil
    ```
    *Note: On some systems, you might need to install `python3-pip` first (`sudo apt update && sudo apt install python3-pip`).*
    *Note: The `curses` library is usually built-in with Python on Linux/macOS.*

## Usage

Run the script from your terminal:

```bash
python3 dashboard.py
