"""Entry point for generating individual panels of Figure 2.

This script does not assemble the final multi-panel figure. The final figure was arranged in a graphics editor (Inkscape) to best position panels before manuscript submission. This script will generate the individual panels, which can then be manually arranged in a graphics editor to create the final figure. The script will run each of the individual panel generation scripts in sequence, and print a message when all panels have been generated. 
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PANEL_SCRIPTS = [
    "generate_figure_2_panel_a.py",
    "generate_figure_2_panel_b-e.py",
    "generate_figure_2_panel_f.py",
]


def run_panel_script(script_name):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Panel script not found: {script_path}")
    subprocess.check_call([sys.executable, script_path])


def main():
    for script in PANEL_SCRIPTS:
        print(f"Generating panel from {script}...")
        run_panel_script(script)
    print("All requested Figure 2 panel scripts completed.")


if __name__ == "__main__":
    main()
