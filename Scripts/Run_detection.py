"""
run_detection.py
================
Launcher script for the YOLOv7 real-time hand detection module.

Calls real_time_detection.py via subprocess with the configured parameters.
Adjust the paths and thresholds in the Configuration section before running.

Usage
-----
    python run_detection.py
"""

import os
import subprocess


# ── Configuration ──────────────────────────────────────────────────────────────
YOLOV7_DIR   = 'C:/Users/yolov7'          # Path to YOLOv7 root directory
WEIGHTS_PATH = 'C:/Users/best.pt'         # Path to trained YOLOv7 weights
OUTPUT_DIR   = 'C:/Users/Ejecucion'       # Directory to save detection results

CONF_THRESHOLD = 0.8    # Minimum confidence score to accept a detection
SOURCE         = '0'    # '0' = webcam | path to image/video file

SAVE_TXT  = True        # Save detection labels to .txt files
SAVE_CONF = True        # Include confidence scores in saved labels
VIEW_IMG  = True        # Display real-time detection window


# ── Launcher ───────────────────────────────────────────────────────────────────

def build_command():
    """Build the subprocess command list from configuration."""
    cmd = [
        'python', 'real_time_detection.py',
        '--weights', WEIGHTS_PATH,
        '--conf',    str(CONF_THRESHOLD),
        '--source',  SOURCE,
        '--name',    OUTPUT_DIR,
    ]
    if SAVE_TXT:
        cmd.append('--save-txt')
    if SAVE_CONF:
        cmd.append('--save-conf')
    if VIEW_IMG:
        cmd.append('--view-img')
    return cmd


def run():
    """Change to the YOLOv7 directory and launch the detection script."""
    if not os.path.isdir(YOLOV7_DIR):
        raise FileNotFoundError(f"YOLOv7 directory not found: {YOLOV7_DIR}")

    os.chdir(YOLOV7_DIR)
    command = build_command()

    print("Starting detection with command:")
    print("  " + ' '.join(command))
    print()

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Detection completed successfully.")
        print(result.stdout)
    else:
        print("Detection failed with errors:")
        print(result.stderr)


if __name__ == '__main__':
    run()
