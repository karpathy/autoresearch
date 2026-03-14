"""Prevent Windows from sleeping while the agent runs.
Uses SetThreadExecutionState to tell Windows the system is in use.
Run in a separate terminal: python scripts/keep_awake.py
"""
import ctypes
import time
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

def keep_awake():
    print("Keeping PC awake (Ctrl+C to stop)...")
    try:
        while True:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
            time.sleep(30)
    except KeyboardInterrupt:
        # Reset to default
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("\nSleep prevention disabled.")

if __name__ == "__main__":
    keep_awake()
