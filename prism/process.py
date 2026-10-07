"""Launch owned encoder processes without opening a Windows console."""
import subprocess
import sys


def popen_hidden(command, **kwargs):
    if sys.platform == "win32":
        # DETACHED_PROCESS and CREATE_NEW_CONSOLE defeat CREATE_NO_WINDOW.
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(command, **kwargs)
