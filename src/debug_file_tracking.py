"""Debug utility to track file open/close operations.

Import and call enable_file_tracking() at the start of your script to log all file operations.
"""

import sys
import builtins
import io


_original_open = None
_open_files = {}
_tracking_enabled = False


def _debug_open(*args, **kwargs):
    """Wrapper around open() that logs file path and memory address."""
    file_obj = _original_open(*args, **kwargs)
    file_path = args[0] if args else kwargs.get('file', 'unknown')
    addr = hex(id(file_obj))
    _open_files[addr] = file_path

    # Try to write to stderr, but don't fail if it's redirected/closed
    try:
        print(f"[FILE_DEBUG] OPEN: {file_path} -> {addr}", file=sys.stderr, flush=True)
    except:
        pass

    return file_obj


def _make_close_wrapper(original_close):
    """Create a wrapper around file.close() that logs closure."""
    def close_wrapper(self):
        addr = hex(id(self))
        file_path = _open_files.get(addr, 'unknown')

        # Try to write to stderr, but don't fail if it's redirected/closed
        try:
            print(f"[FILE_DEBUG] CLOSE: {file_path} -> {addr}", file=sys.stderr, flush=True)
        except:
            pass

        if addr in _open_files:
            del _open_files[addr]

        return original_close(self)
    return close_wrapper


def enable_file_tracking():
    """Enable file operation tracking. Call this at the start of your script."""
    global _original_open, _tracking_enabled

    if _tracking_enabled:
        return

    print("[FILE_DEBUG] Enabling file tracking", file=sys.stderr, flush=True)

    # Save original open and patch it
    _original_open = builtins.open
    builtins.open = _debug_open

    # Patch close methods for different file types
    io.TextIOWrapper.close = _make_close_wrapper(io.TextIOWrapper.close)
    io.BufferedWriter.close = _make_close_wrapper(io.BufferedWriter.close)
    io.BufferedReader.close = _make_close_wrapper(io.BufferedReader.close)

    _tracking_enabled = True


def get_open_files():
    """Return dictionary of currently open files: {address: path}."""
    return _open_files.copy()


def print_open_files():
    """Print all currently open files."""
    print("\n[FILE_DEBUG] Currently open files:", file=sys.stderr, flush=True)
    if not _open_files:
        print("  (none)", file=sys.stderr, flush=True)
    else:
        for addr, path in _open_files.items():
            print(f"  {addr}: {path}", file=sys.stderr, flush=True)
