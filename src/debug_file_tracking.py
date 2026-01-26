"""Debug utility to track file open/close operations.

Import and call enable_file_tracking() at the start of your script to log all file operations.
"""

import sys
import builtins


_original_open = None
_open_files = {}
_tracking_enabled = False


class FileWrapper:
    """Wrapper that logs close operations."""

    def __init__(self, file_obj, file_path):
        self._file = file_obj
        self._path = file_path
        self._addr = hex(id(file_obj))

    def close(self):
        """Log and close the file."""
        try:
            print(f"[FILE_DEBUG] CLOSE: {self._path} -> {self._addr}", file=sys.stderr, flush=True)
        except:
            pass

        if self._addr in _open_files:
            del _open_files[self._addr]

        return self._file.close()

    def __enter__(self):
        return self._file.__enter__()

    def __exit__(self, *args):
        self.close()
        return False

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped file."""
        return getattr(self._file, name)

    def __iter__(self):
        return iter(self._file)

    def __next__(self):
        return next(self._file)


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

    # Return wrapped file that will log on close
    return FileWrapper(file_obj, file_path)


def enable_file_tracking():
    """Enable file operation tracking. Call this at the start of your script."""
    global _original_open, _tracking_enabled

    if _tracking_enabled:
        return

    print("[FILE_DEBUG] Enabling file tracking", file=sys.stderr, flush=True)

    # Save original open and patch it
    _original_open = builtins.open
    builtins.open = _debug_open

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
