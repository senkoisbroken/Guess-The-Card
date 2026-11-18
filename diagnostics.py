"""Local card-guessing diagnostics and a user-reviewed upload."""
import logging
from logging.handlers import RotatingFileHandler
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from diagnostic_upload import send_snapshot, snapshot_lines


def open_log(root):
    log = logging.Logger('guess-the-card')
    handler = RotatingFileHandler(root / 'macro.log', maxBytes=131072,
                                  backupCount=1, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s guess-the-card %(message)s'))
    log.addHandler(handler)
    return log


def preview_upload(root, folder, results, closed, button):
    try:
        lines = snapshot_lines(folder)
    except (OSError, ValueError) as exc:
        messagebox.showinfo('Guess the Card diagnostics', str(exc), parent=root)
        return
    dialog = tk.Toplevel(root)
    dialog.title('Review Guess the Card diagnostics')
    dialog.transient(root)
    dialog.grab_set()
    ttk.Label(dialog, text='Review the exact log text before sending to the support server.').pack(padx=12, pady=8)
    body = tk.Text(dialog, width=76, height=20, wrap='word')
    body.pack(fill='both', expand=True, padx=12)
    body.insert('1.0', '\n'.join(lines))
    body.config(state='disabled')

    def send():
        dialog.destroy()
        button.state(['disabled'])

        def upload():
            _, status = send_snapshot(folder, lines=lines, cancelled=closed.is_set)
            results.put(('diagnostics', status))

        threading.Thread(target=upload, daemon=True).start()

    ttk.Button(dialog, text='Send reviewed log', command=send).pack(pady=8)
