"""Local card-guessing diagnostics and a user-reviewed upload."""
import json
import logging
from pathlib import Path
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



def read_build(root):
    try:
        manifest = json.loads((Path(root) / 'package_info.json').read_text(encoding='utf-8'))
        build = manifest.get('package_version')
        return build if isinstance(build, str) and build else 'unknown'
    except (OSError, ValueError, AttributeError):
        return 'unknown'


def run_speed_check(root, reader, cancelled=lambda: False):
    """Measure fixed OCR and catalog workloads while the macro is paused."""
    import hashlib
    import os
    import sys
    from statistics import median
    from time import perf_counter_ns

    import cv2
    import numpy as np
    import onnxruntime
    from matcher import CARD, CHOICES, Catalog

    root = Path(root)
    path = root / 'assets' / 'observed' / 'six_paths_of_pain_alternate_art.png'
    sample = cv2.imread(str(path))
    if sample is None:
        raise ValueError('The bundled speed-check card is missing.')
    frame = np.zeros((504, 960, 3), dtype=np.uint8)
    x1, y1, x2, y2 = CARD
    art = frame[y1:y2, x1:x2]
    art[:] = cv2.resize(sample, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
    labels = ('Eleventh Slayer', 'Six Paths of Pain', 'Manga Writer', 'Secret Puppetmaster')
    crops = []
    for (x, y), name in zip(CHOICES, labels):
        crop = frame[y:y+44, x:x+164]
        crop[:] = (205, 205, 205)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (width, _), _ = cv2.getTextSize(name, font, .43, 1)
        cv2.putText(crop, name, ((164-width)//2, 28), font, .43, (25, 25, 25), 1, cv2.LINE_AA)
        crops.append(cv2.resize(crop, (328, 88), interpolation=cv2.INTER_CUBIC))

    catalog = Catalog(include_learned=False)
    catalog.load()
    options = [(label, 1.) for label in labels]

    def measure(fn, trials):
        timings = []
        for index in range(trials + 1):
            if cancelled():
                raise InterruptedError('Speed check cancelled.')
            start = perf_counter_ns()
            result = fn()
            elapsed = (perf_counter_ns() - start) / 1_000_000
            if index:
                timings.append(elapsed)
        return round(median(timings), 2), result

    wide, wide_reads = measure(lambda: reader.options(frame), 5)
    quick, quick_reads = measure(lambda: reader.options_native(crops, quick=True), 3)
    full, full_reads = measure(lambda: reader.options_native(crops), 3)
    lookup, decision = measure(lambda: catalog.find_for_choices(art, options), 5)
    if (any(len(reads) != 4 for reads in (wide_reads, quick_reads, full_reads))
            or decision is None or decision['index'] != 1):
        raise ValueError('Speed-check fixture did not produce the expected result.')
    signature = hashlib.sha256(frame.tobytes())
    for crop in crops:
        signature.update(crop.tobytes())
    return {
        'mode': 'synthetic_benchmark',
        'benchmark_version': 1,
        'build': read_build(root),
        'fixture_sha256': signature.hexdigest(),
        'python': sys.version.split()[0],
        'opencv': cv2.__version__,
        'onnxruntime': onnxruntime.__version__,
        'cpu_threads': os.cpu_count(),
        'wide_ocr_ms': wide,
        'quick_native_ocr_ms': quick,
        'full_native_ocr_ms': full,
        'catalog_ms': lookup,
        'trial_counts': [5, 3, 3, 5],
    }


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
