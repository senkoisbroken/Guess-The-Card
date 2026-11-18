"""Files shipped in a Guess the Card update; local learning is excluded."""
import re

RUNTIME = ('app.py', 'desktop.py', 'layout.py', 'learning.py', 'matcher.py',
           'diagnostics.py', 'diagnostic_upload.py', 'wiki_manifest.json', 'wiki_catalog.csv',
           'requirements.txt', 'updater.py', 'package_layout.py', 'README.md',
           'assets/features.npz', 'assets/event_title.png', 'assets/question_prompt.png',
           'assets/replay_button.png', 'assets/results_header.png',
           'assets/observed/manifest.json')
REQUIRED = frozenset((*RUNTIME, 'start.bat', 'package_info.json'))

def allowed(name):
    return name in REQUIRED or re.fullmatch(
        r'assets/(?:wiki/[a-f0-9]{24}\.img|observed/[A-Za-z0-9_-]+\.png)', name) is not None
