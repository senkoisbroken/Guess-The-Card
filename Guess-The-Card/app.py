"""Guess the Card: visual matching and normal foreground mouse input only."""
import os
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_tcl_root = Path(sys.base_prefix) / 'tcl'
for _env, _folder, _file in [('TCL_LIBRARY', 'tcl8.6', 'init.tcl'), ('TK_LIBRARY', 'tk8.6', 'tk.tcl')]:
    if (_tcl_root / _folder / _file).is_file():
        os.environ.setdefault(_env, str(_tcl_root / _folder))

import tkinter as tk
from tkinter import ttk
import cv2
import mss
import numpy as np

from desktop import WindowsDesktop
from matcher import Catalog, QuestionReader, answer_for, answer_compact, answer_with_native, best_effort_answer, CARD, CLICK_POINTS, CHOICES, ROOT
from diagnostics import open_log, preview_upload, read_build, run_speed_check
from layout import compact_question, compact_replay, give_up_button
from learning import RevealSampler, revealed_answer, reveal_index, save_observation, save_miss_evidence

F7 = 0x76
F8 = 0x77
MAX_STALE_RETRIES = 2
MAX_REPLAY_ATTEMPTS = 3
REPLAY_CONFIRM_SECONDS = .5
LOG = None


@dataclass
class ReplayAttempt:
    target: tuple
    count: int = 0
    confirm_by: float = 0.
    sent: bool = False

def diagnostic(event, **details):
    """Record decisions and learning outcomes without capturing other apps."""
    try:
        if LOG is not None:
            LOG.info('%s', json.dumps({'event':event,**details},ensure_ascii=False))
    except OSError:
        pass

def question_change_metrics(before, after):
    card_a = before[124:404, 207:418].astype(np.int16)
    card_b = after[124:404, 207:418].astype(np.int16)
    choices_a = before[228:320, 448:780].astype(np.int16)
    choices_b = after[228:320, 448:780].astype(np.int16)
    choice_differences = []
    for x, y in CHOICES:
        # Compare each label independently, excluding the animated button border.
        a = before[y+4:y+40, x+8:x+156].astype(np.int16)
        b = after[y+4:y+40, x+8:x+156].astype(np.int16)
        difference = a - b
        shift = np.median(difference, axis=(0, 1))
        choice_differences.append(float(np.mean(np.abs(difference - shift))))
    return {
        'card_difference': float(np.mean(np.abs(card_a - card_b))),
        'choice_difference': float(np.mean(np.abs(choices_a - choices_b))),
        'choice_detail_difference': max(choice_differences),
    }

def same_question(before, after):
    """Reject a changed card or answer label while tolerating button brightness shifts."""
    scores = question_change_metrics(before, after)
    return scores['card_difference'] < 5 and scores['choice_detail_difference'] < 4

def same_pending_question(before, after):
    """Keep a pending card while an answer button changes to its pressed state."""
    card_a=before[124:404,207:418].astype(np.int16)
    card_b=after[124:404,207:418].astype(np.int16)
    return float(np.mean(np.abs(card_a-card_b))) < 5

def pending_is_new_question(pending, frame):
    return not same_pending_question(pending['frame'], frame)

class App:
    def __init__(self, root):
        global LOG
        LOG = open_log(ROOT)
        build = read_build(ROOT)
        diagnostic('session_start', build=build)
        self.root = root
        self.desktop = WindowsDesktop(); self.desktop.prepare_dpi()
        self.running = threading.Event(); self.closed = threading.Event()
        self.speed_requested = threading.Event()
        self.speed_in_progress = False
        self.messages = queue.Queue(); self.last_f7 = False
        self.live = False; self.replay_enabled = False; self.guess_enabled = False
        self.automatic = tk.BooleanVar(value=True)
        self.replay = tk.BooleanVar(value=True)
        self.guess = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Paused. Press F7 or Start, then focus Roblox.')
        self.detail = tk.StringVar(value='Preparing card catalog…')
        root.title(f'Guess the Card - {build}')
        frame = ttk.Frame(root, padding=18); frame.grid()
        ttk.Label(frame, text='Guess the Card', font=('Segoe UI', 16, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Label(frame, text='Screen-only recognition. F7 starts or pauses.', wraplength=480).grid(row=1, column=0, sticky='w', pady=6)
        self.click_box = ttk.Checkbutton(frame, text='Automatically click answers', variable=self.automatic)
        self.click_box.grid(row=2, column=0, sticky='w')
        self.replay_box = ttk.Checkbutton(frame, text='Give up and play again after a finished run', variable=self.replay)
        self.replay_box.grid(row=3, column=0, sticky='w')
        self.guess_box = ttk.Checkbutton(frame, text='Guess when uncertain (can be wrong)', variable=self.guess)
        self.guess_box.grid(row=4, column=0, sticky='w')
        self.button = ttk.Button(frame, text='Start (F7)', command=self.toggle)
        self.button.grid(row=5, column=0, sticky='w', pady=10)
        ttk.Label(frame, textvariable=self.status, wraplength=480).grid(row=6, column=0, sticky='w')
        ttk.Label(frame, textvariable=self.detail, wraplength=480).grid(row=7, column=0, sticky='w', pady=8)
        ttk.Label(frame, text='F7 or F8 starts and pauses. Keep Roblox in front.', wraplength=480).grid(row=8, column=0, sticky='w')
        self.diagnostic_button = ttk.Button(frame, text='Review diagnostic log',
            command=lambda: preview_upload(root, ROOT, self.messages, self.closed, self.diagnostic_button))
        self.diagnostic_button.grid(row=9, column=0, sticky='w', pady=6)
        self.speed_button = ttk.Button(frame, text='Run speed check', command=self.request_speed_check)
        self.speed_button.grid(row=10, column=0, sticky='w', pady=6)
        self.speed_button.state(['disabled'])
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.thread = threading.Thread(target=self.worker, daemon=True); self.thread.start()
        root.after(100, self.poll)

    def request_speed_check(self):
        if self.running.is_set() or self.speed_in_progress or not self.thread.is_alive():
            return
        self.speed_in_progress = True
        self.speed_button.state(['disabled'])
        self.button.state(['disabled'])
        self.detail.set('Running the speed check while paused…')
        self.speed_requested.set()

    def toggle(self):
        if self.speed_in_progress:
            return
        if self.running.is_set():
            self.running.clear(); self.button.config(text='Start (F7)')
            self.click_box.state(['!disabled']); self.replay_box.state(['!disabled']); self.guess_box.state(['!disabled'])
            self.status.set('Paused.')
            if self.thread.is_alive():
                self.speed_button.state(['!disabled'])
        else:
            self.speed_button.state(['disabled'])
            self.live = bool(self.automatic.get()); self.replay_enabled = bool(self.replay.get()); self.guess_enabled = bool(self.guess.get())
            self.click_box.state(['disabled']); self.replay_box.state(['disabled']); self.guess_box.state(['disabled'])
            self.running.set(); self.button.config(text='Pause (F7)')
            self.status.set('Running: automatic clicking enabled.' if self.live else 'Running in preview: no clicks.')

    def poll(self):
        if self.closed.is_set(): return
        down = self.desktop.down(F7) or self.desktop.down(F8)
        if down and not self.last_f7: self.toggle()
        self.last_f7 = down
        try:
            while True:
                kind, text = self.messages.get_nowait()
                if kind == 'detail': self.detail.set(text)
                elif kind == 'speed_ready' and not self.running.is_set():
                    self.speed_button.state(['!disabled'])
                elif kind == 'speed_done':
                    self.speed_in_progress = False
                    self.detail.set(text)
                    self.button.state(['!disabled'])
                    self.speed_button.state(['!disabled'])
                elif kind == 'diagnostics':
                    self.detail.set(text)
                    self.diagnostic_button.state(['!disabled'])
                elif kind == 'error':
                    self.running.clear(); self.status.set(text); self.button.config(text='Start (F7)')
                    self.click_box.state(['!disabled']); self.replay_box.state(['!disabled']); self.guess_box.state(['!disabled'])
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def close(self):
        self.running.clear(); self.closed.set(); self.thread.join(timeout=2); self.root.destroy()

    def worker(self):
        try:
            catalog = Catalog()
            catalog.load(lambda i,n: self.messages.put(('detail', f'Indexing images: {i}/{n}')))
            reader = QuestionReader()
            self.messages.put(('detail', f'Catalog ready: {len(catalog.records)} wiki entries.'))
            self.messages.put(('speed_ready', ''))
            answered = pending = replay = None
            with mss.MSS() as capture:
                while not self.closed.wait(.03):
                    if not self.running.is_set():
                        if pending is not None: diagnostic('pending_abandoned',reason='paused')
                        answered = pending = replay = None
                        if self.speed_requested.is_set():
                            self.speed_requested.clear()
                            try:
                                results = run_speed_check(ROOT, reader, self.closed.is_set)
                                diagnostic('speed_check', **results)
                                detail = ('Speed check saved. Open Review diagnostic log to send it. '
                                          f"OCR: {results['wide_ocr_ms']:.1f} ms.")
                            except Exception as exc:
                                detail = f'Speed check failed: {type(exc).__name__}'
                            self.messages.put(('speed_done', detail))
                        continue
                    hwnd = self.desktop.roblox_window()
                    if not hwnd:
                        if pending is not None: diagnostic('pending_abandoned',reason='roblox_not_foreground')
                        pending = None
                        self.messages.put(('detail', 'Waiting for foreground Roblox.'))
                        continue
                    bounds = self.desktop.client_rect(hwnd)
                    left,top,width,height = bounds
                    if width < 500 or height < 350:
                        self.messages.put(('detail', 'Roblox window is too small to read.'))
                        continue
                    def grab():
                        pixels = np.asarray(capture.grab({'left':left, 'top':top, 'width':width, 'height':height}))
                        return pixels[:,:,:3].copy()
                    def locate(raw):
                        point=give_up_button(raw,reader)
                        if point is not None:return {'kind':'give_up','point':point}
                        wide = cv2.resize(raw,(960,504),interpolation=cv2.INTER_AREA)
                        if 1.75 <= width/height <= 2.05:
                            if reader.is_replay(wide):
                                return {'kind':'replay','point':(round(width*480/960),round(height*382/504))}
                            if reader.is_question(wide):
                                return {'kind':'question','frame':wide,'choice_crops':None,
                                        'card_box':(round(width*207/960),round(height*124/504),round(width*211/960),round(height*280/504)),
                                        'boxes':[(round(width*x/960),round(height*y/504),round(width*164/960),round(height*44/504)) for x,y in CHOICES],
                                        'clicks':[(round(width*x/960),round(height*y/504)) for x,y in CLICK_POINTS]}
                        point=compact_replay(raw,reader)
                        if point is not None:return {'kind':'replay','point':point}
                        compact=compact_question(raw,reader)
                        if compact is not None:
                            compact['kind']='question';return compact
                        return {'kind':'waiting'}
                    capture_started = time.monotonic()
                    frame = grab()
                    capture_ms = round((time.monotonic() - capture_started) * 1000, 2)
                    if not self.running.is_set() or self.desktop.roblox_window() != hwnd or self.desktop.client_rect(hwnd) != bounds:
                        continue
                    allowed = lambda: self.running.is_set() and not self.closed.is_set() and self.live
                    if pending is not None:
                        if pending['bounds'] != bounds or pending['hwnd'] != hwnd or time.monotonic()-pending['time'] > 20:
                            diagnostic('pending_abandoned',reason='window_or_timeout')
                            pending = None
                        else:
                            revealed = reveal_index(pending,frame)
                            if (revealed is not None and pending.get('clicked_index') is not None and
                                    revealed != pending['clicked_index'] and not pending.get('miss_recorded')):
                                pending['miss_recorded']=True
                                try:
                                    evidence=save_miss_evidence(pending,frame,ROOT)
                                    diagnostic('miss_evidence',correct_index=revealed,
                                               clicked_index=pending['clicked_index'],**evidence)
                                except (OSError, ValueError, KeyError, cv2.error) as exc:
                                    diagnostic('miss_evidence_error',error=str(exc))
                            if revealed is not None and not pending.get('reveal_logged'):
                                clicked = pending.get('clicked_index')
                                outcome = 'unknown'
                                if clicked is not None:
                                    outcome = 'correct' if clicked == revealed else 'wrong'
                                diagnostic('answer_outcome', outcome=outcome,
                                           correct_index=revealed, clicked_index=clicked)
                                pending['reveal_logged'] = True
                            if revealed is None:
                                learned = None
                            elif revealed == pending.get('clicked_index') and not pending.get('learn_correct_guess'):
                                pending = None
                                learned = None
                            else:
                                learned = revealed_answer(pending, frame, reader)
                            if learned is not None:
                                try:
                                    saved = save_observation(catalog,*learned,ROOT)
                                    diagnostic('learned' if saved else 'already_known',name=learned[0],correct_index=revealed)
                                    if saved:
                                        self.messages.put(('detail', f'Learned from revealed answer: {learned[0]}'))
                                except (OSError, ValueError, KeyError, TypeError) as exc:
                                    diagnostic('learning_error',error=str(exc))
                                    self.messages.put(('detail', f'Could not save learned card: {exc}'))
                                pending = None
                            elif revealed is not None and pending is not None and not pending.get('learning_rejected_logged'):
                                diagnostic('reveal_not_learned', correct_index=revealed,
                                           clicked_index=pending.get('clicked_index'))
                                pending['learning_rejected_logged'] = True
                    layout_started = time.monotonic()
                    layout = locate(frame)
                    layout_ms = round((time.monotonic() - layout_started) * 1000, 2)
                    if layout['kind'] == 'give_up':
                        answered = replay = None
                        self.messages.put(('detail', 'Run lost. Give up detected.'))
                        if self.live and self.replay_enabled:
                            x,y=layout['point']
                            def give_up_still_visible():
                                return give_up_button(grab(),reader) == (x,y)
                            clicked = self.desktop.click_client(hwnd,bounds,x,y,allowed,before_press=give_up_still_visible)
                            if clicked:
                                self.messages.put(('detail', 'Clicked Give up; waiting for results.'))
                            else:
                                self.messages.put(('detail', 'Give up click interrupted; waiting to retry.'))
                        elif not self.live:
                            self.messages.put(('detail', 'Give up detected; automatic clicks are disabled.'))
                        else:
                            self.messages.put(('detail', 'Give up detected; Give up / Play again setting is disabled.'))
                        continue
                    if layout['kind'] == 'replay':
                        answered = None
                        if not self.live or not self.replay_enabled:
                            self.messages.put(('detail', 'Run finished. Play again detected.'))
                            continue
                        x, y = layout['point']
                        target = (hwnd, bounds, x, y)
                        if replay is None:
                            replay = ReplayAttempt(target)
                        if replay.target != target:
                            self.messages.put(('detail', 'Replay target moved. Pause and resume to retry.'))
                            continue
                        now = time.monotonic()
                        if now < replay.confirm_by:
                            continue
                        if replay.count >= MAX_REPLAY_ATTEMPTS:
                            diagnostic('replay_unconfirmed', attempts=replay.count)
                            self.messages.put(('detail', 'Play Again did not start a run.'))
                            replay.confirm_by = float('inf')
                            continue

                        def replay_still_visible():
                            current = locate(grab())
                            return current['kind'] == 'replay' and current['point'] == (x, y)

                        clicked = self.desktop.click_client(hwnd, bounds, x, y, allowed,
                                                            before_press=replay_still_visible)
                        replay.count += 1
                        replay.sent |= clicked
                        replay.confirm_by = time.monotonic() + REPLAY_CONFIRM_SECONDS
                        diagnostic('replay_attempt', attempt=replay.count, click_sent=clicked)
                        detail = ('Play Again clicked. Waiting for the next question.' if clicked else
                                  'Play Again click interrupted. Waiting to retry.')
                        self.messages.put(('detail', detail))
                        continue
                    if layout['kind'] != 'question':
                        answered = None
                        self.messages.put(('detail', 'Waiting for an active question.'))
                        continue
                    if replay is not None:
                        if replay.sent and replay.target[:2] == (hwnd, bounds):
                            diagnostic('replay_confirmed', attempts=replay.count)
                        replay = None
                    frame_raw = frame
                    frame = layout['frame']
                    if pending is not None and pending_is_new_question(pending, frame):
                        diagnostic('pending_abandoned',reason='next_question',reveal_seen=bool(pending.get('reveal_logged')))
                        pending = None
                    stale_retries = 0
                    if pending is not None:
                        stale_retries = pending.get('stale_retries', 0)
                        if (not 0 < stale_retries <= MAX_STALE_RETRIES
                                or pending.get('reveal_logged')):
                            # Unknown or already answered: keep watching for the reveal.
                            continue
                        pending = None
                    if answered is not None and same_question(answered, frame):
                        continue
                    native = layout['choice_crops']
                    candidate = {'time':time.monotonic(),'raw':frame_raw,'frame':frame,
                                 'boxes':layout['boxes'],'card_box':layout['card_box'],
                                 'bounds':bounds,'hwnd':hwnd,'clicked_index':None}
                    def same_client(hwnd=hwnd, bounds=bounds):
                        return (self.running.is_set() and not self.closed.is_set()
                                and self.desktop.roblox_window() == hwnd
                                and self.desktop.client_rect(hwnd) == bounds)
                    stale_frame = None

                    def fallback_frame_is_current():
                        nonlocal stale_frame
                        if not same_client():
                            stale_frame = {'kind': 'client_changed'}
                            return False
                        current = locate(grab())
                        if current['kind'] == 'question' and same_question(frame, current['frame']):
                            return True
                        stale_frame = current
                        return False

                    with RevealSampler(mss.MSS, same_client, bounds, candidate) as sampler:
                        result = (answer_compact(frame, catalog, reader, native,
                                                 allow_guess=self.guess_enabled,
                                                 still_current=fallback_frame_is_current)
                                  if native is not None else answer_for(frame, catalog, reader))
                        if result['status'] == 'unknown' and native is None:
                            native_crops = [frame_raw[round(y*height/504):round((y+44)*height/504),
                                                      round(x*width/960):round((x+164)*width/960)] for x,y in CHOICES]
                            result = answer_with_native(frame, catalog, reader, native_crops, result,
                                                        allow_guess=self.guess_enabled,
                                                        still_current=fallback_frame_is_current)
                        if stale_frame is None and result['status'] == 'unknown' and self.guess_enabled:
                            result = best_effort_answer(result,catalog,frame)
                    recognition_ms = round((time.monotonic() - candidate['time']) * 1000)
                    if not same_client():
                        continue
                    learned = None
                    for reveal in sampler.frames:
                        learned = revealed_answer(candidate,reveal,reader)
                        if learned is not None:
                            break
                    if learned is not None:
                        try:
                            saved = save_observation(catalog,*learned,ROOT)
                            diagnostic('learned' if saved else 'already_known',name=learned[0],
                                       source='buffered_reveal',correct_index=reveal_index(candidate,reveal))
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            diagnostic('learning_error',error=str(exc))
                        continue
                    def schedule_stale_retry(stage, current):
                        candidate['stale_retries'] = stale_retries + 1
                        retry = candidate['stale_retries'] <= MAX_STALE_RETRIES
                        diagnostic('stale_answer', stage=stage, layout=current['kind'],
                                   client_size=(width, height), retry=retry, recognition_ms=recognition_ms,
                                   capture_ms=capture_ms, layout_ms=layout_ms,
                                   **(question_change_metrics(frame, current['frame'])
                                      if current['kind'] == 'question' else {}))
                        detail = ('Question changed before clicking. Checking again.' if retry else
                                  'Question stayed unstable. Watching for the reveal.')
                        self.messages.put(('detail', detail))

                    if stale_frame is not None:
                        schedule_stale_retry('before_native_fallback', stale_frame)
                        pending = candidate
                        continue
                    if result['status'] not in ('matched','guess'):
                        if pending is None:
                            match = result.get('match') or {}
                            distance = match.get('distance', float('inf'))
                            diagnostic('unanswered', reason=result.get('reason', 'unknown'),
                                       guess_enabled=self.guess_enabled, client_size=(width, height),
                                       recognition_ms=recognition_ms, capture_ms=capture_ms,
                                       layout_ms=layout_ms, options=result.get('options', []),
                                       candidate=match.get('name'), candidate_index=match.get('index'),
                                       candidate_distance=distance if np.isfinite(distance) else None,
                                       covered_choices=match.get('covered_choices'))
                            pending = candidate
                        self.messages.put(('detail', result.get('reason', 'Uncertain; no click.')))
                        continue
                    prefix = f"Guess ({result['guess_kind']}): " if result['status']=='guess' else ''
                    self.messages.put(('detail', f"{prefix}{result['name']} ({result['variant']}); choice {result['index']+1}; image error {result['distance']:.0f}."))
                    if not self.live:
                        continue
                    if not self.running.is_set() or self.desktop.roblox_window() != hwnd or self.desktop.client_rect(hwnd) != bounds:
                        continue
                    latest = locate(grab())
                    if latest['kind'] != 'question' or not same_question(frame, latest['frame']):
                        schedule_stale_retry('before_move', latest)
                        pending = candidate
                        continue
                    x,y = latest['clicks'][result['index']]
                    def question_still_visible():
                        current = locate(grab())
                        if (current['kind'] == 'question' and
                                current['clicks'][result['index']] == (x, y) and
                                same_question(frame, current['frame'])):
                            return True
                        schedule_stale_retry('before_press', current)
                        return False
                    if self.desktop.click_client(hwnd, bounds, x, y, allowed, before_press=question_still_visible):
                        answered = frame.copy()
                        diagnostic('clicked',status=result['status'],name=result['name'],index=result['index'],
                                   reason=result.get('reason',''),distance=result.get('distance'),
                                   recognition_ms=recognition_ms, capture_ms=capture_ms,
                                   layout_ms=layout_ms, client_size=(width, height),
                                   options=result.get('options',[]),match=result.get('match',{}))
                        pending = {'time':time.monotonic(),'raw':frame_raw.copy(),'frame':frame.copy(),
                                   'boxes':layout['boxes'],'card_box':layout['card_box'],
                                   'bounds':bounds,'hwnd':hwnd,'clicked_index':result['index'],
                                   'learn_correct_guess':result['status']=='guess'}
                    else:
                        if 'stale_retries' not in candidate:
                            candidate['stale_retries'] = stale_retries + 1
                            retry = candidate['stale_retries'] <= MAX_STALE_RETRIES
                            diagnostic('click_interrupted', retry=retry, recognition_ms=recognition_ms)
                            detail = ('Answer click interrupted. Checking again.' if retry else
                                      'Answer click interrupted. Watching for the reveal.')
                            self.messages.put(('detail', detail))
                        pending = candidate
        except Exception as exc:
            self.messages.put(('error', f'Stopped: {type(exc).__name__}: {exc}'))

if __name__ == '__main__':
    root = tk.Tk(); app = App(root)
    if '--ui-smoke' in sys.argv: root.after(1500, app.close)
    root.mainloop()
