"""Learn only from an on-screen question and its matching revealed answer."""
import hashlib
import json
import os
import threading
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from matcher import CARD, image_feature, normalize_text, canonical_learned_name


class RevealSampler:
    """Buffer candidate reveals while the answer reader is busy."""
    def __init__(self, capture_factory, allowed, bounds, pending):
        self.capture_factory = capture_factory
        self.allowed = allowed
        self.bounds = bounds
        self.pending = pending
        self.frames = []
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        with self.lock:
            self.stop.set()

    def run(self):
        left, top, width, height = self.bounds
        rect = dict(left=left, top=top, width=width, height=height)
        try:
            with self.capture_factory() as capture:
                for _ in range(100):
                    if self.stop.is_set() or not self.allowed():
                        return
                    frame = np.asarray(capture.grab(rect))[:, :, :3]
                    if not self.allowed():
                        return
                    if reveal_index(self.pending, frame) is not None:
                        with self.lock:
                            if self.stop.is_set():
                                return
                            self.frames.append(frame.copy())
                            if len(self.frames) == 3:
                                return
                    if self.stop.wait(.05):
                        return
        except Exception:
            return

def catalog_name_for_reveal(name, catalog):
    name=canonical_learned_name(name)
    key=normalize_text(name)
    names={record['name'] for record in getattr(catalog,'records',[])}
    names.update(record['name'] for record,_ in getattr(catalog,'observed',[]))
    ranked=sorted(((SequenceMatcher(None,key,normalize_text(candidate)).ratio(),candidate)
                   for candidate in names if normalize_text(candidate)),reverse=True)
    if not ranked:return name
    best,candidate=ranked[0]
    second=ranked[1][0] if len(ranked)>1 else 0.
    if best==1. or (len(key)>=8 and best>=.875 and best-second>=.12):return candidate
    return name

def save_miss_evidence(pending, reveal, root):
    """Save only the game area containing the card and the answer buttons."""
    if pending['raw'].shape != reveal.shape:
        raise ValueError('Question and reveal dimensions differ')
    h,w=reveal.shape[:2]
    boxes=[pending['card_box'],*pending['boxes']]
    x1=max(0,min(x for x,y,bw,bh in boxes));y1=max(0,min(y for x,y,bw,bh in boxes))
    x2=min(w,max(x+bw for x,y,bw,bh in boxes));y2=min(h,max(y+bh for x,y,bw,bh in boxes))
    if x2<=x1 or y2<=y1:raise ValueError('Invalid card and choice bounds')
    folder=Path(root)/'assets'/'miss_diagnostics';folder.mkdir(parents=True,exist_ok=True)
    identifier=uuid.uuid4().hex
    result={}
    for label,image in (('question',pending['raw']),('reveal',reveal)):
        path=folder/(identifier+'_'+label+'.png')
        if not cv2.imwrite(str(path),image[y1:y2,x1:x2]):raise OSError('Could not save miss evidence')
        result[label]=path.relative_to(root).as_posix()
    recent = sorted(folder.glob('*_question.png'), key=lambda file: file.stat().st_mtime_ns,
                    reverse=True)
    for old in recent[24:]:
        old.with_name(old.name.replace('_question.png', '_reveal.png')).unlink(missing_ok=True)
        old.unlink(missing_ok=True)
    return result

def reveal_index(pending, raw):
    """Identify a single green answer at the same positions as the question."""
    if raw.shape != pending['raw'].shape:return None
    fractions=[]
    for x,y,w,h in pending['boxes']:
        crop=raw[y:y+h,x:x+w]
        if crop.size==0:return None
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        fractions.append(float(np.mean(cv2.inRange(hsv,(35,50,80),(95,255,255))>0)))
    green=[i for i,value in enumerate(fractions) if value>.40]
    if len(green)!=1 or any(fractions[i]>.15 for i in range(4) if i!=green[0]):return None
    return green[0]


def revealed_answer(pending, raw, reader, now=None):
    """Return (name, card image) after a verified reveal, otherwise None."""
    now = time.monotonic() if now is None else now
    if now - pending['time'] > 20 or raw.shape != pending['raw'].shape:
        return None
    index=reveal_index(pending,raw)
    if index is None:return None
    crops=[]; prior=[]
    for x,y,w,h in pending['boxes']:
        crop=raw[y:y+h,x:x+w]; old=pending['raw'][y:y+h,x:x+w]
        if crop.size == 0 or old.shape != crop.shape: return None
        crops.append(crop); prior.append(old)
    # Already-recognized answers need a new sample only when the macro picked
    # a different option from the one the game revealed as correct.
    if pending.get('clicked_index') == index and not pending.get('learn_correct_guess'):
        return None
    oldcard=pending['frame'][CARD[1]:CARD[3],CARD[0]:CARD[2]]
    cx,cy,cw,ch=pending['card_box']
    newcard=raw[cy:cy+ch,cx:cx+cw]
    if newcard.size==0:return None
    newcard=cv2.resize(newcard,(oldcard.shape[1],oldcard.shape[0]))
    if float(np.mean(np.abs(oldcard.astype(np.float32)-newcard.astype(np.float32))))<12:return None
    # The answer names in the reveal must match the pending question. Comparing
    # all four prevents attaching a later reveal to an unrelated question.
    before=reader.options_native(prior); after=reader.options_native(crops)
    # Native OCR may miss long, low-contrast labels. The fast canonical OCR
    # provides an independent reading of the original choice text.
    alternate=reader.options(pending['frame'])
    if len(before)!=4 or len(after)!=4:return None
    same=[]
    for (a,ac),(b,bc),(alt,altc) in zip(before,after,alternate):
        b=normalize_text(b)
        readings=((normalize_text(a),ac),(normalize_text(alt),altc))
        same.append(bool(b and bc>=.55 and any(v and conf>=.55 and SequenceMatcher(None,v,b).ratio()>=.78
                                               for v,conf in readings)))
    if same[index] and sum(same)>=3 and after[index][1]>=.75:
        name=after[index][0].strip()
    else:
        # Green highlighting can obscure the text. Only fall back to the
        # original choice when every other choice confirms this same question
        # and two independent readings agree on the original answer.
        original, confidence = before[index]
        second, second_confidence = alternate[index]
        first_key, second_key = normalize_text(original), normalize_text(second)
        if (sum(same[i] for i in range(4) if i != index) != 3 or
                confidence < .80 or second_confidence < .55 or not second_key or
                SequenceMatcher(None, first_key, second_key).ratio() < .78):
            return None
        name=original.strip()
    if len(normalize_text(name))<4:return None
    # Confirm this is a reveal rather than a green selection on the question.
    return name,oldcard.copy()


def save_observation(catalog, name, card, root):
    """Save a learned sample and make it available immediately and after restart."""
    name=catalog_name_for_reveal(name,catalog)
    root=Path(root); folder=root/'assets'/'learned'; folder.mkdir(parents=True,exist_ok=True)
    feature=image_feature(card,False)
    for record, existing in catalog.observed:
        if normalize_text(record['name'])==normalize_text(name) and float(np.mean((existing-feature)**2))<20:
            return False
    digest=hashlib.sha256(card.tobytes()+normalize_text(name).encode()).hexdigest()[:24]
    image_path=folder/(digest+'.png'); manifest=folder/'manifest.json'
    records=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
    if not cv2.imwrite(str(image_path),card): raise OSError('Could not save learned image')
    record={'name':name,'variant':'Learned from reveal','image_file':f'assets/learned/{digest}.png'}
    records.append(record)
    temp=folder/'manifest.tmp'
    temp.write_text(json.dumps(records,indent=2),encoding='utf-8')
    os.replace(temp,manifest)
    catalog.observed.append((record,feature))
    return True
