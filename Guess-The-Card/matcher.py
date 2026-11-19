"""Screen-only recognition for the four-choice Guess the Card event."""
import hashlib
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from threading import local

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# Coordinates use a normalized 960x504 client image. They were measured from
# the active event screen; unsupported aspect ratios are rejected in the UI.
CARD = (207, 124, 418, 404)
HEADER = (449, 207, 782, 225)
CHOICES = [(448, 228), (616, 228), (448, 276), (616, 276)]
CLICK_POINTS = [(530, 250), (698, 250), (530, 298), (698, 298)]

def normalize_text(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

def recover_missing_options(reader, frame, options, *, canonical=None):
    if len(options)!=4 or all(text and confidence >= .3 for text,confidence in options):
        return options
    alternate = reader.options(frame) if canonical is None else canonical
    if len(alternate)!=4:return options
    return [alternate[i] if (not text or confidence < .3) and alternate[i][0] and alternate[i][1]>=.5
            else (text,confidence) for i,(text,confidence) in enumerate(options)]

def canonical_learned_name(name):
    """Repair a specifically observed OCR spelling, without fuzzy relabeling."""
    aliases = {
        'cybarsoldiar': 'Cyber Soldier',
        'herooftennpsnobunwnh': 'Hero of Ten (Humungousaur)',
        'harooftannpsnobunwnh': 'Hero of Ten (Humungousaur)',
        'thelrregular': 'The Irregular',
        'damondog': 'Demon Dog',
    }
    return aliases.get(normalize_text(name), name)

_label_matcher = local()


@lru_cache(maxsize=8192)
def _catalog_spellings(name):
    # Skin names can have a suffix that is absent from the answer button.
    short = re.sub(r'\s*\([^)]*\)', '', name).strip()
    return tuple(dict.fromkeys(filter(None, (normalize_text(name), normalize_text(short)))))


@lru_cache(maxsize=1024)
def _ocr_spelling(text):
    return normalize_text(canonical_learned_name(text))


@lru_cache(maxsize=32768)
def _spelling_similarity(name, observed):
    spellings = _catalog_spellings(name)
    if observed in spellings:
        return 1.0
    if not hasattr(_label_matcher, 'matcher'):
        _label_matcher.matcher = SequenceMatcher()
    matcher = _label_matcher.matcher
    matcher.set_seq2(observed)
    scores = []
    for spelling in spellings:
        matcher.set_seq1(spelling)
        scores.append(matcher.ratio())
    return max(scores)


def label_similarity(name, text):
    return _spelling_similarity(name, _ocr_spelling(text))

def best_effort_answer(result, catalog=None, frame=None):
    """Use the closest supported candidate; leave unsupported choices alone."""
    if result.get('status') != 'unknown':
        return result
    match=result.get('match') or {}
    options=result.get('options',[])
    if catalog is not None and frame is not None and len(options)==4:
        x1,y1,x2,y2=CARD
        recovered=recover_unread_choice(catalog,frame[y1:y2,x1:x2],options)
        if recovered:
            return recovered
    if not match and catalog is not None and frame is not None:
        options=result.get('options',[])
        if len(options)==4:
            x1,y1,x2,y2=CARD
            match=catalog.find_for_choices(frame[y1:y2,x1:x2],options,relaxed=True) or {}
    idx=match.get('index')
    distance=match.get('distance',float('inf'))
    if type(idx) is int and 0 <= idx < 4 and np.isfinite(distance) and distance >= 0:
        return {'status':'guess','index':idx,'name':match.get('name','Unknown'),
                'variant':match.get('variant','Unknown'),'distance':match.get('distance',float('nan')),
                'guess_kind':'closest image','reason':result.get('reason','Uncertain'),
                'options':result.get('options',[]),'match':match}
    return result


def recover_unread_choice(catalog, image, options):
    """Identify a single unread slot only with independent image and label evidence."""
    unread = [i for i, (label, confidence) in enumerate(options)
              if not normalize_text(label) or confidence < .3]
    if len(unread) != 1:
        return None
    match = catalog.find(image)
    if not match or match['distance'] >= 100 or match['distance'] >= .12 * match['second_distance']:
        return None
    labels = [label for i, (label, confidence) in enumerate(options)
              if i != unread[0] and confidence >= .70]
    if len(labels) != 3 or len({normalize_text(label) for label in labels}) != 3:
        return None
    if any(label_similarity(match['name'], label) >= .58 for label in labels):
        return None
    return {'status':'guess', 'guess_kind':'unread slot', 'index':unread[0],
            'name':match['name'], 'variant':match['variant'], 'distance':match['distance'],
            'options':options, 'match':match, 'reason':'Strong catalog image; one unread choice.'}

def image_feature(image, reference):
    image = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    # Align the wiki's card art with the game's blurred presentation. The
    # lower part of the wiki art contains card statistics, so omit it.
    # OpenCV GaussianBlur: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
    image = cv2.GaussianBlur(image, (0, 0), 8 if reference else 3)
    return cv2.resize(image[18:80, 5:91], (24, 18), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)

class Catalog:
    def __init__(self):
        raw = (ROOT / 'wiki_manifest.json').read_bytes()
        self.manifest_hash = hashlib.sha256(raw).hexdigest()
        self.records = json.loads(raw)['records']
        self.features = None
        self.observed = []
        for observed_manifest in (ROOT / 'assets' / 'observed' / 'manifest.json', ROOT / 'assets' / 'learned' / 'manifest.json'):
            if observed_manifest.exists():
                try:
                    for record in json.loads(observed_manifest.read_text(encoding='utf-8')):
                        image = cv2.imread(str(ROOT / record['image_file']))
                        if image is not None:
                            record = dict(record, name=canonical_learned_name(record['name']))
                            self.observed.append((record, image_feature(image, False)))
                except (OSError, ValueError, KeyError, TypeError):
                    continue

    def load(self, progress=lambda *_: None):
        cache = ROOT / 'assets' / 'features.npz'
        if cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as data:
                    if str(data['manifest_hash']) == self.manifest_hash:
                        self.features = data['features']
                        if self.features.shape == (len(self.records), 1296):
                            return
            except (OSError, ValueError, KeyError):
                pass
        result = []
        for i, item in enumerate(self.records):
            image = cv2.imread(str(ROOT / item['image_file'])) if item.get('downloaded') else None
            result.append(image_feature(image, True) if image is not None else np.full(1296, np.nan, np.float32))
            if i % 100 == 0:
                progress(i, len(self.records))
        self.features = np.stack(result)
        np.savez_compressed(cache, features=self.features, manifest_hash=self.manifest_hash)

    def find(self, image):
        if self.features is None:
            raise RuntimeError('Catalog not loaded')
        feature = image_feature(image, False)
        # These samples were paired with the game's green revealed answer.
        # Use a tight distance: an uncertain resemblance must not override wiki art.
        if self.observed:
            observed = sorted(((float(np.mean((f-feature)**2)), r) for r,f in self.observed), key=lambda x:x[0])
            distance, record = observed[0]
            other = next((d for d,r in observed[1:] if normalize_text(r['name']) != normalize_text(record['name'])),float('inf'))
            if distance < 65 and distance < .25*other:
                return {'name':record['name'],'variant':record['variant'],'distance':distance,
                        'second_distance':other,'second_name':'','source':'observed'}
        scores = np.mean((self.features - feature) ** 2, axis=1)
        scores = np.where(np.isfinite(scores), scores, np.inf)
        order = np.argsort(scores)
        if not len(order) or not np.isfinite(scores[order[0]]):
            return None
        best = int(order[0]); name = normalize_text(self.records[best]['name'])
        second_i = next((int(i) for i in order[1:] if normalize_text(self.records[int(i)]['name']) != name), None)
        second = float(scores[second_i]) if second_i is not None else float('inf')
        return {'name': self.records[best]['name'], 'variant': self.records[best]['variant'],
                'distance': float(scores[best]), 'second_distance': second,
                'second_name': self.records[second_i]['name'] if second_i is not None else ''}

    def find_for_choices(self, image, options, relaxed=False):
        """Compare images only for names that can be mapped to one visible choice."""
        if self.features is None:
            raise RuntimeError('Catalog not loaded')
        @lru_cache(maxsize=None)
        def similarity(name, text):
            return label_similarity(name,text)
        names = {r['name'] for r in self.records} | {r['name'] for r, _ in self.observed}
        spellings = {name: _catalog_spellings(name) for name in names}
        reserved = []
        for text, confidence in options:
            label = _ocr_spelling(text) if confidence >= .70 else ''
            owners = {name for name, variants in spellings.items() if label in variants} if label else set()
            # Parenthetical skin names still share ownership of their base label.
            base_names = {spellings[name][-1] for name in owners}
            reserved.append(owners if len(base_names) == 1 else set())

        best_text = [
            1.0 if reserved[i] else max((similarity(name, text) for name in names), default=0.)
            if text else 0.
            for i, (text, _) in enumerate(options)
        ]

        @lru_cache(maxsize=None)
        def choice_for(name):
            ranked = sorted(((similarity(name, text), i)
                             for i, (text, confidence) in enumerate(options)
                             if text and confidence >= (.3 if relaxed else .5)
                             and (not reserved[i] or name in reserved[i])
                             and similarity(name, text) >= best_text[i] - .12), reverse=True)
            if not ranked or ranked[0][0]<(.45 if relaxed else .53):
                return None
            if len(ranked)>1 and ranked[0][0]-ranked[1][0]<(.03 if relaxed else .08):
                return None
            return ranked[0][1]
        feature=image_feature(image,False)
        observed=[]
        for record,ref in self.observed:
            idx=choice_for(record['name'])
            if idx is not None:
                observed.append((float(np.mean((ref-feature)**2)),idx,record))
        observed.sort(key=lambda v:v[0])
        def summarize_candidates(entries):
            best={}
            for distance,index,record,source in entries:
                if not np.isfinite(distance):continue
                if index not in best or distance<best[index]['distance']:
                    best[index]={'index':index,'name':record['name'],'variant':record['variant'],
                                 'distance':float(distance),'source':source}
            return sorted(best.values(),key=lambda item:item['distance'])
        observed_candidates=[(d,i,r,'learned') for d,i,r in observed]
        if observed:
            dist,idx,record=observed[0]
            second=next((d for d,i,_ in observed[1:] if i!=idx),float('inf'))
            if dist<65 and dist<.25*second:
                return {'name':record['name'],'variant':record['variant'],'index':idx,'distance':dist,
                        'second_distance':second,'second_name':'','source':'observed',
                        'candidates':summarize_candidates(observed_candidates)}
        indices=[]; choices=[]
        for i,record in enumerate(self.records):
            idx=choice_for(record['name'])
            if idx is not None:
                indices.append(i);choices.append(idx)
        if not indices:
            if not observed:return None
            dist,idx,record=observed[0]
            return {'name':record['name'],'variant':record['variant'],'index':idx,'distance':dist,
                    'second_distance':next((d for d,i,_ in observed[1:] if i!=idx),float('inf')),
                    'second_name':'','source':'observed_weak','covered_choices':len({i for _,i,_ in observed}),
                    'candidates':summarize_candidates(observed_candidates)}
        scores=np.nanmean((self.features[indices]-feature)**2,axis=1)
        scores=np.where(np.isfinite(scores),scores,np.inf)
        order=np.argsort(scores)
        if not len(order) or not np.isfinite(scores[order[0]]):return None
        candidates=summarize_candidates(observed_candidates+[(float(scores[j]),choices[j],self.records[indices[j]],'wiki') for j in range(len(indices))])
        first=int(order[0]); idx=choices[first];record=self.records[indices[first]]
        runner=next((int(i) for i in order[1:] if choices[int(i)]!=idx),None)
        if observed and observed[0][0] < scores[first]:
            dist, observed_idx, observed_record=observed[0]
            competitors=[(d,r['name']) for d,i,r in observed[1:] if i!=observed_idx]
            competitors += [(float(scores[j]),self.records[indices[j]]['name']) for j in order if choices[int(j)]!=observed_idx]
            second_dist,second_name=min(competitors,default=(float('inf'),''))
            return {'name':observed_record['name'],'variant':observed_record['variant'],
                    'index':observed_idx,'distance':dist,'second_distance':second_dist,
                    'second_name':second_name,'source':'observed_weak',
                    'covered_choices':len(set(choices)|{i for _,i,_ in observed}),'candidates':candidates}
        return {'name':record['name'],'variant':record['variant'],'index':idx,
                'distance':float(scores[first]),'second_distance':float(scores[runner]) if runner is not None else float('inf'),
                'second_name':self.records[indices[runner]]['name'] if runner is not None else '',
                'covered_choices':len(set(choices)),'source':'wiki','candidates':candidates}

class QuestionReader:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR()
        self._quick_native_ocr = None
        self.prompt = cv2.imread(str(ROOT / 'assets' / 'question_prompt.png'), cv2.IMREAD_GRAYSCALE)
        self.results = cv2.imread(str(ROOT / 'assets' / 'event_title.png'), cv2.IMREAD_GRAYSCALE)
        self.replay = cv2.imread(str(ROOT / 'assets' / 'replay_button.png'), cv2.IMREAD_GRAYSCALE)
        if any(x is None for x in (self.prompt, self.results, self.replay)):
            raise RuntimeError('Event screen template missing')

    def is_question(self, frame):
        x1,y1,x2,y2 = HEADER
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if crop.shape != self.prompt.shape:
            return False
        return float(cv2.matchTemplate(crop, self.prompt, cv2.TM_CCOEFF_NORMED)[0, 0]) >= .83

    def is_replay(self, frame):
        # The results heading changes to "New personal best" on record runs;
        # the event title remains stable in both cases.
        header = cv2.cvtColor(frame[75:110,335:500], cv2.COLOR_BGR2GRAY)
        button = cv2.cvtColor(frame[370:396,424:536], cv2.COLOR_BGR2GRAY)
        if header.shape != self.results.shape or button.shape != self.replay.shape:
            return False
        a = float(cv2.matchTemplate(header, self.results, cv2.TM_CCOEFF_NORMED)[0, 0])
        b = float(cv2.matchTemplate(button, self.replay, cv2.TM_CCOEFF_NORMED)[0, 0])
        return a >= .83 and b >= .83

    def options(self, frame):
        crops = []
        for x,y in CHOICES:
            crop = frame[y+10:y+34, x+4:x+160]
            crops.append(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
        results, _ = self.ocr.text_recognizer(crops)
        return [(str(text), float(confidence)) for text, confidence in results]

    def reread_label(self, crop, name):
        """Check for a known label using a tight crop of the dark letter outlines."""
        height, width = crop.shape[:2]
        inside = crop[round(.13 * height):round(.89 * height),
                      round(.10 * width):round(.90 * width)]
        if inside.size == 0:
            return None
        gray = cv2.cvtColor(inside, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(cv2.inRange(gray, 0, 90))
        if not (3 <= h < .72 * gray.shape[0] and 4 <= w < .90 * gray.shape[1]):
            return None
        pad = max(2, round(.035 * height))
        text = gray[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
        variants = [text, cv2.equalizeHist(text), cv2.bitwise_not(text)]
        crops = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in variants]
        reads, _ = self.ocr.text_recognizer(crops)
        spellings = _catalog_spellings(name)
        return max(((str(value), float(confidence)) for value, confidence in reads
                    if confidence >= .70 and _ocr_spelling(value) in spellings),
                   key=lambda read: read[1], default=None)

    def options_native(self, crops, *, quick=False):
        """Read buttons at their native size, including wrapped two-line labels."""
        ocr = self.ocr
        if quick:
            if self._quick_native_ocr is None:
                from rapidocr_onnxruntime import RapidOCR
                self._quick_native_ocr = RapidOCR(det_limit_side_len=256, det_model_path=None)
            ocr = self._quick_native_ocr
        output = []
        for crop in crops:
            image = cv2.resize(crop,None,fx=1.4,fy=1.4,interpolation=cv2.INTER_CUBIC)
            lines, _ = ocr(image, text_score=.35)
            if not lines:
                h,w=crop.shape[:2]
                center=crop[round(.11*h):round(.90*h),round(.04*w):round(.96*w)]
                direct,_=ocr.text_recognizer([cv2.resize(center,None,fx=3,fy=3)]) if center.size else ([],None)
                text,confidence=direct[0] if direct else ('',0.)
                output.append((str(text),float(confidence)) if confidence>=.7 else ('',0.))
                continue
            ordered = sorted(lines,key=lambda row: min(p[1] for p in row[0]))
            text=' '.join(str(row[1]) for row in ordered)
            confidence=sum(float(row[2]) for row in ordered)/len(ordered)
            if len(ordered)==1:
                h,w=crop.shape[:2]
                areas=[crop[round(.11*h):round(.51*h),round(.04*w):round(.96*w)],
                       crop[round(.49*h):round(.90*h),round(.04*w):round(.96*w)],
                       crop[round(.11*h):round(.90*h),round(.04*w):round(.96*w)]]
                if all(area.size for area in areas):
                    # Both halves are wider relative to their height than the
                    # full crop, so removing it leaves batch padding unchanged.
                    values,_=ocr.text_recognizer([cv2.resize(area,None,fx=3,fy=3) for area in areas[:2]])
                    if len(values)==2:
                        (top,tc),(bottom,bc)=values
                        if (tc>=.75 and bc>=.75 and normalize_text(top) and normalize_text(bottom) and
                                max(SequenceMatcher(None,normalize_text(text),normalize_text(part)).ratio()
                                    for part in (top,bottom))>=.8):
                            text=f'{top} {bottom}';confidence=(float(tc)+float(bc))/2
                        elif confidence<.7:
                            separate,_=ocr.text_recognizer([cv2.resize(areas[2],None,fx=3,fy=3)])
                            if separate:
                                full,fc=separate[0]
                                if fc>=.7 and fc>confidence+.1:text=str(full);confidence=float(fc)
            output.append((text,confidence))
        return output

def answer_compact(frame, catalog, reader, crops, *, allow_guess=False, still_current=None):
    fast=answer_for(frame,catalog,reader)
    if fast['status'] == 'waiting' or (
            fast['status'] == 'matched' and fast['distance'] < 150
            and fast['distance'] < .15 * fast['second_distance']):
        return fast
    return answer_with_native(frame, catalog, reader, crops, fast,
                              allow_guess=allow_guess, still_current=still_current)


def _clear_guess_choice(result, options):
    match = result.get('match') or {}
    index = match.get('index')
    distance = match.get('distance', float('inf'))
    runner = match.get('second_distance', float('inf'))
    name = match.get('name')
    if (type(index) is not int or not 0 <= index < 4 or len(options) != 4
            or not name or match.get('covered_choices', 0) < 3
            or not np.isfinite(runner) or not 0 <= distance < runner
            or distance > .5 * runner):
        return None
    text, confidence = options[index]
    similarity = label_similarity(name, text)
    alternative = max(label_similarity(name, option)
                      for i, (option, _) in enumerate(options) if i != index)
    if confidence >= .55 and similarity >= .65 and similarity - alternative >= .12:
        return index, normalize_text(name)
    return None


def recover_visual_label(frame, catalog, reader, crops, fast):
    """Re-read a garbled button only when the card strongly matches known art."""
    options = fast.get('options', ())
    if fast['status'] != 'unknown' or len(options) != 4 or len(crops) != 4:
        return None
    suspects = [i for i, (_, confidence) in enumerate(options) if confidence < .70]
    if not 1 <= len(suspects) <= 2:
        return None
    x1, y1, x2, y2 = CARD
    visual = catalog.find(frame[y1:y2, x1:x2])
    if visual is None:
        return None
    limit, margin = (12, .05) if visual.get('source') == 'observed' else (100, .12)
    if (not 0 <= visual['distance'] < limit
            or not visual['distance'] < margin * visual['second_distance']):
        return None
    name = visual['name']
    if any(confidence >= .50 and label_similarity(name, text) >= .80
           for text, confidence in options):
        return None
    names = {record['name'] for record in catalog.records}
    names.update(record['name'] for record, _ in catalog.observed)
    exact_labels = {label for card_name in names for label in _catalog_spellings(card_name)}
    verified = []
    for index in suspects:
        if _ocr_spelling(options[index][0]) in exact_labels:
            continue
        read = reader.reread_label(crops[index], name)
        if read is not None:
            verified.append((index, read))
    if len(verified) != 1:
        return None
    index, read = verified[0]
    repaired = list(options)
    repaired[index] = read
    match = answer_for(frame, catalog, reader, repaired)
    if (match['status'] == 'matched' and match['index'] == index
            and normalize_text(match['name']) == normalize_text(name)):
        return match
    return None


def answer_with_native(frame, catalog, reader, crops, fast, *,
                       allow_guess=False, still_current=None):
    """Try lower-resolution OCR before rerunning full-size text detection."""
    canonical = fast.get('options', ())

    def match_reads(reads):
        options = recover_missing_options(reader, frame, reads, canonical=canonical)
        if options == canonical:
            return fast
        return answer_for(frame, catalog, reader, options)

    if still_current is not None and not still_current():
        return fast
    repaired = recover_visual_label(frame, catalog, reader, crops, fast)
    if repaired is not None:
        return repaired
    quick_reads = reader.options_native(crops, quick=True)
    if still_current is not None and not still_current():
        return fast
    quick = match_reads(quick_reads)
    if (quick['status'] == 'matched'
            and quick['match'].get('covered_choices', 0) >= 2
            and quick['distance'] < .18 * quick['second_distance']):
        return quick
    if allow_guess and fast['status'] == 'unknown':
        previous = _clear_guess_choice(fast, canonical)
        if previous is not None and previous == _clear_guess_choice(quick, quick_reads):
            return best_effort_answer(quick)
    return prefer_answer(fast, match_reads(reader.options_native(crops)))


def prefer_answer(first, second):
    def evidence(result):
        match = result.get('match') or {}
        return (result.get('status') == 'matched',
                sum(bool(text) and confidence >= .5 for text, confidence in result.get('options', ())),
                -match.get('distance', float('inf')))
    return max((first, second), key=evidence)

def answer_for(frame, catalog, reader, options_override=None):
    if not reader.is_question(frame):
        return {'status': 'waiting', 'reason': 'Waiting for an active question.'}
    x1,y1,x2,y2 = CARD
    if float(np.std(frame[y1:y2, x1:x2])) < 2:
        return {'status':'waiting','reason':'Waiting for card image to load.'}
    options = (reader.options(frame) if options_override is None else
               options_override() if callable(options_override) else options_override)
    if len(options) != 4:
        return {'status': 'unknown', 'reason': 'Could not read four choices.'}
    match = catalog.find_for_choices(frame[y1:y2, x1:x2],options)
    if match is None:
        return {'status': 'unknown', 'reason': 'No catalog images match the visible choices.', 'options':options}
    if match.get('source') == 'observed_weak':
        return {'status':'unknown','reason':'Learned card image is uncertain.','match':match,'options':options}
    if match['distance'] > 550:
        return {'status': 'unknown', 'reason': 'Card image is uncertain.', 'match': match,'options':options}
    if match.get('source') != 'observed' and match.get('covered_choices',0)<2 and match['distance']>150:
        return {'status': 'unknown', 'reason': 'Too few choices have comparable images.', 'match':match,'options':options}
    # SequenceMatcher: https://docs.python.org/3/library/difflib.html
    ranked = sorted(((label_similarity(match['name'], text), i, text, confidence)
                     for i, (text, confidence) in enumerate(options)), reverse=True)
    best, runner = ranked[0], ranked[1]
    if best[1] != match['index']:
        return {'status':'unknown','reason':'Answer text is uncertain.','match':match,'options':options}
    strong_image = match['distance'] < 150 and match['distance'] < .15 * match['second_distance']
    text_threshold = .53 if strong_image else .65
    text_margin = .20 if strong_image else .12
    if best[0] < text_threshold or best[0] - runner[0] < text_margin or best[3] < .50:
        return {'status': 'unknown', 'reason': 'Answer text is uncertain.', 'match': match,
                'options': options}
    if match['distance'] > .55 * match['second_distance']:
        # If two pictures are similar, only accept a clear label that doesn't
        # match the visually competing card anywhere among the four choices.
        competitor = max((label_similarity(match['second_name'], text) for text, _ in options), default=0.)
        if match['distance'] > 200 or best[0] < .82 or best[0]-runner[0] < .20 or competitor >= .70:
            return {'status': 'unknown', 'reason': 'Card image is uncertain.', 'match': match, 'options': options}
    return {'status': 'matched', 'name': match['name'], 'variant': match['variant'],
            'index': best[1], 'ocr': best[2], 'distance': match['distance'],
            'second_distance': match['second_distance'], 'options': options,'match':match}
