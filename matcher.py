"""Screen-only recognition for the four-choice Guess the Card event."""
import hashlib
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

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

def recover_missing_options(reader, frame, options):
    if len(options)!=4 or all(text and confidence >= .3 for text,confidence in options):
        return options
    alternate=reader.options(frame)
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

def label_similarity(name, text):
    observed = normalize_text(canonical_learned_name(text))
    # Wiki skin entries sometimes append a bundle or form description that
    # the game does not put on its answer button.
    short = re.sub(r'\s*\([^)]*\)', '', name).strip()
    return max(SequenceMatcher(None, normalize_text(candidate), observed).ratio()
               for candidate in (name, short) if normalize_text(candidate))

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
    if isinstance(idx,int) and 0<=idx<4 and distance<=950:
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
        names={r['name'] for r in self.records}|{r['name'] for r,_ in self.observed}
        best_text=[max((similarity(name,text) for name in names),default=0.) if text else 0.
                   for text,conf in options]
        @lru_cache(maxsize=None)
        def choice_for(name):
            ranked=sorted(((similarity(name,text),i) for i,(text,conf) in enumerate(options)
                           if text and conf>=(.3 if relaxed else .5)
                           and similarity(name,text)>=best_text[i]-.12), reverse=True)
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

    def options_native(self, crops):
        """Read buttons at their native size, including wrapped two-line labels."""
        output = []
        for crop in crops:
            image = cv2.resize(crop,None,fx=1.4,fy=1.4,interpolation=cv2.INTER_CUBIC)
            lines, _ = self.ocr(image, text_score=.35)
            if not lines:
                h,w=crop.shape[:2]
                center=crop[round(.11*h):round(.90*h),round(.04*w):round(.96*w)]
                direct,_=self.ocr.text_recognizer([cv2.resize(center,None,fx=3,fy=3)]) if center.size else ([],None)
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
                    values,_=self.ocr.text_recognizer([cv2.resize(area,None,fx=3,fy=3) for area in areas])
                    if len(values)==3:
                        (top,tc),(bottom,bc),(full,fc)=values
                        if (tc>=.75 and bc>=.75 and normalize_text(top) and normalize_text(bottom) and
                                max(SequenceMatcher(None,normalize_text(text),normalize_text(part)).ratio()
                                    for part in (top,bottom))>=.8):
                            text=f'{top} {bottom}';confidence=(float(tc)+float(bc))/2
                        elif confidence<.7:
                            separate,_=self.ocr.text_recognizer([cv2.resize(areas[2],None,fx=3,fy=3)])
                            if separate:
                                full,fc=separate[0]
                                if fc>=.7 and fc>confidence+.1:text=str(full);confidence=float(fc)
            output.append((text,confidence))
        return output

def answer_compact(frame, catalog, reader, crops):
    fast=answer_for(frame,catalog,reader)
    if (fast['status']=='matched' and fast.get('distance',float('inf'))<150 and
            fast['distance'] < .15*fast.get('second_distance',0)):
        return fast
    native = answer_for(frame,catalog,reader,recover_missing_options(reader,frame,reader.options_native(crops)))
    return prefer_answer(fast, native)


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
