"""Find an active question in a compact Roblox client using visible UI shapes."""
import itertools
from difflib import SequenceMatcher
import cv2
import numpy as np
from matcher import normalize_text

CANONICAL_CHOICES = [(448,228),(616,228),(448,276),(616,276)]

def give_up_button(image, reader):
    """Locate the free Give up button on the Continue screen, never Continue."""
    height,width=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray=cv2.inRange(hsv,(0,0,145),(179,55,245))
    contours,_=cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if not (.08*width < w < .5*width and .025*height < h < .13*height and
                .5*height < y < .9*height and abs(x+w/2-width/2) < .08*width and
                cv2.contourArea(contour) > .5*w*h):
            continue
        inner=image[y+round(.15*h):y+round(.85*h),x+round(.08*w):x+round(.92*w)]
        texts,_=reader.ocr.text_recognizer([cv2.resize(inner,None,fx=3,fy=3)])
        if not texts: continue
        value,confidence=texts[0]
        if float(confidence)<.5 or SequenceMatcher(None,normalize_text(value),'giveup').ratio()<.75:
            continue
        # The distinctive orange heading provides context for this common label.
        heading=image[max(0,y-7*h):max(0,y-5*h),x:x+w]
        lines,_=reader.ocr(heading,text_score=.35)
        if any(float(line[2])>=.5 and SequenceMatcher(None,normalize_text(str(line[1])),'continue').ratio()>=.85
               for line in lines or []):
            return (round(x+w/2),round(y+h/2))
    return None

def compact_question(image, reader):
    """Return a canonical frame plus native button boxes, or None when uncertain."""
    height,width = image.shape[:2]
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray_buttons = cv2.inRange(hsv,(0,0,145),(179,55,245))
    contours,_ = cv2.findContours(gray_buttons,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (x >= .3*width and y >= .15*height and w >= .1*width and h >= .035*height
                and 1.3 < w/h < 6 and cv2.contourArea(c) > w*h*.5):
            boxes.append((x,y,w,h))
    if len(boxes) > 20:
        return None
    grid=None
    for group in itertools.combinations(boxes,4):
        group=sorted(group,key=lambda v:(v[1],v[0]))
        top=sorted(group[:2],key=lambda v:v[0]);bottom=sorted(group[2:],key=lambda v:v[0])
        a,b=top;c,d=bottom
        avg_w=np.mean([v[2] for v in group]);avg_h=np.mean([v[3] for v in group])
        if (max(abs(v[2]-avg_w) for v in group) > .08*avg_w or
            max(abs(v[3]-avg_h) for v in group) > .08*avg_h): continue
        if (abs(a[1]-b[1]) > .12*avg_h or abs(c[1]-d[1]) > .12*avg_h or
            abs(a[0]-c[0]) > .10*avg_w or abs(b[0]-d[0]) > .10*avg_w or
            not .9*avg_h < c[1]-a[1] < 1.6*avg_h or
            not .9*avg_w < b[0]-a[0] < 1.4*avg_w): continue
        grid=top+bottom;break
    if grid is None:return None
    x,y,w,h=grid[0];right=grid[1][0]+grid[1][2]
    top=max(0,round(y-.50*h));bottom=max(top+1,round(y-.10*h))
    prompt=image[top:bottom,x:right]
    if prompt.size==0:return None
    # Recognize the event's actual prompt before trusting these generic shapes.
    prompt_img=cv2.resize(prompt,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    result,_=reader.ocr.text_recognizer([prompt_img])
    if not result:return None
    text,confidence=result[0]
    if float(confidence)<.5 or SequenceMatcher(None,normalize_text(text),'choosethematchingcard').ratio()<.75:
        return None
    orange=cv2.inRange(hsv,(5,100,140),(35,255,255))
    orange=cv2.morphologyEx(orange,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(orange,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cards=[]
    for c in contours:
        cx,cy,cw,ch=cv2.boundingRect(c)
        if (width*.10 < cw < width*.5 and height*.2 < ch < height*.8 and
                .45 < cw/ch < 1.05 and cx+cw < x and cy < y and cy+ch > grid[2][1]):
            cards.append((cx,cy,cw,ch))
    if not cards:return None
    cx,cy,cw,ch=max(cards,key=lambda box:box[2]*box[3])
    card=image[cy:cy+ch,cx:cx+cw]
    canonical=np.zeros((504,960,3),dtype=np.uint8)
    canonical[124:404,207:418]=cv2.resize(card,(211,280),interpolation=cv2.INTER_AREA)
    canonical[207:225,449:782]=cv2.cvtColor(reader.prompt,cv2.COLOR_GRAY2BGR)
    crops=[];clicks=[]
    for box,(target_x,target_y) in zip(grid,CANONICAL_CHOICES):
        bx,by,bw,bh=box;crop=image[by:by+bh,bx:bx+bw];crops.append(crop)
        canonical[target_y:target_y+44,target_x:target_x+164]=cv2.resize(crop,(164,44),interpolation=cv2.INTER_AREA)
        clicks.append((round(bx+bw/2),round(by+bh/2)))
    return {'frame':canonical,'choice_crops':crops,'clicks':clicks,'boxes':grid,'card_box':(cx,cy,cw,ch)}

def compact_replay(image, reader):
    """Locate Play again only when the event title is also visibly recognized."""
    height,width=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,(35,35,80),(95,255,255))
    contours,_=cv2.findContours(green,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if (y > .55*height and .15*width < w < .5*width and .03*height < h < .15*height
                and abs((x+w/2)-width/2) < .15*width and cv2.contourArea(c) > .55*w*h):
            candidates.append((x,y,w,h))
    for x,y,w,h in candidates:
        inner=image[y+round(.22*h):y+round(.78*h),x+round(.05*w):x+round(.95*w)]
        if inner.size==0:continue
        texts,_=reader.ocr.text_recognizer([cv2.resize(inner,None,fx=3,fy=3)])
        if not texts:continue
        value,conf=texts[0]
        if float(conf)<.5 or SequenceMatcher(None,normalize_text(value),'playagain').ratio()<.82:continue
        heading=image[:round(.24*height),:]
        lines,_=reader.ocr(heading,text_score=.35)
        if any(float(line[2])>=.5 and SequenceMatcher(None,normalize_text(str(line[1])),'guessthecard').ratio()>=.8
               for line in lines or []):
            return (round(x+w/2),round(y+h/2))
    return None
