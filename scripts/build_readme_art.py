"""Build editable SVG diagrams and matching high-resolution PNGs for the README.

Requires Pillow only for PNG previews. No network resources or model calls.
"""

from pathlib import Path
from html import escape
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'assets'
ROOT.mkdir(parents=True, exist_ok=True)
INK, MUTED, TEAL, PURPLE = '#15243b', '#65758e', '#089e91', '#7860df'
FONT_ROOT = Path('C:/Windows/Fonts')


class Figure:
    def __init__(self, width, height, background):
        self.width, self.height = width, height
        self.image = Image.new('RGB', (width * 2, height * 2), background)
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">']
        self.rect(0, 0, width, height, background)

    def rect(self, x, y, w, h, fill, radius=0, stroke=None):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}"' + (f' stroke="{stroke}"' if stroke else '') + '/>')
        self.draw.rounded_rectangle((x*2, y*2, (x+w)*2, (y+h)*2), radius=radius*2, fill=fill, outline=stroke, width=2)

    def text(self, x, y, value, size=22, color=INK, bold=False):
        self.svg.append(f'<text x="{x}" y="{y+size}" font-family="Segoe UI,DejaVu Sans,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')
        filename = 'segoeuib.ttf' if bold else 'segoeui.ttf'
        candidate = FONT_ROOT / filename
        if not candidate.exists():
            candidate = Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
        font = ImageFont.truetype(str(candidate), size*2)
        self.draw.text((x*2, (y+size)*2), value, font=font, fill=color, anchor='ls')

    def line(self, points, color, width=2, arrow=False):
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
        self.draw.line([(x*2,y*2) for x,y in points], fill=color, width=width*2)
        if arrow:
            x,y=points[-1]; px,py=points[-2]; angle=math.atan2(y-py,x-px)
            tri=[(x,y),(x-10*math.cos(angle-.5),y-10*math.sin(angle-.5)),(x-10*math.cos(angle+.5),y-10*math.sin(angle+.5))]
            self.svg.append(f'<polygon points="{" ".join(f"{a},{b}" for a,b in tri)}" fill="{color}"/>')
            self.draw.polygon([(a*2,b*2) for a,b in tri],fill=color)

    def save(self, name):
        (ROOT/f'{name}.svg').write_text('\n'.join(self.svg+['</svg>']),encoding='utf-8')
        self.image.save(ROOT/f'{name}.png',optimize=True)


def hero():
    f=Figure(1280,420,'#101b30')
    for x in range(760,1280,40): f.line([(x,0),(x,420)],'#1a2941',1)
    for y in range(20,420,40): f.line([(760,y),(1280,y)],'#1a2941',1)
    f.rect(64,55,262,32,'#1c3445',16)
    f.text(81,60,'VERIFIABLE REASONING',15,'#77e1ce',True)
    f.text(59,106,'VeriGate',83,'#ffffff',True)
    f.text(65,222,'Verifier-Gated On-Policy Distillation',29,'#dce6f7')
    f.text(65,272,'Learn from the teacher. Stay grounded in correctness.',20,'#99acc8')
    for x,w,label in [(65,174,'ON-POLICY'),(251,170,'TOKEN-LEVEL'),(433,179,'VERIFIER-GATED')]:
        f.rect(x,343,w,33,'#1b2a43',16)
        f.text(x+17,349,label,13,'#c0cfe4',True)
    f.rect(866,67,310,286,'#17263d',26, '#304461')
    f.text(893,86,'THE CORRECTNESS GATE',16,'#b3c3da',True)
    for i,h in enumerate([34,66,46,83,54,70]):
        f.rect(896+i*40,218-h,22,h,'#67d9c1',5)
        f.rect(896+i*40,236,22,[35,16,44,22,40,20][i],'#a58ced',5)
    f.line([(891,227),(1147,227)],'#49617d',2)
    f.text(896,305,'TEACHER SIGNAL  +  VERIFICATION',12,'#c5d4e9',True)
    f.save('hero')


def method():
    f=Figure(1280,650,'#f5f8fc')
    f.text(46,26,'01  /  THE TRAINING LOOP',15,TEAL,True)
    f.text(46,58,'Two signals. One gated reward.',37,INK,True)
    f.text(46,111,'Token-level guidance meets response-level correctness.',20,MUTED)
    # Edges are drawn before cards so endpoints remain clean.
    f.line([(279,319),(344,319)],'#90a0b7',3,True)
    f.line([(599,319),(632,319),(632,228),(683,228)],PURPLE,3,True)
    f.line([(632,319),(632,415),(683,415)],TEAL,3,True)
    f.line([(925,228),(960,228),(960,319),(999,319)],PURPLE,3,True)
    f.line([(925,415),(960,415),(960,319)],TEAL,3)
    cards=[(46,250,234,138,'#ffffff','01','Reasoning prompt',['Task + answer','verification rule'],INK),
           (344,250,255,138,'#ffffff','02','Student rollout',['Sample a response','Record token log probs'],INK),
           (683,168,242,132,'#f0ecff','03A','Teacher guidance',['Score sampled tokens','Compute log-prob ratio'],PURPLE),
           (683,354,242,132,'#e6f5f1','03B','Task verifier',['Check the final answer','Return outcome score'],TEAL),
           (999,250,234,138,'#16253d','04','Reward gate',['Keep rewards aligned','with correctness'],'#ffffff')]
    for x,y,w,h,bg,n,title,lines,accent in cards:
        f.rect(x,y,w,h,bg,16,'#dbe3ef' if bg=='#ffffff' else None)
        f.text(x+20,y+13,n,14,accent,True)
        f.text(x+20,y+40,title,22,accent,True)
        for i,line in enumerate(lines): f.text(x+20,y+77+i*23,line,16,'#b9cbe1' if bg=='#16253d' else MUTED)
    f.line([(1115,388),(1115,545),(470,545),(470,398)],'#90a0b7',3,True)
    f.rect(640,522,356,46,'#f5f8fc',12)
    f.text(660,532,'Policy update with gated token rewards',17,INK,True)
    f.text(46,598,'OPTIONAL: group-relative scaling multiplies gated rewards by |A| + u before the update.',17,MUTED)
    f.save('method')


def gate():
    f=Figure(1280,552,'#f5f8fc')
    f.text(46,26,'02  /  REWARD GEOMETRY',15,TEAL,True)
    f.text(46,56,'Keep the signal that agrees with the outcome.',33,INK,True)
    f.text(46,109,'Illustrative token rewards — schematic values, not experimental measurements.',18,MUTED)
    for x,title,subtitle,color,correct in [(46,'Correct response','r = max(d, 0)',TEAL,True),(665,'Incorrect response','r = min(d, 0)',PURPLE,False)]:
        f.rect(x,163,570,324,'#ffffff',18,'#dbe3ef')
        f.rect(x+23,184,7,52,color,3)
        f.text(x+43,179,title,24,INK,True)
        f.text(x+43,216,subtitle,18,color,True)
        axis=346
        f.line([(x+35,axis),(x+534,axis)],'#ccd6e4',2)
        f.text(x+14,axis-12,'0',14,MUTED)
        values=[48,-42,73,-29,35,-60,60,-38]
        for i,v in enumerate(values):
            bx=x+53+i*59
            f.rect(bx,axis-v if v>0 else axis,28,abs(v),'#e6ebf3',4)
            kept=v>0 if correct else v<0
            if kept: f.rect(bx,axis-v if v>0 else axis,28,abs(v),color,4)
            else: f.line([(bx,axis),(bx+28,axis)],color,4)
        f.text(x+29,440,'Positive rewards retained' if correct else 'Negative rewards retained',18,color,True)
    f.rect(47,513,14,14,'#e6ebf3',3)
    f.text(70,508,'Removed by the gate',15,MUTED)
    f.rect(290,513,14,14,TEAL,3)
    f.text(314,508,'Retained reward',15,MUTED)
    f.text(665,508,'d = teacher log probability − student log probability',15,MUTED)
    f.save('reward-gate')


if __name__=='__main__':
    hero(); method(); gate()
    print('Built three SVG diagrams and PNG renders in',ROOT)
