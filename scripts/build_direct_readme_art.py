"""Rebuild README artwork locally: python scripts/build_direct_readme_art.py.

Requires Pillow and matplotlib. SVG equations contain paths, not runtime TeX.
"""
from io import StringIO
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_readme_art import Figure

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'assets'
NAVY, PANEL, WHITE, MUTED, TEAL, BLUE = '#0b1220', '#142235', '#f3f7ff', '#a6b6cd', '#65e0c2', '#a5a0ff'


def cover():
    f = Figure(1280, 560, NAVY)
    for x in range(760, 1280, 40):
        f.line([(x, 0), (x, 455)], '#172437', 1)
    for y in range(15, 455, 40):
        f.line([(760, y), (1280, y)], '#172437', 1)
    f.rect(56, 51, 5, 22, TEAL, 2)
    f.text(76, 49, 'VERIGATE  /  ON-POLICY DISTILLATION', 16, TEAL, True)
    f.text(50, 106, 'VeriGate', 94, WHITE, True)
    f.text(57, 234, 'Where to distill.', 40, WHITE, True)
    f.text(58, 301, 'Student contexts. Teacher distributions.', 23, MUTED)
    f.text(58, 338, 'Supervision, deliberately allocated.', 23, MUTED)
    f.text(823, 66, 'DISAGREEMENT → ALLOCATION', 15, TEAL, True)
    vals = [56, 95, 74, 140, 112, 181, 157, 220]
    for i, h in enumerate(vals):
        x = 817 + 46 * i
        f.rect(x, 353-h, 25, h, TEAL if i % 2 else BLUE, 6)
        f.rect(x, 374, 25, 7, '#34465f', 3)
    f.text(823, 409, 'Illustrative allocation · unit coefficient sum', 14, MUTED)
    f.line([(56, 470), (1224, 470)], '#2a3b50', 1)
    for x, n, title in [(58, '01', 'FULL-VOCABULARY KL'), (452, '02', 'ORDINAL WEIGHTS'), (846, '03', 'CONTROLLED BASELINES')]:
        f.text(x, 501, n, 16, TEAL, True)
        f.text(x+38, 501, title, 16, WHITE, True)
    f.save('direct-opd-cover')


def flow():
    f = Figure(1280, 350, NAVY)
    f.text(42, 30, 'THE LEARNING LOOP', 17, TEAL, True)
    f.text(42, 62, 'Fresh contexts. One direct objective.', 31, WHITE, True)
    cards = [
        ('01 / GENERATE', 'Student rollouts', 'Sample fresh responses', 'from the current student.'),
        ('02 / COMPARE', 'Full-vocabulary KL', 'Evaluate teacher and student', 'on the same sampled prefixes.'),
        ('03 / ALLOCATE', 'Rank + uniform mix', 'Detach scores and weights;', 'normalize each response.'),
        ('04 / UPDATE', 'Direct distillation', 'Backpropagate weighted KL', 'through the student only.'),
    ]
    for i, (label, title, a, b) in enumerate(cards):
        x = 42 + i*307
        f.rect(x, 127, 274, 153, PANEL, 12, '#2a3b50')
        f.text(x+18, 145, label, 14, TEAL, True)
        f.text(x+18, 180, title, 22, WHITE, True)
        f.text(x+18, 220, a, 16, MUTED)
        f.text(x+18, 245, b, 16, MUTED)
        if i < 3:
            f.line([(x+280, 203), (x+301, 203)], TEAL, 2, arrow=True)
    f.text(42, 306, 'Repeat after every optimizer update', 16, MUTED)
    f.text(750, 306, 'Answer verification runs separately, offline.', 16, TEAL)
    f.save('direct-opd-flow')


def objective():
    plt.rcParams['svg.fonttype'] = 'path'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    fig = plt.figure(figsize=(12.8, 2.8), facecolor='#f4f7fc')
    fig.text(.04, .82, 'THE OBJECTIVE', color='#39756d', size=12, weight='bold')
    fig.text(.04, .54, r'$D_{it}=D_{\mathrm{KL}}\left(p_T(\cdot\mid s_{it})\,\Vert\,p_\theta(\cdot\mid s_{it})\right)$', color=NAVY, size=23)
    fig.text(.04, .22, r'$\mathcal{L}=\frac{1}{N}\sum_i\sum_t\mathrm{sg}(w_{it})D_{it},\qquad w_{it}\geq 0,\quad\sum_t w_{it}=1$', color=NAVY, size=23)
    fig.savefig(ROOT/'direct-opd-objective.png', dpi=200, facecolor=fig.get_facecolor())
    stream = StringIO()
    fig.savefig(stream, format='svg', facecolor=fig.get_facecolor())
    svg = re.sub(r'<metadata>.*?</metadata>', '', stream.getvalue(), flags=re.S)
    (ROOT/'direct-opd-objective.svg').write_text(svg, encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    cover()
    flow()
    objective()
    print('Built cover, learning loop, and path-based objective (SVG + PNG).')
