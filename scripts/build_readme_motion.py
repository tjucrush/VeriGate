"""Build deterministic GIF schematics and a chart of recorded measurements.

Run from the repository root: python scripts/build_readme_motion.py
Requires Pillow and matplotlib. Animation positions are illustrative, not data.
"""
import json
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_readme_art import Figure

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
BG, PANEL, INK, MUTED, TEAL, VIOLET = '#0b1220', '#142235', '#f3f7ff', '#a6b6cd', '#65e0c2', '#a5a0ff'


def animate(name, painter):
    frames = []
    for n in range(32):
        f = painter(n / 32)
        if n == 0:
            f.save(name)
        frames.append(f.image.resize((1080, round(f.height * 1080 / f.width)), getattr(Image, 'Resampling', Image).LANCZOS))
    palette = frames[0].quantize(colors=128)
    frames = [im.quantize(palette=palette, dither=getattr(Image, 'Dither', Image).NONE) for im in frames]
    frames[0].save(ASSETS / (name + '.gif'), save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True, disposal=1)


def hero(phase):
    f = Figure(1280, 460, BG)
    for x in range(768, 1280, 40):
        f.line([(x, 0), (x, 460)], '#172437', 1)
    f.text(54, 38, 'VERIFIABLE FEEDBACK / ON-POLICY LEARNING', 16, TEAL, True)
    f.text(48, 91, 'VeriGate', 84, INK, True)
    f.text(54, 205, 'Learn. Verify. Distill.', 38, INK, True)
    f.text(54, 269, 'VeriOPD + VeriGRPD', 26, VIOLET, True)
    f.text(54, 319, 'Teacher guidance meets verifiable feedback.', 21, MUTED)
    f.text(54, 400, 'TWO METHODS    /    SIX BENCHMARKS    /    OPEN EXPERIMENT RECORDS', 14, TEAL, True)
    f.rect(808, 107, 370, 234, PANEL, 18, '#2a3b50')
    f.text(835, 126, 'SUPERVISION IN MOTION', 17, TEAL, True)
    for row in range(3):
        y = 198 + 43 * row
        f.line([(839, y), (1147, y)], '#34465f', 2)
        for k in range(5):
            x = 839 + ((phase + k / 5 + row / 12) % 1) * 306
            f.rect(x, y-5, 10, 10, TEAL if row != 1 else VIOLET, 5)
    f.text(823, 366, 'Illustrative signal flow', 17, MUTED)
    return f


def gate(phase):
    f = Figure(1280, 368, BG)
    f.text(38, 25, 'VERIFIER-GUIDED DISTILLATION', 17, TEAL, True)
    f.text(38, 62, 'Teacher evidence, filtered by the outcome.', 31, INK, True)
    for x, title, subtitle in [(38, 'Teacher evidence', 'Signed token-level signal'), (472, 'Verifier gate', 'Filter by response correctness'), (906, 'Retained signal', 'Guide the policy update')]:
        f.rect(x, 125, 334, 168, PANEL, 12, '#2a3b50')
        f.text(x+18, 142, title, 23, INK, True)
        f.text(x+18, 180, subtitle, 16, MUTED)
    for row in range(2):
        y = 234 + row * 32
        for k in range(8):
            x = 75 + k * 35
            color = TEAL if (k+row) % 2 == 0 else VIOLET
            f.rect(x, y, 18, 10, color, 4)
            if (k+row) % 2 == row:
                f.rect(x+866, y, 18, 10, color, 4)
        f.text(494, y-9, 'Correct → keep positive' if row == 0 else 'Incorrect → keep negative', 17, TEAL if row == 0 else VIOLET, True)
    for left, right in [(379, 463), (813, 897)]:
        f.line([(left, 228), (right, 228)], '#34465f', 2, arrow=True)
        f.rect(left + phase*(right-left-10), 223, 10, 10, TEAL, 5)
    f.text(38, 322, 'VeriOPD: verifier-guided OPD    /    VeriGRPD: group-relative extension', 17, MUTED)
    f.text(1036, 322, 'Schematic only', 16, TEAL)
    return f


def results():
    data = json.loads((ROOT/'docs/results/reported-results.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'path'})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), facecolor=BG)
    for ax, exp in zip(axes.flat, data['experiments']):
        rows = exp['rows']
        labels = [r['method'].replace('Sampled-Token OPD', 'Sampled OPD') for r in rows]
        values = [r['scores'][-1] for r in rows]
        colors = [VIOLET if r['method']=='VeriGRPD' else TEAL if r['method']=='VeriOPD' else '#526780' for r in rows]
        ax.set_facecolor(BG)
        ax.barh(labels, values, color=colors, height=.58, zorder=3)
        ax.invert_yaxis()
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 20, 40, 60])
        ax.tick_params(colors=MUTED, length=0, pad=8)
        ax.set_xlabel('Reported average (%)', color=MUTED, fontsize=9)
        ax.set_title(exp['title'], loc='left', color=INK, fontsize=13, fontweight='bold', pad=18)
        ax.grid(axis='x', color='#233146', linewidth=.7, zorder=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        for i,v in enumerate(values): ax.text(v+.8, i, f'{v:.2f}', va='center', color=INK, fontsize=10)
    fig.suptitle('Four studies. Every comparison in view.', color=INK, fontsize=21, fontweight='bold', x=.04, ha='left')
    fig.text(.04,.02,'Source: supplied experiment records. Initial student and teacher are references. Panels are separate studies.',color=MUTED,fontsize=10)
    fig.subplots_adjust(left=.15,right=.95,top=.87,bottom=.10,wspace=.55,hspace=.60)
    fig.savefig(ASSETS/'benchmark-overview.svg', facecolor=BG)
    fig.savefig(ASSETS/'benchmark-overview.png', facecolor=BG, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    animate('verigate-motion', hero)
    animate('verifier-motion', gate)
    results()
    print('Built two GIFs with static fallbacks and the four-study benchmark chart.')
