"""Render a code-grounded, illustrative VeriGRPD walkthrough (Pillow only)."""
from pathlib import Path
from PIL import Image
from build_readme_art import Figure

ROOT = Path(__file__).resolve().parents[1] / 'docs/assets'
BG, PANEL, WHITE, MUTED, GREEN, PURPLE = '#0b1220', '#142235', '#f3f7ff', '#a6b6cd', '#65e0c2', '#a5a0ff'
SIGNALS = [[.8, -.4, .2, -.6], [.4, -.8, .6, -.2], [-.2, .6, -.4, .8], [.6, -.4, .8, -.2]]
STEPS = [
    ('Sample a response group', 'One prompt → four student responses', ['Generate several responses to the SAME prompt.', 'Keep the prompt-group ID and old-policy scores.', 'Here: four responses, each shown with four tokens.']),
    ('Read teacher evidence', 'd = log p_teacher − log p_student', ['Score sampled tokens at their response prefixes.', 'Positive: teacher assigns greater probability.', 'Negative: student assigns greater probability.']),
    ('Verify and gate', 'Correct: max(d, 0)   /   Wrong: min(d, 0)', ['The verifier scores each completed answer.', 'Correct answers retain positive token evidence.', 'Wrong answers retain negative token evidence.']),
    ('Compare within the group', 'A = (score − group mean) / (std + epsilon)', ['Example scores: [1, 1, 1, 0]; mean = 0.75.', 'Sample standard deviation = 0.50.', 'Advantages: [0.50, 0.50, 0.50, −1.50].']),
    ('Scale the retained signal', 'reward = gated evidence × (|A| + u)', ['Set u = 0 in this illustrative example.', 'Correct-response scale: 0.50; wrong: 1.50.', 'The nonnegative scale preserves retained signs.']),
    ('Update the student', 'Detached token rewards → policy update', ['The archived implementation uses GSPO-token.', 'One sequence ratio; local token gradients.', 'Optimize the student, then sample a fresh group.']),
]


def render(stage, phase=0):
    f = Figure(1280, 690, BG)
    f.text(40, 26, 'VERIGRPD / A COMPLETE TRAINING STEP', 17, GREEN, True)
    f.text(40, 62, STEPS[stage][0], 37, WHITE, True)
    for i, label in enumerate(['SAMPLE', 'TEACHER', 'GATE', 'GROUP', 'SCALE', 'UPDATE']):
        x = 40+i*204
        f.rect(x, 126, 184, 39, GREEN if i==stage else PANEL, 8)
        f.text(x+14, 135, f'{i+1:02d} / {label}', 14, BG if i==stage else MUTED, True)
    f.rect(40, 194, 710, 383, PANEL, 14, '#2a3b50')
    f.text(61, 212, 'ONE PROMPT / FOUR RESPONSES', 15, MUTED, True)
    f.text(61, 253, 'ANSWER', 13, MUTED)
    f.text(208, 253, 'TOKEN SIGNAL', 13, MUTED)
    f.text(511, 253, 'SCORE', 13, MUTED)
    f.text(620, 253, 'SCALE', 13, MUTED)
    for row, values in enumerate(SIGNALS):
        y = 298+row*60
        color = GREEN if row<3 else PURPLE
        f.text(63, y+3, f'R{row+1}', 19, WHITE, True)
        if stage>=2: f.text(111, y+6, 'correct' if row<3 else 'wrong', 15, color)
        for k,v in enumerate(values):
            x = 210+k*67
            if stage>=2: v = max(v,0) if row<3 else min(v,0)
            if stage>=4: v *= .5 if row<3 else 1.5
            c = GREEN if v>0 else PURPLE if v<0 else '#42536b'
            f.rect(x, y, 57, 34, '#213148', 6, c if stage else '#34465f')
            f.text(x+6, y+9, f'{v:+.2f}' if stage else f't{k+1}', 13, c if stage else MUTED, True)
        f.text(531, y+6, str(1 if row<3 else 0) if stage>=2 else '—', 18, color if stage>=2 else MUTED, True)
        f.text(633, y+6, ('0.50' if row<3 else '1.50') if stage>=4 else '—', 18, color if stage>=4 else MUTED, True)
    f.rect(778, 194, 462, 383, PANEL, 14, '#2a3b50')
    f.text(799, 216, f'STEP {stage+1:02d} / WHAT HAPPENS', 16, GREEN, True)
    for i,line in enumerate(STEPS[stage][2]):
        f.text(799, 266+i*40, line, 16, WHITE if i==0 else MUTED)
    f.text(799, 428, 'GRPO-style group signal', 21, PURPLE, True)
    f.text(799, 468, '+ OPD token evidence', 21, GREEN, True)
    f.text(799, 508, '+ verifiable answer feedback', 21, WHITE, True)
    f.text(40, 604, STEPS[stage][1], 23, WHITE, True)
    f.text(40, 655, 'Illustrative numbers · sample std · u = 0 · epsilon = 1e-6 · valid response tokens only', 14, MUTED)
    f.rect(40, 680, (stage+phase)*200, 3, GREEN, 1)
    return f


def main():
    frames=[]
    for stage in range(6):
        for n in range(8):
            f=render(stage,n/8)
            if n==0: f.save(f'verigrpd-step-{stage+1}')
            frames.append(f.image.resize((1080,582),getattr(Image,'Resampling',Image).LANCZOS))
    palette=frames[0].quantize(colors=128)
    frames=[im.quantize(palette=palette,dither=getattr(Image,'Dither',Image).NONE) for im in frames]
    frames[0].save(ROOT/'verigrpd-walkthrough.gif',save_all=True,append_images=frames[1:],duration=280,loop=0,optimize=True,disposal=1)
    f=Figure(1280,294,BG)
    f.text(40,25,'WHY THE GROUP MATTERS',26,WHITE,True)
    for x,title,outcomes,scale,note,color in [(40,'Mixed outcomes','1  /  1  /  1  /  0','0.50 / 0.50 / 0.50 / 1.50','Relative differences set response intensity.',GREEN),(660,'Uniform outcomes','1  /  1  /  1  /  1','0.00 / 0.00 / 0.00 / 0.00','With u = 0, this group contributes no reward.',PURPLE)]:
        f.rect(x,79,580,178,PANEL,12,'#2a3b50')
        f.text(x+20,94,title,23,color,True)
        f.text(x+20,132,'Scores: '+outcomes,19,WHITE)
        f.text(x+20,172,'Scales: '+scale,19,color)
        f.text(x+20,216,note,16,MUTED)
    f.text(40,270,'Illustrative binary outcomes; sample-std normalization; u = 0. A positive u retains a baseline scale.',14,MUTED)
    f.save('verigrpd-group-effect')
    print('Built 48-frame walkthrough, six static stages, and group-effect comparison.')


if __name__=='__main__':
    main()
