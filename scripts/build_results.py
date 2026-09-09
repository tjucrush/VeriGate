"""Build the reported-results tables and artwork from one auditable data file."""
import json
from pathlib import Path
from build_readme_art import Figure

ROOT = Path(__file__).resolve().parents[1]


def table(rows):
    best = [max(r['scores'][i] for r in rows[2:]) for i in range(7)]
    out = ['| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |',
           '| :-- | --: | --: | --: | --: | --: | --: | --: |']
    for j, row in enumerate(rows):
        name = row['method']
        if name in {'VeriOPD', 'VeriGRPD'}:
            name = '**' + name + '**'
        vals = [f'**{v:.1f}**' if j >= 2 and v == best[i] else f'{v:.1f}' for i, v in enumerate(row['scores'])]
        out.append('| ' + ' | '.join([name] + vals) + ' |')
    return '\n'.join(out)


def main():
    data = json.loads((ROOT/'docs/results/reported-results.json').read_text(encoding='utf-8'))
    parts = ['## Reported benchmark results',
             '<img src="docs/assets/results-highlight.svg" alt="Reported averages: VeriOPD 49.1 for 4B to 4B, VeriOPD 22.8 for 4B to 1.7B, and VeriGRPD 49.4 for the group-relative study." width="100%">',
             '**Six mathematics benchmarks · Four controlled comparisons**',
             'Scores below are transcribed from the supplied experiment records. **Bold** marks the best score among trained student methods within each table, including ties; the initial student and teacher are reference rows. `Avg.` preserves the reported average.',
             '> These records evaluate the verifier-gated method family. They are not benchmark measurements of the current direct-KL ordinal trainer. [Protocol and source notes](docs/RESULTS.md#reporting-notes).']
    for exp in data['experiments']:
        parts += ['### '+exp['title'], '**Teacher:** '+exp['teacher']+' → **Student:** '+exp['student'], table(exp['rows']), exp['finding']]
    parts += ['[Experiment records and reporting notes →](docs/RESULTS.md) · [Machine-readable scores →](docs/results/reported-results.json)']
    section = '\n\n'.join(parts)
    readme = ROOT/'README.md'
    text = readme.read_text(encoding='utf-8')
    start, end = '<!-- reported-results:start -->', '<!-- reported-results:end -->'
    if start in text:
        text = text[:text.index(start)] + start+'\n'+section+'\n'+end + text[text.index(end)+len(end):]
    else:
        text = text.replace('## The method',start+'\n'+section+'\n'+end+'\n\n## The method')
    readme.write_text(text,encoding='utf-8')
    notes = '''# Experimental results

## Method names

**VeriOPD — Verifier-Guided On-Policy Distillation** names the verifier-guided token-update method. **VeriGRPD — Verifier-Guided Group-Relative Policy Distillation** names its group-relative extension. These are presentation names for the supplied experiments; the renaming does not change algorithms or establish equivalence with a different training objective.

## Reporting notes

- Source: two experiment-table screenshots supplied by the project owner. Benchmark cells and reported averages are transcribed without changing their values.
- The earlier result record identifies the metric as avg@16 (%). The newly supplied screenshots do not specify the sampling protocol; confirm it from run metadata before treating all four studies as protocol-matched.
- `Avg.` is the average reported in the screenshots. Rounded benchmark cells may not reproduce it exactly; for example, the first Sampled-Token OPD row averages 47.8167 and is reported as 47.8.
- Bold values compare trained student methods within each table. Student initialization and teacher rows are references, and are excluded from this highlighting. Ties are highlighted equally.
- Differences are percentage points computed from reported averages. No significance, variance, or state-of-the-art claim is implied.
- The group-relative study reports a different OPD baseline (48.4) from the sampled-token baseline (47.8) in the distillation and gate-ablation studies. Their labels and values are kept separate; do not pool the studies.
- Raw predictions, seed counts, exact checkpoint revisions, and evaluation configuration were not supplied with the screenshots. The current direct-KL ordinal implementation has a different objective and is not the source of these benchmark records.
- The 4B → 1.7B comparison changes model size within the Qwen3 family; it is labeled cross-size transfer rather than claiming a different architecture.

## Recorded tables

'''
    for exp in data['experiments']:
        notes += '### '+exp['title']+'\n\nTeacher: '+exp['teacher']+' → Student: '+exp['student']+'\n\n'+table(exp['rows'])+'\n\n'+exp['finding']+'\n\n'
    notes += '[Download the score record](results/reported-results.json) · [Back to the project](../README.md#reported-benchmark-results)\n'
    (ROOT/'docs/RESULTS.md').write_text(notes,encoding='utf-8')
    f=Figure(1280,360,'#0b1220')
    f.text(44,30,'VERIGATE / EMPIRICAL STUDIES',17,'#65e0c2',True)
    f.text(44,65,'Verification-guided learning, measured.',34,'#f3f7ff',True)
    for x, label, score, method, delta in [(44,'01 / SAME-SIZE DISTILLATION','49.1','VeriOPD · 4B → 4B','+1.3 pp vs. Sampled-Token OPD'),(449,'02 / CROSS-SIZE TRANSFER','22.8','VeriOPD · 4B → 1.7B','+1.1 pp vs. Top-64 OPD'),(854,'03 / GROUP-RELATIVE STUDY','49.4','VeriGRPD · 4B → 4B','+4.6 pp vs. GRPO')]:
        f.rect(x,133,382,176,'#142235',12,'#2a3b50')
        f.text(x+20,148,label,14,'#a6b6cd',True)
        f.text(x+18,176,score,51,'#f3f7ff',True)
        f.text(x+148,207,'AVG. (%)',14,'#a6b6cd')
        f.text(x+20,243,method,18,'#a5a0ff',True)
        f.text(x+20,277,delta,16,'#65e0c2')
    f.text(44,329,'Reported experiment records · Six benchmarks · Separate comparison settings',15,'#a6b6cd')
    f.save('results-highlight')
    family = Figure(1280, 244, '#0b1220')
    for x, number, name, subtitle, ingredients, color in [
        (28, '01', 'VeriOPD', 'VERIFIER-GUIDED ON-POLICY DISTILLATION', 'OPD + verifiable feedback', '#65e0c2'),
        (654, '02', 'VeriGRPD', 'VERIFIER-GUIDED GROUP-RELATIVE POLICY DISTILLATION', 'GRPO + OPD + verifiable feedback', '#a5a0ff'),
    ]:
        family.rect(x, 24, 598, 196, '#142235', 12, '#2a3b50')
        family.rect(x+24, 49, 4, 26, color, 2)
        family.text(x+42, 48, 'METHOD ' + number, 15, color, True)
        family.text(x+24, 82, name, 42, '#f3f7ff', True)
        family.text(x+24, 144, subtitle, 13, '#a6b6cd', True)
        family.text(x+24, 177, ingredients, 20, color)
    family.save('method-family')
    print('Built four tables in README and RESULTS, plus SVG/PNG results artwork.')


if __name__ == '__main__':
    main()
