import os
import json
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
PERFORMANCE_DIR = os.path.join(ROOT, 'performance')
os.makedirs(PERFORMANCE_DIR, exist_ok=True)


def run_cmd(cmd, cwd=ROOT):
    print("Running:", cmd)
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_suggestions(misfile=os.path.join(PERFORMANCE_DIR, 'misclassifications.json'), reportfile=os.path.join(PERFORMANCE_DIR, 'evaluation_report.json')):
    if not os.path.exists(misfile) or not os.path.exists(reportfile):
        print('Missing analysis files; cannot generate suggestions')
        return {}
    mis = load_json(misfile)
    report = load_json(reportfile)

    suggestions = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'summary': {
            'total_patterns': mis.get('total_patterns'),
            'total_misclassified': mis.get('total_misclassified')
        },
        'fixes': []
    }

    for pair in mis.get('top_confused_pairs', []):
        true = pair['true']
        pred = pair['pred']
        count = pair['count']
        examples = pair.get('examples', [])
        if count >= 3:
            suggestions['fixes'].append({
                'action': 'add_patterns',
                'target_intent': true,
                'reason': f'Model confused {count} patterns labeled {true} as {pred}',
                'suggested_patterns': examples,
                'note': 'Review these examples and, if appropriate, accept or expand them, or move them into a more specific intent.'
            })

    for tag, stats in report.items():
        if tag in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        prec = stats.get('precision', 0)
        rec = stats.get('recall', 0)
        if prec < 0.6 and rec > 0.8:
            suggestions['fixes'].append({
                'action': 'review_intent',
                'intent': tag,
                'reason': f'Low precision ({prec:.2f}) but high recall ({rec:.2f}) — may be too general or absorbing others',
                'note': 'Consider tightening patterns, splitting, or adding clarifying examples to other intents.'
            })

    write_json(os.path.join(PERFORMANCE_DIR, 'suggested_fixes.json'), suggestions)
    print('Wrote performance/suggested_fixes.json')
    return suggestions


def main():
    # 1) Run training
    try:
        # use the same python executable and run the train module
        run_cmd(f'{sys.executable} -m train.train')
    except Exception as e:
        print('Training failed:', e)
        return

    # 2) Run evaluation
    try:
        run_cmd(f'{sys.executable} -m train.evaluate')
    except Exception as e:
        print('Evaluation failed:', e)
        return

    # 3) Analyze misclassifications
    try:
        run_cmd(f'{sys.executable} -m train.analyze_misclassifications')
    except Exception as e:
        print('Misclassification analysis failed:', e)
        return

    # 4) Generate conservative suggestions for human review
    generate_suggestions()


if __name__ == '__main__':
    main()
