"""Run the complete local forecasting pipeline from downloaded CSVs."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    'scripts/audit_data.py',
    'scripts/analyze_data.py',
    'scripts/evaluate_baseline.py',
    'scripts/train_tree_model.py',
    'scripts/explain_model.py',
    'scripts/generate_forecasts.py',
]


def main():
    for step in STEPS:
        print(f'\n=== {step} ===', flush=True)
        subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT, check=True)
    print('\nPipeline complete. Start the dashboard with: streamlit run app/dashboard.py')


if __name__ == '__main__':
    main()
