"""Download, preprocess and evaluate one explicit experimental profile."""
import argparse
from pathlib import Path
import subprocess
import sys
from src.experiment_profiles import PROFILES

def build_commands(args):
    profile = PROFILES[args.profile]
    preprocessing = ['--sampling-rate', str(profile.sampling_rate),
                     '--epoch-start', str(profile.epoch_start), '--epoch-end', str(profile.epoch_end)]
    return [
        ['download_real_data.py', '--subjects', *map(str, args.subjects), '--provider', args.provider, '--workers', '4'],
        ['prepare_real_data.py', '--subjects', *map(str, args.subjects),
         '--output-dir', Path(profile.data_root).name, *preprocessing],
        ['run_real_batch.py', '--subjects', *map(str, args.subjects),
         '--targets', *map(str, args.targets or args.subjects), '--seeds', *map(str, args.seeds),
         '--epochs', str(args.epochs), '--adapt-epochs', str(args.adapt_epochs),
         '--budgets', *map(str, args.budgets), '--data-root', profile.data_root,
         '--samples', str(profile.samples), *preprocessing,
         '--temporal-kernel', str(profile.temporal_kernel), '--dropout', str(profile.dropout),
         '--output', args.output or f'results/real/{args.profile}_rerun',
         *(['--no-plots'] if args.no_plots else [])],
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=PROFILES, default='paper')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 10)))
    parser.add_argument('--targets', type=int, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--provider', choices=['upstream', 'nemar'], default='nemar')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--adapt-epochs', type=int, default=50)
    parser.add_argument('--budgets', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--output')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (len(set(args.subjects)) < 3 or len(set(args.subjects)) != len(args.subjects)
            or not set(args.subjects) <= set(range(1, 10))
            or not set(args.targets or args.subjects) <= set(args.subjects)
            or min(args.epochs, args.adapt_epochs, *args.budgets) < 1):
        parser.error('Use at least 3 distinct subjects in 1..9 and positive budgets/epochs')
    for command in build_commands(args):
        print(subprocess.list2cmdline([sys.executable, *command]), flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, *command], cwd=Path(__file__).resolve().parent, check=True)

if __name__ == '__main__':
    main()
