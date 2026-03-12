"""
Convenience script to run all experiments end-to-end.

Usage:
    python run_all.py [--no-save] [--domains banking hotel]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run all FuDGE/FF1 experiments")
    parser.add_argument("--domains", nargs="+", default=["banking", "hotel"])
    parser.add_argument("--no-save", action="store_true", help="Don't save figures")
    parser.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3],
                        help="Which experiments to run (1, 2, 3)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" FuDGE / FF1 Replication — arxiv 2411.10416")
    print("=" * 70 + "\n")

    # Verify data loads
    print("Step 0: Verifying data loader...")
    from src.data_loader import load_star
    data = load_star(domains=args.domains, max_dialogues_per_domain=50)
    for domain, info in data.items():
        print(f"  {domain}: {len(info['dialogues'])} dialogues, flow={info['flow']}")
    print("  Data OK\n")

    if 1 in args.exp:
        print("Running Experiment 1: Discrimination...")
        from experiments.exp1_discrimination import run_experiment as exp1
        results = exp1(domains=args.domains, variant="min", save_fig=not args.no_save)

    if 2 in args.exp:
        print("\nRunning Experiment 2: Hyperparameter (k) selection...")
        from experiments.exp2_hyperparam import run_experiment as exp2
        results = exp2(domains=args.domains, save_fig=not args.no_save)

    if 3 in args.exp:
        print("\nRunning Experiment 3: Supervised vs Unsupervised...")
        from experiments.exp3_sup_vs_unsup import run_experiment as exp3
        results = exp3(domains=args.domains, save_fig=not args.no_save)

    print("\n" + "=" * 70)
    print(" All experiments complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
