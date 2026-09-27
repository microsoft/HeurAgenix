"""Run Jev evaluation directly from the repository checkout.

Usage: python run_benchmark.py --config configs/pilot.yaml
This starts live evaluation; TYPESAFE_API_KEY must be set in the environment.
"""

from src.benchmark.__main__ import main


if __name__ == "__main__":
    main()