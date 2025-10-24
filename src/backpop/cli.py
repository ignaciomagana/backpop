from argparse import ArgumentParser
from .main import BackPop
from time import time

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--ini_file', help='Path to INI file', type=str, default="params.ini")
    parser.add_argument('-t', '--timeit', help='Time the run', action='store_true', default=False)
    args = parser.parse_args()

    start = time() if args.timeit else None
    bp = BackPop(config_file=args.ini_file)
    bp.run_sampler()
    
    if args.timeit:
        print(f"Run time: {time() - start:.1f} seconds")

if __name__ == "__main__":
    main()
