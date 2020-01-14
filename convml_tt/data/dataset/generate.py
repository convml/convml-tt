from pathlib import Path

from . import TripletDataset



if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('path')
    argparser.add_argument('source_data_path')
    argparser.add_argument('--offline', action='store_true')

    args = argparser.parse_args()

    dataset = TripletDataset.load(args.path)
    dataset.generate(Path(args.source_data_path), offline_cli=args.offline)
