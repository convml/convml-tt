from pathlib import Path

from . import TripletDataset



if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset-path', default=".")

    args = argparser.parse_args()

    dataset = TripletDataset.load(args.dataset_path)
    dataset.fetch_source_data()
