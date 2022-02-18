from . import TripletDataset

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("path")

    args = argparser.parse_args()

    dataset = TripletDataset.load(args.path)
    print(repr(dataset))
