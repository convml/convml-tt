from pathlib import Path

from . import TripletDataset



if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('path')
    argparser.add_argument('scene_num', type=int)

    args = argparser.parse_args()

    dataset = TripletDataset.load(args.path)
    da_scene = dataset.get_scene(scene_num=args.scene_num)
