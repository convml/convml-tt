import matplotlib.pyplot as plt

from . import TripletDataset

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("path")
    argparser.add_argument("scene_num", type=int)
    args = argparser.parse_args()

    dataset = TripletDataset.load(args.path)
    da_scene = dataset.get_scene(scene_num=args.scene_num)

    fig, ax = plt.subplots(figsize=(10, 4), subplot_kw=dict(projection=da_scene.crs))

    da_scene.coarsen(dict(x=100, y=100), boundary="trim").max().plot.imshow(
        rgb="bands", ax=ax
    )
    plt.savefig("scene_{}.png".format(args.scene_num))
