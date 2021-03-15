import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from . import TripletDataset


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", default=".")
    argparser.add_argument("--projection", default="PlateCarree", choices=vars(ccrs))
    args = argparser.parse_args()

    Projection = getattr(ccrs, args.projection)

    dataset = TripletDataset.load(args.path)
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=Projection()))
    dataset.plot_domain(ax=ax)
    fn = "domain.png"
    plt.savefig(fn)
    print("Saved domain plot to `{}`".format(fn))
