#!/usr/bin/env python
# coding: utf-8
"""
Sample for producing predictions on a rectangular domain that can be used
during training
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from convml_tt.interpretation.rectpred.transform import apply_transform
from convml_tt.interpretation.rectpred.data import make_sliding_tile_model_predictions
from convml_tt.system import TripletTrainerModel
from convml_tt.interpretation.rectpred.plot import make_rgb

DEFAULT_IMAGE_PATH = Path(__file__).parent.parent / "doc" / "goes16_202002051400.png"


def make_plot(model, image_path, prediction_batch_size=128):
    scene_id = image_path.name.split(".")[0]
    image_path = Path(image_path)
    img = Image.open(image_path)

    da_emb = make_sliding_tile_model_predictions(
        img=img, model=model, step=(50, 50), prediction_batch_size=prediction_batch_size
    )

    da_emb_pca = apply_transform(da_emb, "pca")

    da_rgba = make_rgb(da_emb_pca, pca_dim=[0, 1, 2], alpha=0.6)

    r_aspect = img.size[0] / img.size[1]
    fig, ax = plt.subplots(figsize=(4.0 * r_aspect, 4.0))
    ax.imshow(img)
    da_rgba.plot.imshow(rgb="rgba", y="j0", ax=ax)

    return fig


def main(model_path, image_path):
    model_path = Path(model_path)

    model_name = model_path.parent.parent.name
    model = TripletTrainerModel.load_from_checkpoint(model_path)

    make_plot(model=model, image_path=image_path, model_name=model_name)

    output_img_filename = f"{model_name}.{scene_id}.PCA012_rgb.png"
    plt.savefig(output_img_filename)
    print(f"wrote image to `{output_img_filename}`")


if __name__ == "__main__":

    import argparse
    import glob

    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_path", type=Path)
    argparser.add_argument("--image_path", type=Path, default=DEFAULT_IMAGE_PATH)

    args = argparser.parse_args()

    if not args.model_path.name.endswith(".ckpt"):
        # try to find a checkpoint file in a subdir
        g_pattern = str(args.model_path / "**" / "*.ckpt")
        paths = list(glob.glob(g_pattern, recursive=True))

        if len(paths) != 1:
            raise Exception(
                "Expected to find exactly one checkpoint file in the sub-directories"
                f" of the path provided ({args.model_path}), but found {len(paths)}:"
                f" {', '.join(paths)}"
            )
        else:
            model_path = paths[0]
    else:
        model_path = args.model_path

    main(model_path=model_path, image_path=args.image_path)
