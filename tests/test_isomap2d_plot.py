#!/usr/bin/env python
# coding: utf-8

from convml_tt.data.dataset import ImageTripletDataset
from convml_tt.data.examples import (
    ExampleData,
    PretrainedModel,
    fetch_example_dataset,
    load_pretrained_model,
)
from convml_tt.data.transforms import get_transforms
from convml_tt.interpretation.plots import isomap2d
from convml_tt.utils import get_embeddings


def test_isomap2d():
    data_path = fetch_example_dataset(dataset=ExampleData.SMALL100)
    model = load_pretrained_model(pretrained_model=PretrainedModel.FIXED_NORM_STAGE2)

    dataset = ImageTripletDataset(
        data_dir=data_path,
        transform=get_transforms(step="predict", normalize_for_arch=model.base_arch),
        stage="train",
    )

    da_embs = get_embeddings(
        tile_dataset=dataset,
        model=model,
        prediction_batch_size=4,
    )

    variants = [
        dict(plot_type="grid", dx=0.2),
        dict(
            plot_type="scatter",
            dx=0.2,
            an_dist_ecdf_threshold=0.3,
            inset_triplet_distance_distributions=True,
        ),
    ]

    for variant in variants:
        isomap2d.make_isomap_reference_plot(da_embs=da_embs, **variant)
