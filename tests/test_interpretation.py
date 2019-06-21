from fastai.datasets import untar_data
import fastai.vision

from convml_tt.architectures.triplet_trainer import (NPMultiImageItemList,
                                                     loss_func,
                                                     monkey_patch_fastai)

from convml_tt.data.examples import ExampleData

from convml_tt.utils import get_embeddings


def test_getting_embeddings():
    data_path = untar_data(ExampleData.TINY10)

    monkey_patch_fastai()

    tile_path = data_path/"train"

    item_list = NPMultiImageItemList.from_folder(path=tile_path)

    src = (item_list
           .random_split_by_pct()
           .label_empty(embedding_length=100)
           )

    data = (src
            .transform(fastai.vision.get_transforms(flip_vert=True,))
            .databunch(bs=3)
            .normalize(fastai.vision.imagenet_stats)
            )

    learn = fastai.vision.create_cnn(data=data,
                                     base_arch=fastai.vision.models.resnet18,
                                     loss_func=loss_func
                                     )

    learn.fit_one_cycle(cyc_len=3, max_lr=4.0e-2)


    items_study = item_list[:2]
    embs_anchor = get_embeddings(triplets=items_study, model=learn)
    embs_all = get_embeddings(triplets=items_study, model=learn, tile_type=None)

    assert len(items_study) == embs_all.tile_id.count()

    assert (embs_all.sel(tile_type='anchor') == embs_anchor).all()
