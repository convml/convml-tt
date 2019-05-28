from fastai.datasets import untar_data, _checks
import fastai.vision

from convml_tt.architectures.triplet_trainer import (NPMultiImageItemList,
                                                     loss_func,
                                                     monkey_patch_fastai)

SRC_DATA_URL = "http://homepages.see.leeds.ac.uk/~earlcd/ml-datasets/Nx256_s200000.0_N0study_N10train"
_checks[SRC_DATA_URL] = (1144669, 'c599454acb4ff07fbd1551135c350ba9')


def test_load_data_and_train():
    data_path = untar_data(SRC_DATA_URL)

    print(data_path)

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
