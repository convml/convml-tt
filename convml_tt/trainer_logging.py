from .interpretation.plots import dendrogram
from .utils import get_embeddings


def make_dendrogram_logger(datamodule):
    import wandb

    def _logger(model, stage):
        try:
            dataset = datamodule.get_dataset(stage="predict")
        except FileNotFoundError:
            dataset = datamodule.get_dataset(stage="fit")
        da_embeddings = get_embeddings(tile_dataset=dataset, model=model)
        fig = dendrogram(
            da_embeddings=da_embeddings,
            tile_type="anchor",
            sampling_method="best_triplets",
        )
        wandb.log({f"{stage}_dendrogram": wandb.Image(fig)})

    return _logger
