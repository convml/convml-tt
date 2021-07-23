from .interpretation.rectpred.sample import make_plot as rectpred_sample_plot
from .interpretation.plots import dendrogram
from .utils import get_embeddings


def make_rectpred_logger(image_path):
    import wandb

    def _logger(model, stage):
        fig = rectpred_sample_plot(model=model, image_path=image_path)
        wandb.log({f"{stage}_rectpred": fig})

    return _logger


def make_dendrogram_logger(datamodule):
    import wandb

    def _logger(model, stage):
        try:
            dataset = datamodule.get_dataset(stage="predict")
        except FileNotFoundError:
            dataset = datamodule.get_dataset(stage="fit")
        da_embeddings = get_embeddings(tile_dataset=dataset, model=model)
        fig = dendrogram(da_embeddings=da_embeddings)
        wandb.log({f"{stage}_dendrogram": fig})

    return _logger
