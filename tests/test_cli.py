from convml_tt.trainer import main as trainer_main
from convml_tt.data.examples import main as data_examples_main


def test_cli():
    """
    Ensure that running a simple training example through the command-line interface works
    """
    dl_args = ["TINY10"]
    data_examples_main(args=dl_args)
    train_args = ["data/Nx256_s200000.0_N0study_N10train"]
    trainer_main(args=train_args)
