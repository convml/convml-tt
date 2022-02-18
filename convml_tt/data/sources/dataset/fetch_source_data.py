import luigi

from . import TripletDataset


def _ensure_task_run(t):
    if not t.output().exists():
        luigi.build(
            [
                t,
            ],
            local_scheduler=True,
        )
    if not t.output().exists():
        raise Exception("Task didn't complete")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset-path", default=".")

    args = argparser.parse_args()

    dataset = TripletDataset.load(args.dataset_path)
    t = dataset.fetch_source_data()
    _ensure_task_run(t)
