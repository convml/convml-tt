from pathlib import Path
import importlib
import pprint

import yaml
import luigi


class TripletDataset:
    def __init__(self, N_triplets, data_path, name, extra={}):
        """
        N_triplets can be an integer (interpreted as no study data being
        requested) or a dictionary with {'train': N_train, 'study': N_study}
        """
        self.N_triplets = N_triplets
        self.data_path = data_path
        self.name = name
        self.extra = extra

    def save(self):
        tile_path_base = self._get_bae_path()
        tile_path_base.mkdir(exist_ok=True, parents=True)

        if tile_path_base.exists():
            raise NotImplementedError('A dataset already exists in `{}`'
                                      ''.format(tile_path_base))

        data = {}
        for k, v in vars(self).items():
            if not k.startswith('_'):
                data[k] = v
        del(data['data_path'])
        del(data['name'])

        data['type'] = self.__module__ + '.' + self.__class__.__name__
        p = tile_path_base/"meta.yaml"
        with open(str(p), 'w') as fh:
            fh.write(yaml.dump(data))
        print("Triplet dataset saved to `{}`".format(str(p)))

    def _get_base_path(self):
        raise NotImplementedError

    @staticmethod
    def load(path):
        path_abs = Path(path).expanduser().absolute()
        p = path_abs/"meta.yaml"
        name = p.parent.name
        with open(str(p)) as fh:
            data = yaml.load(fh.read())
        data['name'] = name
        data['data_path'] = path_abs
        class_fqn = data.pop('type')
        i = class_fqn.rfind('.')
        module_name, class_name = class_fqn[:i], class_fqn[i+1:]
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**data)

    def __repr__(self):
        return pprint.pformat({
            k: v for k, v in vars(self).items() if not k.startswith('_')
        })

    def generate(self):
        raise NotImplementedError

    def plot_domain(self, ax, **kwargs):
        raise NotImplementedError


class SceneBulkProcessingBaseTask(luigi.Task):
    dataset_path = luigi.Parameter()
    TaskClass = None

    def requires(self):
        if self.TaskClass is None:
            raise Exception("Please set TaskClass to the type you would like"
                            " to process for every scene"
                            )
        return self._load_dataset().fetch_source_data()

    def _get_task_class_kwargs(self):
        raise NotImplementedError("Please implement `_get_task_class_kwargs`"
                                  " to provide the necessary kwargs for your"
                                  " selected task")

    def _load_dataset(self):
        return TripletDataset.load(self.dataset_path)

    def _build_runtime_tasks(self):
        all_source_data = self.input().read()
        kwargs = self._get_task_class_kwargs()

        tasks = {}
        for scene_id in all_source_data.keys():
            tasks[scene_id] = self.TaskClass(
                scene_id=scene_id,
                dataset_path=self.dataset_path,
                **kwargs
            )
        return tasks

    def run(self):
        yield self._build_runtime_tasks()

    def output(self):
        if not self.input().exists():
            # the fetch has not completed yet, so we don't know how many scenes
            # we will be working on. Therefore we just return a target we know
            # will newer exist
            return luigi.LocalTarget("__fake_file__.nc")
        else:
            tasks = self._build_runtime_tasks()
            return { scene_id: t.output() for scene_id, t in tasks.items() }
