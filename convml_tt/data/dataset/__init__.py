from pathlib import Path
import importlib
import pprint

import yaml


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
        return pprint.pformat({k:v for k,v in vars(self).items() if not k.startswith('_')})

    def generate(self):
        raise NotImplementedError

    def plot_domain(self, ax, **kwargs):
        raise NotImplementedError
