from pathlib import Path
import importlib
import pprint

import yaml


class TripletDataset:
    def __init__(self, N_triplets, data_path):
        """
        N_triplets can be an integer (interpreted as no study data being
        requested) or a dictionary with {'train': N_train, 'study': N_study}
        """
        self.N_triplets = N_triplets
        self.data_path = data_path

    def save(self, name):
        tile_path_base = self.data_path/"tiles"/"goes16"/name
        tile_path_base.mkdir(exist_ok=True, parents=True)

        data = {}
        for k, v in vars(self).items():
            if not k.startswith('_'):
                data[k] = v

        data['type'] = self.__module__ + '.' + self.__class__.__name__
        p = tile_path_base/"meta.yaml"
        with open(str(p), 'w') as fh:
            fh.write(yaml.dump(data))
        print("Triplet dataset saved to `{}`".format(str(p)))

    def __repr__(self):
        return pprint.pformat({k:v for k,v in vars(self).items() if not k.startswith('_')})

    @staticmethod
    def load(path):
        p = Path(path)/"meta.yaml"
        with open(str(p)) as fh:
            data = yaml.load(fh.read())
        class_fqn = data.pop('type')
        i = class_fqn.rfind('.')
        module_name, class_name = class_fqn[:i], class_fqn[i+1:]
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**data)
