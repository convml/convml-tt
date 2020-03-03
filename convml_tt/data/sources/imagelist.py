from pathlib import Path
from fastai.vision import open_image
import re


class SingleTileImageList(list):
    def __init__(self, src_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_path = Path(src_path)
        self.tile_ids = []
        
        files = list(Path(src_path).glob("*_anchor.png"))
        if len(files) == 0:
            raise Exception("No images found in `{}`, files should be" 
                             " named as `????_anchor.png`, e.g. `00000_anchor.png")
        else:
            for fn in files:
                m = re.match(r"^(?P<tile_id>\d{5})_anchor\.png$", fn.name)
                if m is not None:
                    tile_id = int(m.groupdict()['tile_id'])
                else:
                    raise Exception("Invalid tile filename `{}`".format(fn))
                self.append(open_image(fn))
                self.tile_ids.append(tile_id)
            print("Loaded {} image tiles".format(len(self)))

    @property
    def size(self):
        return self[0].size

    def apply_tfms(self, tfms, **kwargs):
        items = []
        for item in self:
            items.append(item.apply_tfms(tfms, **kwargs))
        return items
