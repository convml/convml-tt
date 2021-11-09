from .sampling import GenerateCroppedScenes
from .scene_sources import (
    SCENE_ID_DATE_FORMAT,
    GenerateSceneIDs,
    make_scene_id,
    parse_scene_id,
)
from .tiles import GenerateRegriddedScenes, SceneRegriddedData
from .triplets import GenerateTiles
from .utils import SceneBulkProcessingBaseTask


def get_source_files_filepaths(data_source):
    t_scene_ids = GenerateSceneIDs(data_path=data_source.data_path)
    if not t_scene_ids.output().exists():
        raise Exception(
            "The source files database hasn't been populated yet. Please run "
            "then `GenerateSceneIDs` task to build it"
        )

    scene_sources = t_scene_ids.output().open()
    return scene_sources
