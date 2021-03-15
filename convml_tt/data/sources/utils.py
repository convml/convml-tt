import datetime
from pathlib import Path

SOURCE_DIR = Path("source_data")
SCENE_ID_DATE_FORMAT = "%Y%m%d%H%M"


def parse_scene_id(scene_id):
    datasource_name, datetime_str = scene_id.split("_")
    t_scene = datetime.datetime.strptime(datetime_str, SCENE_ID_DATE_FORMAT)
    return datasource_name, t_scene


def create_scene_id(datasource_name, t_scene):
    assert "_" not in datasource_name
    datetime_str = t_scene.strftime(SCENE_ID_DATE_FORMAT)
    return f"{datasource_name}_{datetime_str}"
