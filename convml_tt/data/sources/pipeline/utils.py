import luigi
import re
from .scene_sources import GenerateSceneIDs, parse_scene_id


class SceneBulkProcessingBaseTask(luigi.Task):
    """
    Base class task for processing all scenes in a dataset in separate task for
    each scene. Must set `TaskClass` in the derived class
    """

    data_path = luigi.Parameter(default=".")
    TaskClass = None
    SceneIDsTaskClass = GenerateSceneIDs
    scene_filter = luigi.OptionalParameter(default=None)

    def requires(self):
        if self.TaskClass is None:
            raise Exception(
                "Please set TaskClass to the type you would like"
                " to process for every scene"
            )
        return self.SceneIDsTaskClass(
            data_path=self.data_path, **self._get_scene_ids_task_kwargs()
        )

    def _get_task_class_kwargs(self, scene_ids):
        raise NotImplementedError(
            "Please implement `_get_task_class_kwargs`"
            " to provide the necessary kwargs for your"
            " selected task"
        )

    def _get_scene_ids_task_kwargs(self):
        return {}

    def _filter_scene_ids(self, scene_ids):
        if self.scene_filter is not None:
            scene_ids = [
                scene_id
                for scene_id in scene_ids
                if re.match(self.scene_filter, scene_id)
            ]
        return scene_ids

    def _build_runtime_tasks(self):
        all_source_data = self.input().read()

        tasks = {}
        scene_ids = all_source_data.keys()
        scene_ids = self._filter_scene_ids(scene_ids=scene_ids)

        kwargs = self._get_task_class_kwargs(scene_ids=scene_ids)

        for scene_id in scene_ids:
            tasks[scene_id] = self.TaskClass(
                scene_id=scene_id, data_path=self.data_path, **kwargs
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
            return {scene_id: t.output() for scene_id, t in tasks.items()}


class GroupedSceneBulkProcessingBaseTask(SceneBulkProcessingBaseTask):
    """
    Groups all scenes in dataset by date and runs them provided Task
    """
    scene_group_prefix = "DATE"

    def _build_runtime_tasks(self):
        all_source_data = self.input().read()

        scene_ids = all_source_data.keys()
        scene_ids = self._filter_scene_ids(scene_ids=scene_ids)

        scenes_by_prefix = {}
        for scene_id in scene_ids:
            source_type, t_scene = parse_scene_id(scene_id)
            if self.scene_group_prefix == "DATE":
                prefix = f"{source_type}__{t_scene:%Y%m%d}"
            else:
                raise NotImplementedError(self.scene_group_prefix)

            if prefix not in scenes_by_prefix:
                scenes_by_prefix[prefix] = []
            scenes_by_prefix[prefix].append(scene_id)

        tasks = {}
        for prefix, scene_ids in scenes_by_prefix.items():
            kwargs = self._get_task_class_kwargs(scene_ids=scene_ids)
            tasks[prefix] = self.TaskClass(
                scene_ids=scene_ids,
                prefix=prefix,
                data_path=self.data_path,
                **kwargs,
            )
        return tasks
