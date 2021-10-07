import luigi
from . import GenerateSceneIDs


class SceneBulkProcessingBaseTask(luigi.Task):
    """
    Base class task for processing all scenes in a dataset in separate task for
    each scene. Must set `TaskClass` in the derived class
    """

    data_path = luigi.Parameter()
    TaskClass = None

    def requires(self):
        if self.TaskClass is None:
            raise Exception(
                "Please set TaskClass to the type you would like"
                " to process for every scene"
            )
        return GenerateSceneIDs(data_path=self.data_path)

    def _get_task_class_kwargs(self):
        raise NotImplementedError(
            "Please implement `_get_task_class_kwargs`"
            " to provide the necessary kwargs for your"
            " selected task"
        )

    def _build_runtime_tasks(self):
        all_source_data = self.input().read()
        kwargs = self._get_task_class_kwargs()

        tasks = {}
        for scene_id in all_source_data.keys():
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

    scene_prefix = "DATE"

    def _build_runtime_tasks(self):
        all_source_data = self.input().read()
        kwargs = self._get_task_class_kwargs()

        if not self.scene_prefix == "DATE":
            raise NotImplementedError(self.scene_prefix)

        scenes_by_prefix = {}
        for scene_id in all_source_data.keys():
            if not scene_id.startswith("goes16_"):
                raise NotImplementedError(scene_id)

            # the last four characters are the hour and minute
            # TODO: generalise this to have a parser for the scene_id
            prefix = scene_id[:-4]

            if prefix not in scenes_by_prefix:
                scenes_by_prefix[prefix] = []
            scenes_by_prefix[prefix].append(scene_id)

        tasks = {}
        for prefix, scene_ids in scenes_by_prefix.items():
            tasks[prefix] = self.TaskClass(
                scene_ids=scene_ids,
                prefix=prefix,
                dataset_path=self.dataset_path,
                **kwargs,
            )
        return tasks
