from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.builder import Builder

class BetterBuilder(Builder):
    def __init__(self,
                task_class,
                config = None,
                render_mode = None,
                width = 256,
                height = 256,
                camera_id = None,
                camera_name = None):

        self.task_class = task_class

        super().__init__("", config, render_mode, width, height, camera_id, camera_name)

    def _get_task(self):
        task = self.task_class(config=self.config)

        task.build_observation_space()
        return task

