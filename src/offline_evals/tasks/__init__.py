from typing import Dict

from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY

from . import (
    squad,
    squad2,
)

new_task_registry: Dict = {
    "squad": squad.SQuAD,
    "squad2": squad2.SQuAD2,
}

TASK_REGISTRY.update(new_task_registry)