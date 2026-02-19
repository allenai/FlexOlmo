from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
from oe_eval.data.mmlu_pro_categories import MMLU_PRO_CATEGORIES


def get_task_suite_configs():
    TASK_SUITE_CONFIGS.update(
        {
            "mmlu_pro:mc": {
                "tasks": [f"mmlu_pro_{cat}:mc::none" for cat in MMLU_PRO_CATEGORIES],
                "primary_metric": "macro",
            },
            "sciriff5": {
                "tasks": [
                    "sciriff_bioasq_factoid_qa",  # Abstractive
                    "sciriff_bioasq_general_qa",  # Abstractive
                    "sciriff_bioasq_yesno_qa",  # Y/N
                    "sciriff_covid_deepset_qa",  # Extractive
                    "sciriff_pubmedqa_qa",  # Y/N
                ],
                "primary_metric": "macro",
            },
        },
    )
    return TASK_SUITE_CONFIGS
