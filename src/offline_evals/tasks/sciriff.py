import json
import os
from typing import List, Union

import smart_open
from oe_eval.components.instances import RequestInstance
from oe_eval.metrics.metric import RougeMetric, SQuADF1EMRecallMetric
from oe_eval.tasks.base_task import MultipleChoiceTask, Task
from oe_eval.utils import get_dict_with_defaults

SCIRIFF_BASE_DIR = "s3://ai2-llm/pretraining-data/sources/ds-olmo-data/sciriff/"

BIO_SCIRIFF_TASKS = [
    # biomedical
    "anat_em_ner",
    "bioasq_factoid_qa",
    "bioasq_general_qa",
    "bioasq_list_qa",
    "bioasq_yesno_qa",
    "biored_ner",
    "cdr_ner",
    "chemdner_ner",
    "chemprot_ner",
    "chemprot_re",
    "covid_deepset_qa",
    "craftchem_ner",
    "ddi_ner",
    "genia_ner",
    "gnormplus_ner",
    "linnaeus_ner",
    "medmentions_ner",
    "ncbi_ner",
    "nlmchem_ner",
    "nlmgene_ner",
    "pubmedqa_qa",
    "scientific_lay_summarisation_elife_single_doc_summ",
    "scientific_lay_summarisation_plos_single_doc_summ",
    "scientific_papers_summarization_single_doc_pubmed",
    "covidfact_entailment",
    "scifact_entailment",
    # clinical
    "bc7_litcovid_topic_classification",
    "chia_ner",
    "drug_combo_extraction_re",
    "evidence_inference",
    "healthver_entailment",
    "mslr2022_cochrane_multidoc_summarization",
    "mslr2022_ms2_multidoc_summarization",
    "pico_ner",
]

BIO_SCIRIFF_QA_TASKS = [
    "bioasq_factoid_qa",  # Abstractive
    "bioasq_general_qa",  # Abstractive
    "bioasq_yesno_qa",  # Y/N
    "covid_deepset_qa",  # Extractive
    "pubmedqa_qa",  # Y/N
]


def create_bio_sciriff_qa_tasks() -> dict:
    all_tasks = {}
    for task_type in BIO_SCIRIFF_QA_TASKS:
        if task_type in ["bioasq_yesno_qa", "pubmedqa_qa"]:
            primary_metric = "acc_raw"
            base_class = GenericSciRiffMC
        else:
            primary_metric = "rougeL_f1"
            base_class = GenericSciRiff

        class SciRiff(base_class):
            TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
                {"dataset_name": task_type, "primary_metric": primary_metric},
                GenericSciRiff.TASK_CONFIG_DEFAULTS,
            )

        all_tasks[f"sciriff_{task_type}"] = SciRiff
    return all_tasks


class GenericSciRiff(Task):
    VERSION = 0
    MAX_TRAINING_DOCS = 10000
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": SCIRIFF_BASE_DIR,
        "fewshot_source": None,
        "num_shots": 5,
        "split": "test",
        "context_kwargs": {"description": None},
        "generation_kwargs": {
            "max_gen_toks": 50,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["Question:", "Q:", "\n\n"],
        },
        "chat_overrides": {
            "generation_kwargs": {
                "stop_sequences": ["Question:", "Q:", "<|eot_id|>"],
            },
        },
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = {}
        for split in ["train", "validation", "test"]:
            data_path = os.path.join(
                self.task_config["dataset_path"], self.task_config["dataset_name"], f"{split}.jsonl"
            )
            data = []
            with smart_open.open(data_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            self.dataset[split] = data

    def make_metrics(self):
        self._metrics = [
            SQuADF1EMRecallMetric(**self.task_config["metric_kwargs"]),
            RougeMetric(
                metric_names=["rouge1", "rouge2", "rougeL"], **self.task_config["metric_kwargs"]
            ),
        ]
        return self._metrics

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _get_inst_question_context(self, doc):
        inst, question, context = doc["input"].split("\n\n")
        inst = inst.strip()
        question = question.replace("\n", " ").strip()
        context = context.replace("\n", " ").strip()
        assert question.startswith("Question:")
        assert context.startswith("Context:")
        return inst, question, context

    def _get_inst_para_question(self, doc):
        chunks = doc["input"].split("\n\n")
        inst = chunks[0]
        para = "\n\n".join(chunks[1:-1])
        question = chunks[-1]

        inst = inst.strip()
        question = question.replace("\n", " ").strip()
        para = para.replace("\n", " ").strip()
        assert question.startswith("Question:")
        assert para.startswith("Paragraph:")
        return inst, para, question

    def _get_inst_context_question(self, doc):
        inst, para, question = doc["input"].split("\n\n")
        inst = inst.strip()
        question = question.replace("\n", " ").strip()
        para = para.replace("\n", " ").strip()
        assert question.startswith("Question:")
        assert para.startswith("Context:")
        return inst, para, question

    def _process_doc(self, doc, index=-1):
        if self.task_config["dataset_name"] in ["bioasq_factoid_qa", "bioasq_general_qa"]:
            inst, question, context = self._get_inst_question_context(doc)
            query = inst + "\n\n" + context + "\n\n" + question + "\n\nAnswer:"
            answer = doc["output"]
            choices = None
            gold = None
        elif self.task_config["dataset_name"] in ["bioasq_yesno_qa"]:
            inst, question, context = self._get_inst_question_context(doc)
            query = (
                inst + "\n\n" + context + "\n\n" + question + " Answer with Yes or No.\n\nAnswer:"
            )
            answer = {"yes": "Yes", "no": "No"}[doc["output"]]
            choices = ["Yes", "No"]
            gold = {"yes": 0, "no": 1}[doc["output"]]
        elif self.task_config["dataset_name"] in ["covid_deepset_qa"]:
            inst, para, question = self._get_inst_para_question(doc)
            query = inst + "\n\n" + para + "\n\n" + question + "\n\nAnswer:"
            answer = doc["output"]
            choices = None
            gold = None
        elif self.task_config["dataset_name"] in ["pubmedqa_qa"]:
            if doc["input"].strip().endswith("\n\nAnswer:"):
                doc["input"] = doc["input"].strip().split("\n\nAnswer:")[0]
            inst, context, question = self._get_inst_context_question(doc)
            query = (
                inst
                + "\n\n"
                + context
                + "\n\n"
                + question
                + " Answer with Yes, No, or Maybe.\n\nAnswer:"
            )
            answer = {"YES": "Yes", "NO": "No", "MAYBE": "Maybe"}[doc["output"]]
            choices = ["Yes", "No", "Maybe"]
            gold = {"YES": 0, "NO": 1, "MAYBE": 2}[doc["output"]]
        else:
            raise NotImplementedError()
        out_doc = {
            "index": doc["_instance_id"],
            "question": query,
            "query": query,
            "answer": answer,
            "choices": choices,
            "gold": gold,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["answer"])


class GenericSciRiffMC(GenericSciRiff, MultipleChoiceTask):
    def make_metrics(self):
        return MultipleChoiceTask.make_metrics(self)

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return MultipleChoiceTask.construct_requests(self, doc, ctx, doc_id)
