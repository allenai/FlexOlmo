from typing import Dict

from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY

from .tasks import (
    agi_eval,
    hatespeech,
    medmcqa,
    medqa,
    narrativeqa,
    news_gen,
    poem_gen,
    ruler,
    sciriff,
    story_gen,
    xsum,
)

new_task_registry: Dict = {
    "xsum": xsum.XSum,
    "narrativeqa": narrativeqa.NarrativeQA,
    "story_gen": story_gen.Story_Gen_LMJudge,
    "news_gen": news_gen.News_Gen_LMJudge,
    "poem_gen": poem_gen.Poem_Gen_LMJudge,
    "medqa": medqa.MedQA,
    "medmcqa:mc": medmcqa.MedMCQAMC,
    "hatespeech18": hatespeech.HateSpeech18,
    "tweet_eval_hate": hatespeech.TweetEvalHate,
    "hate_speech_offensive": hatespeech.HateSpeechOffensive,
    "hatexplain": hatespeech.Hatexplain,
    **agi_eval.create_core_agi_eval_tasks(),
    **ruler.create_ruler_tasks(),
    **sciriff.create_bio_sciriff_qa_tasks(),
}

TASK_REGISTRY.update(new_task_registry)
