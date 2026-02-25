# import easyapi
import logging
import os

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Warning: vllm is not installed.")
    LLM = None
    SamplingParams = None

log = logging.getLogger(__name__)


class LanguageModel(object):
    def __init__(
        self,
        model_name,
        model_type="vllm",
        device="cuda:0",
        max_input_len=None,
        enable_chunked_prefill=True,
    ):
        if model_type == "vllm" and LLM is None:
            model_type = "hf"
            log.warning(
                "vllm is not installed. Using transformers instead. "
                "Install vllm to use it as the model backend."
            )
        self.model_name = model_name
        self.model_type = model_type

        self.hf_access_token = os.environ.get("HF_TOKEN", "")
        self.device = device
        self.max_input_len = max_input_len  # TODO: Max context len and (left) truncation only supported for vllm currently
        self.enable_chunked_prefill = enable_chunked_prefill

        self.llm = None

    def load_model_and_tokenizer(self):
        # TODO: support other models
        model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.hf_access_token)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=self.hf_access_token, padding_side="left"
        )
        eos_token_id = model.config.eos_token_id
        tokenizer.pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        return tokenizer, model

    def load_model(self, **kwargs):
        if not self.llm:
            if self.model_type == "vllm" and LLM is not None:
                if self.model_name.startswith("allenai/"):
                    self.max_input_len = 4096
                if self.max_input_len:
                    kwargs = kwargs | {"max_model_len": self.max_input_len}
                self.llm = LLM(
                    model=self.model_name, tensor_parallel_size=torch.cuda.device_count(), **kwargs
                )
            elif self.model_type == "hf":
                tokenizer, model = self.load_model_and_tokenizer()
                self.llm = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer, device=self.device
                )
            # elif self.model_type == 'easyapi':
            #     self.llm = easyapi.Api('jupiter')
            #     print(f"Launching {self.model_name}...")
            #     self.llm.launch_model(self.model_name, gpus=1, hf_token=self.hf_access_token) # launch on jupiter
            #     while not self.llm.has_model(self.model_name):
            #         print(f"Waiting for {self.model_name} to be launched...")
            #         time.sleep(10)

            #     print(f"{self.model_name} loaded!")
            else:
                raise NotImplementedError()

    def __call__(self, queries, batch_size=32, max_output_length=512, temperature=1.0, **kwargs):
        inputs = queries  # [datum["query"] for datum in queries]

        if not self.llm:
            self.load_model()

        # for allenai
        if self.model_name.startswith("allenai/"):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenized_inputs = tokenizer(inputs)["input_ids"]
            cnt = 0
            for i, (input_, tokens) in enumerate(zip(inputs, tokenized_inputs)):
                if len(tokens) >= 4096 - 10:
                    cnt += 1
                    while True:
                        input_ = " ".join(input_.split(" ")[10:])
                        tokenized_input = tokenizer([input_])["input_ids"][0]
                        if len(tokenized_input) >= 4096 - 10:
                            continue
                        else:
                            inputs[i] = input_
                            break
            print(f"{cnt}/{len(inputs)} inputs got truncated!")

        if self.model_type == "vllm" and SamplingParams is not None:
            sampling_params_dict = {"temperature": temperature, "max_tokens": max_output_length}
            if self.model_name.startswith("allenai/"):
                sampling_params_dict["truncate_prompt_tokens"] = 4096 - 10
            if self.max_input_len:  # model_name.startswith("allenai/"):
                sampling_params_dict["truncate_prompt_tokens"] = self.max_input_len

            sampling_params = SamplingParams(**sampling_params_dict, **kwargs)
            assert self.llm is not None
            outputs = self.llm.generate(inputs, sampling_params, use_tqdm=False)
            outputs = [output.outputs[0].text.strip() for output in outputs]
            return outputs

        elif self.model_type == "hf":
            inputs_ds = Dataset.from_list([{"input": input} for input in inputs])
            outputs = []
            assert self.llm is not None
            for batch_out in tqdm(
                self.llm(
                    KeyDataset(inputs_ds, "input"),
                    batch_size=batch_size,
                    max_new_tokens=max_output_length,
                    do_sample=False,
                    return_full_text=False,
                )
            ):
                outputs.extend([out["generated_text"].strip() for out in batch_out])
            return outputs

        else:
            raise NotImplementedError()


if __name__ == "__main__":
    lm = LanguageModel("meta-llama/Llama-3.3-70B-Instruct", max_input_len=2048)
    gen = lm(["Llama 3.3 is"])
    print(gen)
