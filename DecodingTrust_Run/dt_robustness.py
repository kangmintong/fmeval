import glob
import warnings
import numpy as np
import sagemaker
import torch
import json
from dataclasses import dataclass
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.data_loaders.data_config import DataConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.eval_algorithms.decodingtrust_robustness import DTRobustnessConfig, DTRobustness
from tqdm import tqdm
from datasets import load_dataset

@dataclass(frozen=True)
class HFModelConfig:
    model_name: str
    max_new_tokens: int
    remove_prompt_from_generated_text: bool = True
    do_sample: bool = False

class HuggingFaceCausalLLMModelRunner(ModelRunner):
    def __init__(self, model_config: HFModelConfig):
        self.config = model_config
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, trust_remote_code=True)
        self.model = self.model.to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generations = self.model.generate(
            **input_ids,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=self.config.do_sample,
        )
        generation_contains_input = (
            input_ids["input_ids"][0] == generations[0][: input_ids["input_ids"].shape[1]]
        ).all()
        if self.config.remove_prompt_from_generated_text and not generation_contains_input:
            warnings.warn(
                "Your model does not return the prompt as part of its generations. "
                "`remove_prompt_from_generated_text` does nothing."
            )
        if self.config.remove_prompt_from_generated_text and generation_contains_input:
            output = self.tokenizer.batch_decode(generations[:, input_ids["input_ids"].shape[1] :])[0]
        else:
            output = self.tokenizer.batch_decode(generations, skip_special_tokens=True)[0]

        with torch.inference_mode():
            input_ids = self.tokenizer(self.tokenizer.bos_token + prompt, return_tensors="pt")["input_ids"].to(self.model.device)
            model_output = self.model(input_ids, labels=input_ids)
            probability = -model_output[0].item()
        return output, probability

# Specification of models and parameters
hf_config = HFModelConfig(model_name="microsoft/phi-2", max_new_tokens=32)
model = HuggingFaceCausalLLMModelRunner(model_config=hf_config)
eval_algo =DTRobustness(DTRobustnessConfig())

TASK = "sst2"

# aggregated dataset evaluation
config = DataConfig(
    dataset_name=f"{TASK}",
    dataset_uri=f"AI-Secure/DecodingTrust",
    dataset_mime_type=MIME_TYPE_JSONLINES,
    model_input_location="sentence",
    target_output_location="label",
)
eval_output = eval_algo.evaluate(model=model, dataset_config=config, save=True, num_records=100)