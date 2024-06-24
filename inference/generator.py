from abc import ABC
import torch
import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference.utils import InferenceModel, ProgressBarStreamer, generate_outputs_local

class Generator(InferenceModel, ABC):
    def __init__(self, model_name, inference_param_dict):
        super().__init__(model_name, inference_param_dict)


class LocalGenerator(Generator):
    def __init__(self, model_name, load_dtype, inference_param_dict):
        super().__init__(model_name, inference_param_dict)
        self.load_dtype = load_dtype
        print("Loading generator model...")
        self.tokenizer, self.model = self.load_generator()

    def load_generator(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            load_in_4bit=self.load_dtype == "int4",
            load_in_8bit=self.load_dtype == "int8",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        return tokenizer, model

    @torch.no_grad()
    def generate_outputs(self, text, n_tokens, n=1, batch_size=1):
        return generate_outputs_local(self.model, self.tokenizer, self.inference_param_dict, text, n_tokens, n, batch_size)


class RemoteGenerator(Generator):
    def __init__(self, model_name, api_base, api_key, inference_param_dict):
        super().__init__(model_name, inference_param_dict)
        self.api_base = api_base
        self.api_key = api_key

    def generate_outputs(self, text, n_tokens, n=1, verbose=False):
        payload = {"n":n,
                  "max_tokens": n_tokens,
                  "model":self.model_name,
                  "prompt":text,
                  "stream":False,
                  **self.inference_params_dict
                  }
        if self.api_key is None:
            header = {}
        else:
            header = {"Authorization": f"Bearer {self.api_key}"}
        header["Content-Type"] = "application/json"
        response = requests.post(self.api_base,
                                headers=header,
                                data=json.dumps(payload))
        if verbose:
            print("GEN API RESPONSE:", response.json())
        texts = [choice["text"] for choice in response.json()["choices"]]
        return texts
