from abc import ABC
import torch
import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference.utils import InferenceModel, ProgressBarStreamer


class Generator(InferenceModel, ABC):
    def __init__(self, model_name, inference_param_dict):
        self.model_name = model_name
        self.inference_param_dict = inference_param_dict

    def generate_outputs(self, text, n_tokens, n=1, verbose=False):
        pass


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
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096 - n_tokens,
        ).to(self.model.device)

        outputs = []
        with ProgressBarStreamer(total=n_tokens * n) as pbar:
            for i in range(0, n, batch_size):
                n_batch = min(batch_size, n - i)
                input_ids = inputs.input_ids.tile((n_batch, 1))
                attention_mask = inputs.attention_mask.tile((n_batch, 1))
                outputs_batch = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    min_new_tokens=n_tokens,
                    max_new_tokens=n_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=pbar,
                    **self.inference_param_dict,
                )
                outputs.append(outputs_batch)
        outputs = torch.cat(outputs)
        out_texts = [self.tokenizer.decode(toks, skip_special_tokens=True)
                     for toks in outputs]
        in_length = len(self.tokenizer.decode(
            inputs.input_ids[0], skip_special_tokens=True))
        return [out_texts[i][in_length:] for i in range(len(out_texts))]


class RemoteGenerator(Generator):
    def __init__(self, model_name, api_base, api_key, inference_param_dict):
        super().__init__(model_name, inference_param_dict)
        self.api_base = api_base
        self.api_key = api_key

    def generate_outputs(self, text, n_tokens, n=1, verbose=False):
        payload = {"n": n,
                   "max_tokens": n_tokens,
                   "model": self.model_name,
                   "prompt": text,
                   "stream": False,
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
