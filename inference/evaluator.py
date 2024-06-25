from abc import ABC
import torch
import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference.utils import Choice, MockLogProbs, get_score_from_completion, InferenceModel, evaluate_outputs_local

class Evaluator(InferenceModel, ABC):
    def __init__(self, model_name, inference_param_dict, prompt_template, prompt_suffix):
        self.model_name = model_name
        self.inference_param_dict = inference_param_dict
        self.template = prompt_template
        self.suffix = prompt_suffix

    def score_prompt_fn(self, prompt, response):
        pass

    def evaluate_outputs(self, texts):
        pass


class LocalEvaluator(Evaluator):
    def __init__(self, model_name, load_dtype, inference_param_dict, prompt_template, prompt_suffix):
        super().__init__(model_name, inference_param_dict, prompt_template, prompt_suffix)
        self.load_dtype = load_dtype
        print("Loading evaluator model...")
        self.tokenizer, self.model = self.load_evaluator()

    def load_evaluator(self):
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

    def score_prompt_fn(self, prompt, response):
        template_toks = self.tokenizer(self.template,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=4096)
        template_length = len(template_toks.input_ids[0])
        response_toks = self.tokenizer(response,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=4096 - template_length)
        response_length = len(response_toks.input_ids[0])
        prompt_toks = self.tokenizer(prompt,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=4096 - template_length - response_length)
        response = self.tokenizer.decode(response_toks.input_ids[0], skip_special_tokens=True)
        prompt = self.tokenizer.decode(prompt_toks.input_ids[0], skip_special_tokens=True)

        return self.template.format(prompt = prompt, response = response) + self.suffix

    def evaluate_outputs(self, texts):
        return evaluate_outputs_local(self.model, self.tokenizer, self.inference_param_dict, texts)


class RemoteEvaluator(Evaluator):
    def __init__(self, model_name, api_base, api_key, inference_param_dict, prompt_template, prompt_suffix):
        super().__init__(model_name, inference_param_dict, prompt_template, prompt_suffix)
        self.api_base = api_base
        self.api_key = api_key

    def score_prompt_fn(self, prompt, response):
        return self.template.format(prompt = prompt, response = response) + self.suffix

    def evaluate_outputs(self, texts, n=1, verbose=False):
        scores = []
        for text in texts:
            prompts = [self.score_prompt_fn(text[0], text)]
            payload = {"n":n,
                      "max_tokens": 1,
                      "model":self.model_name,
                      "prompt":prompts,
                      "stream":False,
                      "logprobs":100,
                      **self.inference_param_dict
                      }
            if self.api_key is None:
                header = {}
            else:
                header = {"Authorization": f"Bearer {self.api_key}"}
            header["Content-Type"] = "application/json"
            response = requests.post(self.api_base,
                                    headers=header,
                                    data=json.dumps(payload))
            choices = []
            if verbose:
                print("EVAL API RESPONSE:", response.json())
            for choice in response.json()["choices"]:
                choice_o = Choice()
                mocklogprobs_o = MockLogProbs()
                choice_o.logprobs = mocklogprobs_o
                choice_o.logprobs.top_logprobs = choice["logprobs"]["top_logprobs"]
                choices.append(choice_o)
            scores.append(torch.tensor([get_score_from_completion(choice) for choice in choices]))
        # TODO: Return these unpooled so the separate components can be stored in the
        # weave tree
        return torch.stack(scores).mean(dim=1)