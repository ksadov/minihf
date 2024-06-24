import torch
import json
import requests
import math
from functools import partial

from abc import ABC
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from tqdm import tqdm

class Choice:
    pass

class MockLogProbs:
    pass

def get_scores_from_logits(logits, pos_tokens, neg_tokens, alpha=float("-inf")):
    logits = logits[:, -1, :].float()
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    pos_logits = torch.logsumexp(logits[:, pos_tokens], dim=-1)
    neg_logits = torch.logsumexp(logits[:, neg_tokens], dim=-1)
    alpha = logits.new_tensor(alpha)
    return torch.logaddexp(pos_logits, alpha) - torch.logaddexp(neg_logits, alpha)


get_scores_from_logits_gpt2 = partial(
    get_scores_from_logits,
    pos_tokens=[3363, 3763, 5297, 8505, 21560, 43335],
    neg_tokens=[645, 1400, 2949, 3919, 8005, 15285],
)


get_scores_from_logits_neox = partial(
    get_scores_from_logits,
    pos_tokens=[4374, 4754, 6279, 9820, 22487, 24239],
    neg_tokens=[642, 1621, 2302, 2369, 7651, 7716],
)


get_scores_from_logits_llama = partial(
    get_scores_from_logits,
    pos_tokens=[3582, 3869, 4874, 8241, 21143, 22483],
    neg_tokens=[694, 1217, 1939, 3782, 6632, 11698],
)


get_scores_from_logits_openllama = partial(
    get_scores_from_logits,
    pos_tokens=[4583, 6464, 9257, 12075, 27214],
    neg_tokens=[697, 1398, 3976, 5258, 9332, 14928],
)


get_scores_from_logits_falcon = partial(
    get_scores_from_logits,
    pos_tokens=[4879, 5007, 5159, 9109, 31792, 41489],
    neg_tokens=[658, 1684, 2749, 2929, 9395, 10630],
)

get_scores_from_logits_mistral = partial(
    get_scores_from_logits,
    # 'Y', 'Yes', 'yes'
    pos_tokens=[627, 5592, 5081],
    # 'NO', 'No', 'no'
    neg_tokens=[7929, 1770, 708],
)

def get_score_from_completion(choice):
    p_yes, p_no, p_all = 0.0, 0.0, 0.0
    for token, logprob in choice.logprobs.top_logprobs[0].items():
        token = token.lower().lstrip()
        prob = math.exp(logprob)
        p_all += prob
        if token.startswith("yes"):
            p_yes += prob
        elif token.startswith("no"):
            p_no += prob
    p_yes = p_yes if p_yes else 1 - p_all
    p_no = p_no if p_no else 1 - p_all
    if (p_yes <= 0) or (p_no <= 0):
        return float("nan")
    return math.log(p_yes) - math.log(p_no)

class ProgressBarStreamer(BaseStreamer):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs.setdefault("unit", "tok")
        self.next_tokens_are_prompt = True
        self.pbar = None

    def __enter__(self):
        self.pbar = tqdm(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.close()

    def put(self, value):
        if not self.next_tokens_are_prompt:
            self.pbar.update(value.numel())
        self.next_tokens_are_prompt = False

    def end(self):
        self.next_tokens_are_prompt = True

class InferenceModel(ABC):
    def __init__(self, model_name, inference_param_dict):
        self.model_name = model_name
        self.inference_param_dict = inference_param_dict

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
        out_texts = [self.tokenizer.decode(toks, skip_special_tokens=True) for toks in outputs]
        in_length = len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        return [out_texts[i][in_length:] for i in range(len(out_texts))]

class Evaluator(InferenceModel, ABC):
    def __init__(self, model_name, inference_param_dict, prompt_template, prompt_suffix):
        super().__init__(model_name, inference_param_dict)

    def score_prompt_fn(self, prompt, response):
        pass

class LocalEvaluator(Evaluator):
    def __init__(self, model_name, load_dtype, inference_param_dict, prompt_template, prompt_suffix):
        super().__init__(model_name, inference_param_dict, prompt_template, prompt_suffix)
        self.load_dtype = load_dtype
        print("Loading evaluator model...")
        self.tokenizer, self.model = self.load_evaluator()
        self.template = prompt_template
        self.suffix = prompt_suffix

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

    @torch.no_grad()
    def evaluate_outputs(self, texts):

        if self.tokenizer.vocab["yes"] == 8505:
            get_scores_from_logits = get_scores_from_logits_gpt2
        elif self.tokenizer.vocab["yes"] == 9820:
            get_scores_from_logits = get_scores_from_logits_neox
        elif self.tokenizer.vocab["yes"] == 3582:
            get_scores_from_logits = get_scores_from_logits_llama
        elif self.tokenizer.vocab["yes"] == 9257:
            get_scores_from_logits = get_scores_from_logits_openllama
        elif self.tokenizer.vocab["yes"] == 9109:
            get_scores_from_logits = get_scores_from_logits_falcon
        elif self.tokenizer.vocab["yes"] == 9780:
            get_scores_from_logits = get_scores_from_logits_mistral
        else:
            raise ValueError("Unknown model type")

        scores = []
        for score_prompt_fn in [self.score_prompt_fn]:
            prompts = [score_prompt_fn(text[0], text[1]) for text in texts]
            # TODO: currently, we take inference_params for evaluate_outputs_api, but not for evaluate_outputs
            # is there a reason to prefer using forward() over generate() here?
            tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).input_ids.to(self.model.device)
            logits = self.model(tokens).logits
            scores.append(
                torch.tensor(
                    [score.item() for score in get_scores_from_logits(logits)]))
        # TODO: Return these unpooled so the separate components can be stored in the
        # weave tree
        return torch.stack(scores).mean(dim=0)

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

class RemoteEvaluator(Evaluator):
    def __init__(self, model_name, api_base, api_key, inference_param_dict, prompt_template, prompt_suffix):
        super().__init__(model_name, inference_param_dict, prompt_template, prompt_suffix)
        self.template = prompt_template
        self.suffix = prompt_suffix
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

def make_gen_eval_fns(config, evaluation_prompt):
    if config['generator']['api_base'] is None:
        generator = LocalGenerator(config['generator']['model_name'], config['generator']['load_dtype'],
                                   config['generator']['inference_params'])
    else:
        generator = RemoteGenerator(config['generator']['model_name'], config['generator']['api_base'],
                                    config['generator']['api_key'], config['generator']['inference_params'])

    if config['evaluator']['api_base'] is None:
        evaluator = LocalEvaluator(config['evaluator']['model_name'], config['evaluator']['load_dtype'],
                                   config['evaluator']['inference_params'], evaluation_prompt, "<|end|>")
    else:
        evaluator = RemoteEvaluator(config['evaluator']['model_name'], config['evaluator']['api_base'],
                                    config['evaluator']['api_key'], config['evaluator']['inference_params'],
                                    evaluation_prompt, "<|end|>")
    return generator.generate_outputs, evaluator.evaluate_outputs