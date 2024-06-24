import math
from functools import partial
from tqdm import tqdm
import torch
from transformers.generation.streamers import BaseStreamer
from abc import ABC

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


@torch.no_grad()
def generate_outputs_local(model, tokenizer, inference_param_dict, text, n_tokens, n=1, batch_size=1):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096 - n_tokens,
    ).to(model.device)

    outputs = []
    with ProgressBarStreamer(total=n_tokens * n) as pbar:
        for i in range(0, n, batch_size):
            n_batch = min(batch_size, n - i)
            input_ids = inputs.input_ids.tile((n_batch, 1))
            attention_mask = inputs.attention_mask.tile((n_batch, 1))
            outputs_batch = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                min_new_tokens=n_tokens,
                max_new_tokens=n_tokens,
                pad_token_id=tokenizer.eos_token_id,
                streamer=pbar,
                **inference_param_dict,
            )
            outputs.append(outputs_batch)
    outputs = torch.cat(outputs)
    out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in outputs]
    in_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    return [out_texts[i][in_length:] for i in range(len(out_texts))]


@torch.no_grad()
def evaluate_outputs_local(self, texts):

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