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
