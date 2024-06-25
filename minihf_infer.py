import os
import json
import time
import random
import hashlib
import zipfile
from contextlib import contextmanager
from functools import partial
import argparse
import utils
from flask import Flask, request, jsonify, make_response, render_template
from tqdm import tqdm
import torch
import torch.nn as nn
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig
from inference.weave import weave_tree_search, init_gen_eval, TreeNode
from lora_tune import lora_tune_evaluator
from dataset import ZippedConversationsDataset


def load_models(config):
    global evaluator, evaluate_fn, generator, generate_fn
    if config['evaluator']['model_name'] is None:
        evaluate_fn = None
    else:
        generator, evaluator = init_gen_eval(
            config, config['init_weave_param']['evaluation_prompt'])
        generate_fn = generator.generate_outputs
        evaluate_fn = evaluator.evaluate_outputs


def create_app(config, device):
    app = Flask(__name__)

    @app.route("/generate", methods=['OPTIONS', 'POST'])
    def generate():
        if request.method == 'OPTIONS':
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "*")
            response.headers.add("Access-Control-Allow-Methods", "*")
            return response
        if request.method == 'POST':
            params = request.get_json()
            prompt = params['prompt']
            if 'prompt_node' in params:
                prompt_node = params['prompt_node']
            else:
                prompt_node = False
            new_tokens = int(params['new_tokens'])
            n_outputs = int(params['weave_beam_width'])
            base_model_name = config['generator']['model_name']
            if base_model_name is None:
                base_model_name = generator[1].active_peft_config.base_model_name_or_path
            outs = generate_fn(prompt, new_tokens, n=n_outputs)
            batch = []
            if prompt_node:
                timestamp = str(time.time())
                id_ = hashlib.md5(
                    (prompt + timestamp).encode("UTF-8")).hexdigest()
                batch.append({"id": id_,
                              "prompt": prompt,
                              "text": "",
                              "timestamp": timestamp,
                              "nodes": []})
            for out in outs:
                timestamp = str(time.time())
                id_ = hashlib.md5(out.encode("UTF-8")).hexdigest()
                batch.append({"id": id_,
                              "base_model": base_model_name,
                              "prompt": prompt,
                              "text": out,
                              "timestamp": timestamp,
                              "nodes": []})
            # TODO: Proper CORS
            response = jsonify(utils.jsonify_tensors(batch))
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

    @app.route("/weave", methods=['OPTIONS', 'POST'])
    def weave():
        if evaluate_fn is None:
            # return 400 error
            response = make_response()
            response.status_code = 422
            response.data = "Evaluator model not specified, cannot perform weave"
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
        if request.method == 'OPTIONS':
            response = make_response()
            # TODO: Have the interface served by the server on GET request
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "*")
            response.headers.add("Access-Control-Allow-Methods", "*")
            return response
        if request.method == 'POST':
            params = request.get_json()
            prompt = params['prompt']
            context = params['context']
            if 'prompt_node' in params:
                prompt_node = params['prompt_node']
            else:
                prompt_node = False
            evaluation_prompt = params['evaluationPrompt']
            full_prompt = context + " " + prompt
            tree = TreeNode(full_prompt)
            weave_param_defaults = {"weave_n_tokens": 32, "weave_budget": 72,
                                    "weave_round_budget": 24, "weave_n_expand": 8,
                                    "weave_beam_width": 1, "weave_max_lookahead": 3,
                                    "weave_temperature": 0.25}
            wp = {}
            for key in weave_param_defaults.keys():
                if key in params:
                    try:
                        wp[key] = int(params[key])
                    except ValueError:
                        wp[key] = float(params[key])
                else:
                    wp[key] = weave_param_defaults[key]
            branches = weave_tree_search(tree=tree,
                                         generate_fn=partial(generate_fn,
                                                             n_tokens=wp["weave_n_tokens"]),
                                         evaluate_fn=evaluate_fn,
                                         budget=wp["weave_budget"],
                                         round_budget=wp["weave_round_budget"],
                                         n_expand=wp["weave_n_expand"],
                                         beam_width=wp["weave_beam_width"],
                                         max_lookahead=wp["weave_max_lookahead"],
                                         temperature=wp["weave_temperature"])
            batch = []
            if prompt_node:
                timestamp = str(time.time())
                id_ = hashlib.md5(
                    (prompt + timestamp).encode("UTF-8")).hexdigest()
                batch.append({"id": id_,
                              "prompt": prompt,
                              "evaluationPrompt": evaluation_prompt,
                              "text": "",
                              "timestamp": timestamp,
                              "nodes": []})
            for branch in branches:
                branch_text = branch.branch_text()
                timestamp = str(time.time())
                id_ = hashlib.md5(
                    (branch_text + timestamp).encode("UTF-8")).hexdigest()
                batch.append({"id": id_,
                              "prompt": prompt,
                              "evaluationPrompt": evaluation_prompt,
                              "text": branch_text,
                              "timestamp": timestamp,
                              "nodes": branch.serialize_branch()})
            # TODO: Proper CORS
            response = jsonify(utils.jsonify_tensors(batch))
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

    @app.route("/check-tokens", methods=['OPTIONS', 'POST'])
    def check_tokens():
        if request.method == 'OPTIONS':
            response = make_response()
            # TODO: Have the interface served by the server on GET request
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "*")
            response.headers.add("Access-Control-Allow-Methods", "*")
            return response
        if request.method == 'POST':
            params = request.get_json()
            text = params['text']
            if isinstance(generator, tuple):
                tokenizer, _ = generator
            else:
                tokenizer = generator.tokenizer
            inputs = tokenizer([text] * 1, return_tensors="pt",
                               truncation=True, max_length=4096).to(device)
            # TODO: Proper CORS
            response = jsonify(inputs['input_ids'][0].shape[0])
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

    @app.route("/")
    def index():
        return render_template('minihf.html', **config['init_weave_param'])
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/jdp.json")
    parser.add_argument("--device", "-d", type=str,
                        default=utils.auto_device())
    parser.add_argument("--port", "-p", type=int, default=5000)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    load_models(config)
    app = create_app(config, args.device)
    app.run(port=args.port)


if __name__ == "__main__":
    main()
