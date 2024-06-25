#!/usr/bin/env python3

"""Samples from a language model using Weave tree search."""

import argparse
from functools import partial
import heapq
from itertools import chain
import math
from operator import attrgetter
import os
import random
from inference.evaluator import LocalEvaluator, RemoteEvaluator
from inference.generator import LocalGenerator, RemoteGenerator
from inference.shared_base import SharedBase

from rich import print as rprint
from rich.traceback import install

from utils import load_config


def logsumexp(xs):
    if not len(xs):
        return float("-inf")
    a = max(xs)
    return a + math.log(sum(math.exp(x - a) for x in xs))


def log_softmax(xs):
    lse = logsumexp(xs)
    return [x - lse for x in xs]


def log1mexp(a):
    if a > 0.0:
        return float("nan")
    if a == 0.0:
        return float("-inf")
    if a > -0.693:
        return math.log(-math.expm1(a))
    return math.log1p(-math.exp(a))


def log1pexp(a):
    if a < 18:
        return math.log1p(math.exp(a))
    return a


def gumbelvariate(loc=0.0, scale=1.0):
    return loc - scale * math.log(random.expovariate(1))


def get_score_from_chat_completion(response, smoothing=1.0):
    texts = [choice.message.content.lower().lstrip() for choice in response.choices]
    n_yes, n_no = 0, 0
    for text in texts:
        if text.startswith("yes"):
            n_yes += 1
        elif text.startswith("no"):
            n_no += 1
    return math.log(n_yes + smoothing) - math.log(n_no + smoothing)


class TreeNode:
    max_id = 0

    def __init__(self, text, parent=None):
        self.id = type(self).max_id
        type(self).max_id += 1
        self.text = text
        if parent is None:
            self.root = self
            self.depth = 0
            self.committed = True
            self.gumbel = 0.0
        else:
            self.root = parent.root
            self.depth = parent.depth + 1
            self.committed = False
            self.gumbel = gumbelvariate()
        self.parent = parent
        self.children = []
        self.pruned = False
        self.score = float("-inf")
        self.logit = float("-inf")
        self.phi = 0.0
        self.g_phi = 0.0

    @property
    def priority(self):
        return self.logit + self.gumbel

    def __lt__(self, other):
        a = self.committed and not self.children, self.priority
        b = other.committed and not other.children, other.priority
        # Reversed so that heapq will be a max heap
        return a > b

    def update_phi(self):
        if not self.children:
            return
        logps = log_softmax([child.logit for child in self.children])
        for child, logp in zip(self.children, logps):
            child.phi = self.phi + logp
            child.update_phi()

    def set_score(self, score, temperature):
        self.score = score
        self.logit = score / temperature
        # Backpropagate logit
        node = self.parent
        while node and not node.committed:
            node.logit = logsumexp([child.logit for child in node.children])
            node = node.parent

    def set_pruned(self):
        self.pruned = True
        for child in self.children:
            if not child.committed:
                child.set_pruned()

    def nodes(self):
        node_list = [self]
        for child in self.children:
            node_list.extend(child.nodes())
        return node_list

    def leaves(self):
        return [node for node in self.nodes() if not node.children]

    def branch_text(self, include_root=False):
        branch_texts = [self.text]
        node = self
        while node.parent:
            node = node.parent
            branch_texts.insert(0, node.text)
        if include_root:
            return "".join(branch_texts)
        else:
            return "".join(branch_texts[1:])

    def serialize_branch(self):
        branch_nodes = [{"depth": self.depth,
                         "text": self.text,
                         "score": self.score,
                         }]
        node = self
        while node.parent:
            node = node.parent
            serial_node = {"depth": node.depth,
                           "text": node.text,
                           "score": node.score,
                           }
            branch_nodes.append(serial_node)
        branch_nodes.reverse()
        return branch_nodes

def weave_tree_search(
    tree,
    generate_fn,
    evaluate_fn,
    budget,
    round_budget,
    n_expand,
    beam_width,
    max_lookahead,
    temperature,
):
    if max_lookahead < 1:
        raise ValueError("max_lookahead must be at least 1")

    print("====== Generating with Weave ======")
    if tree.logit == float("-inf"):
        root_score = evaluate_fn([(tree.root.text, tree.branch_text(include_root=False))])[0]
        tree.set_score(root_score, temperature)
    beam = [tree]
    round = 0

    while budget:
        # Set up round
        rprint(f"=== Round {round} starting ===")
        round_budget_remaining = round_budget
        nodes = [
            [node for node in tree.leaves() if node.depth < round + max_lookahead]
            for tree in beam
        ]
        heap = list(chain.from_iterable(nodes))
        heapq.heapify(heap)

        # Expand nodes until round budget is exhausted
        while budget > 0 and round_budget_remaining > 0 and heap:
            rprint(
                f"Budget: {budget}, round budget: {round_budget_remaining}, queue: {len(heap)}"
            )

            # Selection - Select the node to expand
            chosen = heapq.heappop(heap)
            rprint(
                f"Chose node {chosen.id} with score {chosen.score:.4f}, priority {chosen.priority:.4f}"
            )

            # Expansion - Expand the selected node
            n_expand_cur = min(n_expand, budget, round_budget_remaining)
            texts = generate_fn(chosen.branch_text(include_root=True), n=n_expand_cur)
            scores = evaluate_fn(
                [(chosen.root.text, chosen.branch_text(include_root=False) + text)
                for text in texts]
            )
            for text, score in zip(texts, scores):
                new_child = TreeNode(text, chosen)
                chosen.children.append(new_child)
                new_child.set_score(score, temperature)
                if new_child.depth < round + max_lookahead:
                    heapq.heappush(heap, new_child)
                rprint(
                    f"New child {chosen.id}->{new_child.id} has score {new_child.score:.4f}, priority {new_child.priority:.4f}"
                )
                budget -= 1
                round_budget_remaining -= 1

        # Round over, sample beam_width nodes (top-down sampling), prune the rest
        expansions = []
        for node in beam:
            node.update_phi()
            if not node.children:
                expansions.append(node)
                continue
            for child in node.children:
                child.g_phi = child.phi + child.gumbel
                expansions.append(child)
            z = max(child.g_phi for child in node.children)
            for child in node.children:
                v = node.g_phi - child.g_phi + log1mexp(child.g_phi - z)
                child.g_phi = node.g_phi - max(0.0, v) - log1pexp(-abs(v))
                rprint(
                    f"Beam candidate {child.id} has logit {child.logit:.4f}, phi {child.phi:.4f}, and g_phi {child.g_phi:.4f}"
                )
        expansions.sort(key=attrgetter("g_phi"), reverse=True)
        beam = expansions[:beam_width]
        for node in beam:
            node.committed = True
        for node in expansions[beam_width:]:
            node.set_pruned()

        round += 1

        score_s = ", ".join(f"{node.score:.4f}" for node in beam)
        rprint(f"Scores of beam: [{score_s}]")

    # Sample beam_width nodes (bottom-up sampling)
    nodes = sorted(
        chain.from_iterable(tree.leaves() for tree in beam),
        key=lambda node: node.phi + node.gumbel,
        reverse=True,
    )
    return nodes[:beam_width]

def init_gen_eval(config, evaluation_prompt):
    if config['shared_base_adapters']:
        generator_adapter_name = config['generator']['model_name'] if config['generator']['is_adapter'] else None
        evaluator_adapter_name = config['evaluator']['model_name'] if config['evaluator']['is_adapter'] else None
        generator = evaluator = SharedBase(generator_adapter_name, evaluator_adapter_name,
                 config['generator']['inference_params'], config['evaluator']['inference_params'], evaluation_prompt, 
                 "<|end|>")
    if config['generator']['api_base'] is None:
        generator = LocalGenerator(config['generator']['model_name'], config['generator']['load_dtype'],
                                   config['generator']['inference_params'])
    else:
        generator = RemoteGenerator(config['generator']['model_name'], config['generator']['api_base'],
                                    config['generator']['api_key'], evaluator_adapter_name)
    if config['evaluator']['api_base'] is None:
        evaluator = LocalEvaluator(config['evaluator']['model_name'], config['evaluator']['load_dtype'],
                                   config['evaluator']['inference_params'], evaluation_prompt, "<|end|>")
    else:
        evaluator = RemoteEvaluator(config['evaluator']['model_name'], config['evaluator']['api_base'],
                                    config['evaluator']['api_key'], config['evaluator']['inference_params'],
                                    evaluation_prompt, "<|end|>")
    return generator, evaluator

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", help="Path to the configuration file", default="configs/gpt2.json")
    parser.add_argument("--verbose", "-v", help="Print API responses", action="store_true")
    args = parser.parse_args()

    os.environ["BITSANDBYTES_NOWELCOME"] = "1"

    config = load_config(args.config)

    w_p = config['init_weave_param']

    generator, evaluator = init_gen_eval(config, w_p['evaluation_prompt'])

    system_prompt = ""
    prompt = "Once upon a time, there was a woman who"

    def evaluate_without_system_prompt(texts):
        stripped_texts = [text[len(system_prompt) :] for text in texts]
        return evaluator.evaluate_outputs(stripped_texts)

    root_text = system_prompt + prompt
    tree = TreeNode(root_text)
    try:
        branches = weave_tree_search(
            tree=tree,
            generate_fn=partial(generator.generate_outputs, n_tokens=w_p['n_tokens']),
            evaluate_fn=evaluator.evaluate_outputs,
            budget=w_p['budget'],
            round_budget=w_p['round_budget'],
            n_expand=w_p['n_expand'],
            beam_width=w_p['beam_width'],
            max_lookahead=w_p['max_lookahead'],
            temperature=w_p['temperature']
        )

        # Print results
        print()
        for branch in branches:
            rprint(f"====== Branch with score: {branch.score:.4f} ======")
            text = branch.branch_text(include_root=True)
            print(text)
            print()

        # Write graphviz file
        with open("out.gv", "w") as f:
            print("digraph {", file=f)
            print("    rankdir=LR", file=f)
            for node in tree.nodes():
                color = "black"
                if node.committed:
                    color = "blue"
                if node in branches:
                    color = "red"
                fillcolor = "lightgray" if node.pruned else "white"
                print(
                    f'    "{node.id}" [label=<{node.score:.4f}<br/><font point-size="8">{node.phi:.4f}</font>>,style=filled,color={color},fillcolor={fillcolor}]',
                    file=f,
                )
                for child in node.children:
                    print(f'    "{node.id}" -> "{child.id}"', file=f)
            print("}", file=f)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    install()
    main()
