from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from inference.utils import Choice, MockLogProbs, get_score_from_completion, InferenceModel, evaluate_outputs_local, \
  generate_outputs_local
from inference.generator import Generator
from inference.evaluator import Evaluator
import peft
import torch

class SharedBase(Generator, Evaluator):
    def __init__(self, generator_adapter_name, evaluator_adapter_name, generator_inference_param_dict, 
                 evaluator_inference_param_dict, prompt_template, prompt_suffix):
        Evaluator.__init__(self, None, evaluator_inference_param_dict, prompt_template, prompt_suffix)
        Generator.__init__(self, None, generator_inference_param_dict)
        self.model = None
        self.generator_adapter_name = generator_adapter_name
        self.evaluator_adapter_name = evaluator_adapter_name
        self.evaluator_inference_param_dict = evaluator_inference_param_dict
        self.generator_inference_param_dict = generator_inference_param_dict
        self.load_model()
        self.gen_adapter_label = "generator" if "generator" in self.model.peft_config else None
        self.eval_adapter_label = "evaluator"

    def load_model(self):
        peft_config = peft.PeftConfig.from_pretrained(self.evaluator_adapter_name)
        self.model_name = peft_config.base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.evaluator_adapter_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        bnb_config = BitsAndBytesConfig()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = peft.PeftModel.from_pretrained(self.model, self.evaluator_adapter_name, "evaluator")
        if self.generator_adapter_name is not None:
            self.model.load_adapter(self.generator_adapter_name, "generator")
        
    def set_adapter(self, adapter_name):
        old_adapter_name = self.model.active_adapter
        try:
            if adapter_name is not None:
                self.model.set_adapter(adapter_name)
                print(adapter_name)
                yield self.model
            else:
                with self.model.disable_adapter():
                    print("Reached here!")
                    yield self.model
        finally:
            self.model.set_adapter(old_adapter_name)

    def evaluate_outputs(self, texts):
        self.set_adapter(self.eval_adapter_label)
        return evaluate_outputs_local(self.evaluator_inference_param_dict, texts)
    
    def generate_outputs(self, text, n_tokens, n=1, batch_size=1):
        self.set_adapter(self.gen_adapter_label)
        return generate_outputs_local(self.model, self.tokenizer, self.generator_inference_param_dict, text, n_tokens, n, batch_size)
            