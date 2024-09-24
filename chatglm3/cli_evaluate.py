import os
import json
import torch
import argparse
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer, HfArgumentParser
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_model(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = model.to('cuda')
    return tokenizer, model

def load_pt2(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, pre_seq_len=model_args.pre_seq_len)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(model_args.checkpoint_path, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.to('cuda')
    return tokenizer, model

def load_lora(model_args, peft_args):
    tokenizer, model = load_model(model_args)
    model = model.half()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=peft_args.lora_rank,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        target_modules=["query_key_value"],
    )
    model = get_peft_model(model, peft_config)
    if os.path.exists(os.path.join(model_args.checkpoint_path, "pytorch_model.bin")):
        model.load_state_dict(torch.load(os.path.join(model_args.checkpoint_path, "pytorch_model.bin")), strict=False)
    model = model.to('cuda')
    return tokenizer, model

def chat(tokenizer, model, query, history, role):
    # Tokens representing the end of dialogue: eos for generation end; user for waiting for user input; observation for waiting for tool call results
    eos_token_id = [tokenizer.eos_token_id, 
                    tokenizer.get_command("<|user|>"), 
                    tokenizer.get_command("<|observation|>")]
    # Call tokenizer to convert text dialogue to token sequence for model input
    inputs = tokenizer.build_chat_input(query, history=history, role=role)
    inputs = inputs.to('cuda')
    # Generate output token sequence
    outputs = model.generate(**inputs, max_length=4096, eos_token_id=eos_token_id)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
    # Use tokenizer to decode back to text
    response = tokenizer.decode(outputs)
    history.append({"role": role, "content": query})
    # Split dialogue based on assistant token
    for response in response.split("<|assistant|>"):
        # Split metadata and response based on newline character
        splited = response.split("\n", maxsplit=1)
        if len(splited) == 2:
            metadata, response = splited
        else:
            metadata = ""
            response = splited[0]
        # If metadata is empty, the response contains reply text
        if not metadata.strip():
            response = response.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": response})
        # If metadata is not empty, the response contains tool call, written in Python syntax as a tool_call function
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": response})
            response = "\n".join(response.split("\n")[1:-1])
            # Used to extract the parameter dict filled by the model
            def tool_call(**kwargs):
                return kwargs
            try:
                parameters = eval(response)
            except:
                parameters = {}
            response = {"name": metadata.strip(), "parameters": parameters}
    return response, history

class Evaluator:
    """
    Calculate slot and reply business metrics
    """
    def __init__(self, tokenizer, model, data_path):
        self.tokenizer = tokenizer
        self.model = model
        self.data_path = data_path

    def _bleu4(self, pred, label):
        pred = pred.strip()
        label = label.strip()

        hypothesis = list(pred)
        reference = list(label)

        if len(hypothesis) == 0 or len(reference) == 0:
            return 0

        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
        return bleu_score

    def _slot_accuracy(self, pred, label):
        correct = 0
        if pred is not None:
            for k, v in pred.items():
                if v is None:
                    continue
                if label and k in label:
                    if not isinstance(v,list):
                        correct += int(v==label[k])
                    else:
                        for t in v:
                            correct += int(t in label[k])

        pred_slots = sum(len(v) if isinstance(v, list) else 1 for v in pred.values()) if pred else 0
        true_slots = sum(len(v) if isinstance(v, list) else 1 for v in label.values()) if label else 0

        return correct, pred_slots, true_slots

    def evaluate(self):
        score_dict = {
            "slot_P": None,
            "slot_R": None,
            "slot_F1": None,
            "bleu-4": None,
        }
        bleu_scores = []
        true_slot_count = 0
        pred_slot_count = 0
        correct_slot_count = 0

        # Read test set
        with open(self.data_path,'r') as f:
            test_data = [json.loads(line) for line in f]

        system_prompt = 'Answer the following questions as best as you can. You have access to the following tools:\n'

        for data in tqdm(test_data):
            dialog = data['conversations']
            # Combine to form system message
            tools_prompt = json.dumps(data['tools'],ensure_ascii=False)
            system_message = {'role': 'system', 'content': system_prompt+tools_prompt}
            history = [system_message]
            pred_slot, label_slot = {}, {}
            pred_reply, label_reply = "", ""
            for turn in dialog:
                if turn['role'] == 'user':
                    # Current turn is user, next turn has two possibilities: text reply; tool call
                    response, history = chat(self.tokenizer, self.model, turn['content'], history, 'user')
                    # Response is dict type, meaning it's a tool call turn, extract for calculating slot accuracy and recall
                    if isinstance(response, dict):
                        pred_slot = response['parameters']
                    if isinstance(response, str):
                        pred_reply = response.strip()
                if turn['role'] == 'assistant':
                    # Current turn is assistant with two possibilities: text reply; model thinking about tool call
                    if 'search_hotels' in turn['content']: # Skip model thinking about tool call
                        continue 
                    else: # It's a text reply, so calculate its BLEU score with the label text reply
                        label_reply = turn['content'].strip()
                        if pred_reply and label_reply:
                            score = self._bleu4(pred_reply, label_reply)
                            bleu_scores.append(score)
                            pred_reply, label_reply = "", ""
                if turn['role'] == 'tool':
                    # Current turn is tool, so calculate the accuracy and recall of model-predicted slots
                    label_slot = turn['parameters']
                    correct, pred_slots, true_slots = self._slot_accuracy(pred_slot, label_slot)
                    true_slot_count += true_slots
                    pred_slot_count += pred_slots
                    correct_slot_count += correct
                    pred_slot, label_slot = {}, {}
                    if 'observation' in turn:
                        # Current turn is observation, next turn has two possibilities: text reply; tool call
                        response, history = chat(self.tokenizer, self.model, json.dumps(turn['observation'], ensure_ascii=False), history, 'observation')
                        if isinstance(response, str):
                            pred_reply = response.strip()
                        if isinstance(response, dict):
                            pred_slot = response['parameters']
        
        score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
        score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
        score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
        score_dict["bleu-4"] = sum(bleu_scores)/len(bleu_scores)
        for k, v in score_dict.items():
            score_dict[k] = round(v * 100, 4)
        print(f"score dict: {score_dict}")

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, PeftArguments, DataTrainingArguments))
    model_args, peft_args, data_args = parser.parse_args_into_dataclasses()

    if model_args.checkpoint_path:
        if 'hotel_pt2' in model_args.checkpoint_path:
            tokenizer, model = load_pt2(model_args)
        elif 'hotel_lora' in model_args.checkpoint_path:
            tokenizer, model = load_lora(model_args, peft_args)
        else:
            print("checkpoint path error")
            exit()
    else:
        tokenizer, model = load_model(model_args)

    evaluator = Evaluator(tokenizer, model, data_args.test_file)
    evaluator.evaluate()

