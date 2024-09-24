import sys
import json
from prompt_helper import build_prompt, build_response

class Preprocessor:
    
    def __init__(self,data_args,tokenizer):
        self.prompt_column = data_args.prompt_column
        self.response_column = data_args.response_column
        self.max_source_length = data_args.max_source_length
        self.max_target_length = data_args.max_target_length
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss
    
    # Process test (dev/test) data
    '''
        Test data concatenation method: [pad][pad]...[bos_token]input text[pad][pad]....output text
    '''
    def preprocess_function_eval(self,examples):  
        inputs, targets = [], []

        # Read text from input/output (prompt/response) fields
        inputs, targets = [], []
        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                context = examples[self.prompt_column][i]
                prompt = build_prompt(context)
                response = build_response(examples[self.response_column][i])
                inputs.append(prompt)
                targets.append(response)

        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'left'

        # Tokenize input text (prompt)
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_source_length, 
            truncation=True, 
            padding=True
        )

        self.tokenizer.padding_side = 'right'

        # Tokenize output text (response)
        labels = self.tokenizer(
            text_target=targets, 
            max_length=self.max_target_length, 
            truncation=True, 
            padding=True,
            add_special_tokens=False
        )

        # If not calculating loss for pad tokens, mark pad tokens as -100 (model's agreed value)
        if self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    # Process training data
    '''
        Training data concatenation method: [bos_token]input text output text[eos_token][pad][pad]....
    '''
    def preprocess_function_train(self,examples):
        max_seq_length = self.max_source_length + self.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                context, response = examples[self.prompt_column][i], examples[self.response_column][i]
                prompt = build_prompt(context)
                response = build_response(response)

                #prompt = self.tokenizer.build_prompt(query)
                a_ids = self.tokenizer.encode(
                    text=prompt, 
                    add_special_tokens=False, 
                    truncation=True,
                    max_length=self.max_source_length-1
                )
                b_ids = self.tokenizer.encode(
                    text=response, 
                    add_special_tokens=False, 
                    truncation=True,
                    max_length=self.max_target_length-1
                )

			
                context_length = len(a_ids) + 1

                # Manual concatenation
                input_ids = [self.tokenizer.bos_token_id] + a_ids + b_ids + [self.tokenizer.eos_token_id]
                
                # Manual padding
                labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
                
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [self.tokenizer.pad_token_id] * pad_len

                # If not calculating loss for pad tokens, mark pad tokens as -100 (model's agreed value)
                if self.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]	

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)


        return model_inputs
