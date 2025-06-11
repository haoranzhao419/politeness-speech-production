import csv
import os
import re
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import infer_auto_device_map, dispatch_model

login(token="REPLACE_WITH_YOURS", add_to_git_credential=True)

def load_huggingface_model(model_name, cache_directory="./cache_hf_llms"):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)

    # Configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load the model with device_map="auto" for optimal device allocation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache_directory,
        quantization_config=bnb_config
    )

    device_map = model.hf_device_map

    print(f"Successfully loaded model ({model_name}) with model parallelism and INT4 quantization")
    return model, tokenizer, device_map

def separate_responses(raw_response):
    response = None
    question_pos = raw_response.lower().find("question:")
    if question_pos != -1:
        answer_pos = raw_response.lower().find("answer:", question_pos)
        if answer_pos != -1:
            response = raw_response[answer_pos + len("answer:"):].strip()
    return response

def parse_response(response):
    for line in response.splitlines():
        line = line.strip()
        if "was terrible" in line or "1" in line:
            return "Answer: 1) It was terrible."
        elif "was bad" in line or "2" in line:
            return "Answer: 2) It was bad."
        elif "was good" in line or "3" in line:
            return "Answer: 3) It was good."
        elif "was amazing" in line or "4" in line:
            return "Answer: 4) It was amazing."
        elif "wasn't terrible" in line or "5" in line:
            return "Answer: 5) It wasn't terrible."
        elif "wasn't bad" in line or "6" in line:
            return "Answer: 6) It wasn't bad."
        elif "wasn't good" in line or "7" in line:
            return "Answer: 7) It wasn't good."
        elif "wasn't amazing" in line or "8" in line:
            return "Answer: 8) It wasn't amazing."
    return None

# Argument parser setup
parser = argparse.ArgumentParser()

# Model args
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=50, help='max tokens')
parser.add_argument('--prompt', type=str, default="experiment2_prompt1", help='prompt')
parser.add_argument('--num_completions', type=int, default=1, help='number of completions')
parser.add_argument('--do_sample', type=bool, default=True, help="do_sample")

# eval args
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')

# Data args
parser.add_argument('--no_items', type=int, default=5, help='number of scenarios to evaluate')
parser.add_argument('--file', type=str, default="scenarios_utterance_production_multi_choice_format", help='file name')

# Parse arguments
args = parser.parse_args()

# Read data
scenarios_utterance_production_multi_choice_format = pd.read_csv(f"../../data/{args.file}.csv", header=None)
scenarios_utterance_production_multi_choice_format = scenarios_utterance_production_multi_choice_format[0].tolist()
# scenarios_utterance_production_multi_choice_format = scenarios_utterance_production_multi_choice_format[:3]
print("The length of the dataset is:", len(scenarios_utterance_production_multi_choice_format))

# Load model and tokenizer
print(f"Loading Huggingface model and tokenizer ({args.model})")
model, tokenizer, device_map = load_huggingface_model(model_name=args.model)

# Print the device map for debugging
# print("Device Map:", device_map)

tokenizer.pad_token = tokenizer.eos_token

# Load the prompts
with open(f"../prompts/{args.prompt}.txt") as f:
    prompt = f.read()

# Generate responses
responses_list = []
for i in tqdm(range(len(scenarios_utterance_production_multi_choice_format))):
    query = scenarios_utterance_production_multi_choice_format[i]
    full_prompt = f"Instruction:\n{prompt}\n\n{query}\nAnswer:"
    print(f"Processing query {i+1}/{len(scenarios_utterance_production_multi_choice_format)}")
    
    response = []
    while len(response) < args.num_completions:
        first_device = next(iter(device_map.values()))
        input_ids = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(first_device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids["input_ids"],            
                    # **input_ids,
                    attention_mask=input_ids["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    return_legacy_cache=True,
                    max_new_tokens=args.max_tokens,
                    num_return_sequences=1,  # Generate one response at a time
                    return_dict_in_generate=True,
                    output_scores=True,
                    temperature=args.temperature,
                    do_sample = args.do_sample,
                    top_p=1.0
                )
            raw_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            if args.file == "scenarios_utterance_production_free_response_format":
                # processed_response = process_free_response(raw_response)
                print(f"Processed Free Response: {raw_response}")
                # print(f"query is {query}")
                response.append(query + "\n\n" + raw_response)
                # response.append(raw_response)
                # response.append(query)
            else:
                separated_responses = separate_responses(raw_response)
                if separated_responses:
                    parsed_response = parse_response(separated_responses)
                    if parsed_response:
                        response.append(parsed_response)
                        print(f"Parsed Response: {parsed_response}")
                    else:
                        print("Unparsable response detected, retrying...")
                else:
                    print("Unable to separate response, retrying...")
            
        except Exception as e:
            print(f"Error occurred during generation: {e}")
        
        finally:
            del input_ids
            torch.cuda.empty_cache()

    responses_list.append(response)

# Save responses
results_dir = '../../results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if "/" in args.model:
    args.model = args.model.replace("/", "_")

if args.file == "scenarios_utterance_production_free_response_format":
    print("1")
    file_path = f'../../results/utterance_production_{args.prompt}_results_{args.model}_{args.temperature}_{args.num_completions}_{args.file}.csv'
    flat_responses = [item for sublist in responses_list for item in sublist]
    print(flat_responses)
    df = pd.DataFrame(flat_responses, columns=['Generated Scenarios'])
    print(f"file path is {file_path}.")
    df.to_csv(file_path, index=False)
    print(f"Free responses saved to {file_path}")
else:
    txt_file_path = f'../../results/utterance_production_{args.prompt}_results_{args.model}_{args.temperature}_{args.num_completions}_{args.file}.txt'
    with open(txt_file_path, 'w') as f:
        for item in responses_list:
            f.write("%s\n" % item)
    print(f"Multi-choice responses saved to {txt_file_path}")