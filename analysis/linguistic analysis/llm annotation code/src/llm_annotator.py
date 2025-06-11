import csv
import os
import re
import argparse
import pandas as pd

from tqdm import tqdm
from anthropic_credit import ANTHROPIC_API_KEY
from openai_credit import OPENAI_API_KEY

# from langchain import HuggingFacePipeline
# from langchain.llms import Ollama
# from langchain.chat_models import ChatOpenAI, ChatOllama

# from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langchain.schema import (
    HumanMessage,
    SystemMessage
)


# def parse_response(response):

#     if "was terrible" in response or "1" in response:
#         parsed_response = "Answer: 1) It was terrible."
#     elif "was bad" in response or "2" in response:
#         parsed_response = "Answer: 2) It was bad."
#     elif "was good" in response or "3" in response:
#         parsed_response = "Answer: 3) It was good."
#     elif "was amazing" in response or "4" in response:
#         parsed_response = "Answer: 4) It was amazing."
#     elif "wasn't terrible" in response or "5" in response:
#         parsed_response = "Answer: 5) It wasn't terrible."
#     elif "wasn't bad" in response or "6" in response:
#         parsed_response = "Answer: 6) It wasn't bad."
#     elif "wasn't good" in response or "7" in response:
#         parsed_response = "Answer: 7) It wasn't good."
#     elif "wasn't amazing" in response or "8" in response:
#         parsed_response = "Answer: 8) It wasn't amazing."
#     else:
#         print(f"Response {response} not found.")
#         parsed_response = str(input("Enter response:"))
#     return parsed_response

parser = argparse.ArgumentParser()

# model args
parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=20, help='max tokens')
parser.add_argument('--prompt', type=str, default="0shot_multi_choice", help='prompt')
parser.add_argument('--num_completions', type=int, default=1, help='number of completions')

# eval args
# parser.add_argument('--num', '-n', type=int, default=300, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')

# data args (I need to set up the input and output directory of my data)
parser.add_argument('--no_items', type=int, default= "5", help='number of scenarios to evaluate')
parser.add_argument('--file', type=str, default= "final_human_llm_all_responses", help='file name')
# parser.add_argument('--output_dir', type=str, default='./llm_analogy/results/', help='output directory')


# parse args
args = parser.parse_args()


# read data (data should just be a list)
scenarios_utterance_production = pd.read_csv(f"../data/{args.file}.csv", header=None)

# to list
scenarios_utterance_production = scenarios_utterance_production[0].tolist()


# scenarios_utterance_production = scenarios_utterance_production[:args.no_items]
    
# initialize LLM (Don't need to change)
if args.model in ["gpt-4.1-2025-04-14", "gpt-4-0613", "gpt-4-0125-preview", "gpt-4o-2024-11-20"]:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,)
elif args.model in ["claude-3-opus-20240229", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"]:
    llm = ChatAnthropic(anthropic_api_key=ANTHROPIC_API_KEY,
                        model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,)
# elif args.model in ["mistral", "llama3", "gemma:7b", "phi3", "llama3:text", "llama3.1:8b-text-q4_0"]:
#     llm = ChatOllama(model=args.model,
#              temperature=args.temperature)
else:
    raise ValueError(f"Model {args.model} not found.")
    

# load the prompts
with open(f"../prompts/{args.prompt}.txt") as f:
    prompt = f.read()

assistant_query_version1 =f"""
    Please read the scenario carefully first, then identify and annotate the politeness marker(s) used in the response. 
    For annotation, highlight the specific word(s) or phrase(s) in the response that function as politeness markers, rather than copying the entire response.  
    Then, using the categorization method from Brown and Levinson's Politeness: Some Universals in Language Use (1987), label each politeness marker with its category (Positive politeness, Negative politeness, Off record) and specific politeness strategy.
    
    Please present your answer in the following format for each politeness marker WITHOUT ANY additional text or explanation:

    Politeness marker: [the specific word or phrase from the response]
    Politeness strategy:

    1. [category-1 + specific politeness strategy-1]
    2. [category-2 + specific politeness strategy-2] (if applicable)
    3. ... (if applicable)

    (Repeat the above for each politeness marker found in the response.)
   """


assistant_query_version2 =f"""
    Please read the scenario carefully first, then identify and annotate the politeness marker(s) used in the response. 
    For annotation, highlight the specific word(s) or phrase(s) in the response that function as politeness markers, rather than copying the entire response.  
    Then, using the comprehensive politeness strategy list provided in the system prompt, label each politeness marker with its category (Positive politeness, Negative politeness, Off record) and specific politeness strategy.
    
    Please present your answer in the following format for each politeness marker WITHOUT ANY additional text or explanation:

    Politeness marker: [the specific word or phrase from the response]
    Politeness strategy:

    1. [category-1 + specific politeness strategy-1]
    2. [category-2 + specific politeness strategy-2] (if applicable)
    3. ... (if applicable)


    (Repeat the above for each politeness marker found in the response. Maximum of 3 politeness markers)
   """

# generate LLMs' responses
responses_list= []
for i in tqdm(range(len(scenarios_utterance_production))):
    query = scenarios_utterance_production[i]

    response = []
    for _ in range(args.num_completions):
        message = [SystemMessage(content=prompt), HumanMessage(content=query)]
        # print(message)
        raw_response = llm.generate([message]).generations[0]
        print(raw_response[0].text)
        # if args.file == "scenarios_utterance_production_free_response_format":
        #     print(raw_response[0].text)
        #     response.append(query + "\n" + raw_response[0].text)
        # else:
        #     parsed_response = parse_response(raw_response[0].text)
        #     response.append(parsed_response)
        #     print(parsed_response)

    responses_list.append(raw_response[0].text)

# print(responses_list)
with open(f'../results/{args.model}_annotation_politeness_marker_{args.prompt}_results_{args.temperature}_{args.num_completions}_{args.file}_{len(responses_list)}.txt', 'w') as f:
    # write the responses_list directly to the txt
    f.write("%s\n" % responses_list)

    # for item in responses_list:
    #     # print("hello")
    #     f.write("%s\n" % item)



with open(f'../results/{args.model}_annotation_politeness_marker_{args.prompt}_results_{args.temperature}_{args.num_completions}_{args.file}_{len(responses_list)}.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    for item in responses_list:
        writer.writerow([item])  # Each item in its own row

# def flatten(lst):
#     flat_list = []
#     for item in lst:
#         if isinstance(item, list):
#             flat_list.extend(flatten(item))
#         else:
#             flat_list.append(item)
#     return flat_list

# # save response to a txt file
# if args.file == "scenarios_utterance_production_free_response_format":
#     # save the generated list into a csv file

#     file_path = f'../results/utterance_production_{args.prompt}_results_{args.model}_{args.temperature}_{args.num_completions}.csv'
#     df = pd.DataFrame(flatten(responses_list), columns=['generated-complete scenarios'])
#     df.to_csv(file_path, index=False)

# else:
#     with open(f'../results/{args.model}_annotation_politeness_marker_{args.prompt}_results_{args.temperature}_{args.num_completions}_{args.file}.txt', 'w') as f:
#         for item in responses_list:
#             # print("hello")
#             f.write("%s\n" % item)