import json
import ollama
import pandas as pd
from tqdm import tqdm  
import os
import os.path as osp

def load_dataset(file_path):
    datas=[]
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            datas.append({
                "ID": data["ID"],
                "question": data["question"],
                "reasoning_chains": data["reasoning_chains"],
            })
    return datas

def query_ollama(prompt, model='qwen2.5:7b-instruct', temperature=0.7):
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            'temperature': temperature,
        }
    )
    return response['response']

def process_dataset(data):
    kg_triple_set = []
    reasoning_paths_instances = data["reasoning_chains"]
    question = data["question"]
    for lines in reasoning_paths_instances:
        for l in lines:
            triple = [l[0] , l[1] , l[2]]
            if triple not in kg_triple_set:
                kg_triple_set.append(triple)
                
    prompt = open(
        "",
        'r', encoding='utf-8'
    ).read()
    
    kg_instances_str=""
    for l in kg_triple_set: 
        kg_instances_str += "(" + l[0] + ", " + l[1] + ", " + l[2] + " )\n"
       
    kg_instances_str = kg_instances_str.strip("\n")   
     
    prompt = prompt + "Q: " + question + "\nKnowledge Triplets: " + kg_instances_str + "\nA: "
    
    response=query_ollama(prompt)
    
    return response, prompt

def save_to_jsonl(data, file_path):
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    input_file = '' 
    output_file = ''  
    
    finished = []
    if osp.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                saved = json.loads(line.strip())
                finished.append(saved["ID"])
    
    print("Loading dataset...")
    datas = load_dataset(input_file)
    
    print("Processing dataset...")
    for data in datas:
        if data["ID"] in finished:
            continue
        processed_data, prompt = process_dataset(data)
        with open(output_file, 'a', encoding='utf-8') as f:
            output={
                "ID": data["ID"],
                "question": data["question"],
                "result": processed_data,
                "reasoning_chains": data["reasoning_chains"],
                "prompt": prompt
            }
            save_to_jsonl(output, output_file)