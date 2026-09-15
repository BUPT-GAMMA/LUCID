import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
import json
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from scipy.stats import entropy
import time
import gc


parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument('--model_name', type=str, help='qwen2.5-7b/llama3-8b', default='qwen2.5-7b')
parser.add_argument('--entropy', type=str, default=0)
args = parser.parse_args()


# bge_model = SentenceTransformer('./bge-base-en-v1.5/').to("cuda:0")

input_path = ""
response_dict = {}
source_info_dict = {}
question_dict = {}
prompt_dict = {}
with open(input_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        response_dict[data["ID"]] = data["results"]
        source_info_dict[data["ID"]] = data["reasoning_chains"]
        question_dict[data["ID"]] = data["question"]
        prompt_dict[data["ID"]] = data["prompt"]

if args.model_name == "qwen2.5-7b":
    model_name = ""
elif args.model_name == "llama3-8b":
    model_name = ""
else:
    print("name error")
    exit(-1)

model = AutoModelForCausalLM.from_pretrained(
    f"{model_name}",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
device = "cuda"

tokenizer_for_temp = tokenizer

def print_gpu_mem(tag):
    for dev in [0,1]:
        alloc = torch.cuda.memory_allocated(dev) / 1024 / 1024

def add_special_template(prompt):
    messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
    text = tokenizer_for_temp.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text

def has_overlap(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(1, min_len + 1):
        if s1[-i:] == s2[:i]: 
            return True
    return False

def calculate_prompt_spans(raw_prompt_spans,input_ids, prompt, tokenizer,reasoning_paths_instances):
    entity_spans = []
    relation_spans = []
    start,end=raw_prompt_spans[0]
    start_text = prompt[:start]
    added_start_text = add_special_template(start_text)
    start_text_id = tokenizer(added_start_text, return_tensors="pt").input_ids.shape[-1] - 5
    if prompt[start] not in tokenizer.decode(input_ids[:,start_text_id]):
        if prompt[start] in tokenizer.decode(input_ids[:,start_text_id-1]):
            start_text_id-=1
        elif prompt[start] in tokenizer.decode(input_ids[:,start_text_id+1]):
            start_text_id+=1
     
    i=0   
    kg_triple_set=[]
    if 'Readi' in input_path or 'Readi' in input_path:
        for lines in reasoning_paths_instances:
            for l in lines:
                triple = [l[0] , l[1] , l[2]]
                if triple not in kg_triple_set:
                    kg_triple_set.append(triple)
    elif 'ToG' in input_path or 'RoG' in input_path:
        for l in reasoning_paths_instances:
            triple = [l[0] , l[1] , l[2]]
            if triple not in kg_triple_set:
                kg_triple_set.append(triple)
            
    for l in kg_triple_set:
        for _ in range(len(l)):
            s=""
            start=start_text_id
            while start_text_id+i<input_ids.shape[-1]:
                s+=tokenizer.decode(input_ids[:,start_text_id+i])
                if l[_] in s:
                    if _==0 or _==2:
                        entity_spans.append({(start,start_text_id+i):l[_]})
                    else:
                        relation_spans.append({(start,start_text_id+i):l[_]})
                    start_text_id+=i
                    i=0
                    break
                if not has_overlap(s,l[_]):
                    start=start_text_id+i+1
                    s=""
                i+=1
    
    return entity_spans,relation_spans, kg_triple_set
                    
def calculate_respond_spans(text, input_ids,response_rag, tokenizer,type="answer"):
    if type=="response":
        respond_spans = []
        start_text = text
        end_text = text+response_rag
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]-1
        respond_spans.append([start_id, end_id])
        return respond_spans
    elif type=="answer":
        respond_spans = []
        answer=response_rag.split("{")[-1].split("}")[0]
        s=response_rag.split("{")[0]+"{"
        e=response_rag.split("}")[0]
        start_text = text+s
        end_text = text+e
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]-1
        
        s=""
        while True:
            s+=tokenizer.decode(input_ids[:,start_id])
            if s.endswith("{"):
                start_id+=1
                s=""
            elif not has_overlap(s,answer):
                if answer in s:
                    break
                start_id+=1
                s=""
            else:
                break
            
        end_id = start_id
        s=""
        while True:
            s+=tokenizer.decode(input_ids[:,end_id])
            if answer not in s:
                if len(s)>=len(answer) and answer[0]==s[0] and answer[-1]==s[-1]:
                    break
                if s.endswith("}") or s.endswith('}.'):
                    end_id-=1
                    break
                end_id+=1
            else:
                break
            
        respond_spans.append([start_id, end_id])
        return respond_spans
        

def visualize_attention_from_scores(layer_head_span, save_path):
    num_heads = len(set(head_id for (layer_id, head_id) in layer_head_span.keys()))
    num_spans = len(layer_head_span[list(layer_head_span.keys())[0]])
    

    attention_matrix = np.zeros((num_heads, num_spans))
    
    for (layer_id, head_id), span_scores in layer_head_span.items():
        for span_idx, (span, score) in enumerate(span_scores):
            attention_matrix[head_id, span_idx] = score
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_matrix, 
                cmap='YlOrRd',
                xticklabels=[f'Triple {i+1}' for i in range(num_spans)],
                yticklabels=[f'Head {i+1}' for i in range(num_heads)],
                cbar_kws={'label': 'Attention Score'})
    
    plt.title('Average Attention Scores across Different Heads and Knowledge Triples')
    plt.xlabel('Knowledge Triples')
    plt.ylabel('Attention Heads')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def calculate_normalized_entropy(attention_scores):

    attention_probs = np.array([score for _, score in attention_scores])
    attention_probs = attention_probs / np.sum(attention_probs)

    ent = entropy(attention_probs)

    n = len(attention_probs)
    max_ent = np.log(n)

    normalized_ent = ent / max_ent if max_ent != 0 else 0
    return normalized_ent

def prepare_attention_scores(layer_head_score):
    formatted_scores = []
    for (layer_id, head_id), span_scores in layer_head_score.items():
        head_data = {
            "head_id": head_id,
            "layer_id": layer_id,
            "span_scores":  span_scores
        }
        formatted_scores.append(head_data)
    return formatted_scores


all_entropy_values = {
    'head_wise': [],  
    'sample_wise': []  
}

all_attention_scores = []

def convert_to_fp32(model):
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(torch.float32)
    return model

model = convert_to_fp32(model)
output_path = ''
saved=[]
# if os.path.exists(output_path):
#     with open(output_path,'r') as f:
#         for line in f:
#             data = json.loads(line.strip()) 
#             if data:
#                 saved.append(data['id'])

saveid_path=''
if os.path.exists(saveid_path):
    with open(saveid_path,'r') as f:
        for line in f:
            saved.append(line.strip('\n'))

for i in tqdm(range(len(response_dict))):
    
    sample_start_time = time.time()

    source_id = list(response_dict.keys())[i]
    
    try:
        source_id = list(response_dict.keys())[i]
        if str(source_id) in saved:
            print(source_id)
            continue
        
        response_rag = response_dict[source_id]
        temperature = 0.7
        reasoning_paths_instances =  source_info_dict[source_id]
        question= question_dict[source_id]
        
        if len(reasoning_paths_instances)==0:
            continue
        
        prompt = prompt_dict[source_id]
        prompt_spans=[]
        
        triple_line=prompt.split("Knowledge Triplets: ")[-1]
        start_len=len(prompt)-len(triple_line)
        triple_line=triple_line.split("A:")[0]
        end_len=start_len+len(triple_line)-1
        prompt_spans.append([start_len, end_len])
        
        if len(prompt)>10000:
            print('skip')
            continue
        text = add_special_template(prompt)
        input_text = text+response_rag
        print("all_text_len:", len(input_text))
        print("prompt_len", len(prompt))
        print("respond_len", len(response_rag))

        input_ids = tokenizer([input_text], return_tensors="pt").input_ids
        prefix_ids = tokenizer([text], return_tensors="pt").input_ids
        continue_ids = input_ids[0, prefix_ids.shape[-1]:] 

        entity_spans_ids, relation_spans_ids, kg_triple_set = calculate_prompt_spans(prompt_spans,input_ids, prompt, tokenizer,reasoning_paths_instances)
        triple_num = len(kg_triple_set)

        answer=response_rag.split("{")[-1].split("}")[0]
        if answer==response_rag or answer=="":
            print('skip')
            continue
        respond_spans_ids = calculate_respond_spans(text, input_ids, response_rag, tokenizer,"answer")
        if args.model_name == "qwen2.5-7b":
            start = 0 
            number = 28
        elif args.model_name == "llama3-8b":
            start = 0 
            number = 16
        else:
            print("model name error")

        start_p, end_p = None, None
        with torch.no_grad():
            outputs = model(
                    input_ids=input_ids, 
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=True,
                    # knowledge_layers=list(range(start, number))
                )
        # logits_dict = {key: [value[0].to(device), value[1].to(device)] for key, value in logits_dict.items()}

        # skip tokens without hallucination
        hidden_states = outputs["hidden_states"] # tuple ([batch, seq_len, vocab_size], ..., ) 
        last_hidden_states = hidden_states[-1][0, :, :] # [prefix_len, hidden_size]
        
        span_score_dict = []
        layer_head_score = {}
        layer_score={}
        
        sample_entropies = [] 
        
        for attentions_layer_id in range(len(outputs.attentions)):
            for head_id in range(outputs.attentions[attentions_layer_id].shape[1]):
                layer_head = (attentions_layer_id, head_id)
                p_span_score_dict = []
                for p_span in entity_spans_ids:
                    attention_score = outputs.attentions[attentions_layer_id][0,head_id,:,:]
 
                    p_span_score_dict.append({"attention_score": torch.mean(attention_score[respond_spans_ids[0][0]:respond_spans_ids[0][1]+1, eval(str(list(p_span.keys())[0]))[0]:eval(str(list(p_span.keys())[0]))[1]+1]).cpu().item(),"span":list(p_span.keys())[0],"content":list(p_span.values())[0],"type":"entity"})
                for p_span in relation_spans_ids:
                    attention_score = outputs.attentions[attentions_layer_id][0,head_id,:,:]
      
                    p_span_score_dict.append({"attention_score": torch.mean(attention_score[respond_spans_ids[0][0]:respond_spans_ids[0][1]+1, eval(str(list(p_span.keys())[0]))[0]:eval(str(list(p_span.keys())[0]))[1]+1]).cpu().item(),"span":list(p_span.keys())[0],"content":list(p_span.values())[0],"type":"relation"})
                layer_head_score[layer_head] = {"type":"relation","content": p_span_score_dict}
            
                if args.entropy==1:
                    head_entropy = calculate_normalized_entropy(p_span_score_dict)
                    sample_entropies.append(head_entropy)
                    all_entropy_values['head_wise'].append({
                        'sample_id': source_id,
                        'head_id': head_id,
                        'entropy': head_entropy,
                        'num_triples': len(p_span_score_dict)  
                    })
            
        if args.entropy==1:
            sample_mean_entropy = np.mean(sample_entropies)
            all_entropy_values['sample_wise'].append({
                'sample_id': source_id,
                'mean_entropy': sample_mean_entropy
            })

        attention_scores = []
        num_heads = outputs.attentions[attentions_layer_id].shape[1]
            
        sample_attention_scores = {
            "id": source_id,
            "attention_scores": prepare_attention_scores(layer_head_score)
        }
        
        with open(saveid_path,'a') as f:
            f.write(str(source_id)+'\n')
                
        cost_time = time.time() - sample_start_time
        gc.collect()
        torch.cuda.empty_cache()

        
    except Exception as e:
        print("----------------error----------------")
        print(e)


