import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
from typing import Dict, List, Set, Tuple
from sentence_transformers import SentenceTransformer
import torch

file_path = ''

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
semantic_model.to(device)

def read_reasoning_chains(file_path: str) -> List[Dict]:
    instances = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'reasoning_chains' in data:
                instances.append(data)
    return instances

def extract_triples(reasoning_chains: List[List[List]]) -> Set[Tuple]:
    triples = set()
    if "Readi" in file_path or 'StructGPT' in file_path:
        for chain in reasoning_chains:
            for triple in chain:
                triples.add(tuple(triple))
    elif "ToG" in file_path:
        for triple in reasoning_chains:
            triples.add(tuple(triple))
    return triples

def create_graph(triples: Set[Tuple]) -> nx.Graph:
    G = nx.Graph()
    for head, relation, tail in triples:
        if G.has_edge(head, tail):
            current_relations = G[head][tail]['relations']
            if relation not in current_relations:
                current_relations.append(relation)
        else:
            G.add_edge(head, tail, relations=[relation])
    
    return G

def cal_sim(G: nx.Graph, question: str, id: str):
    question_embedding = semantic_model.encode(question, convert_to_tensor=True)

    result_dict={}
    result_dict['id']=id 
    result_dict['similarity']=[]

    edge_labels = {}
    for u, v, data in G.edges(data=True):
        relations = data['relations']
        rel_texts = []

        for rel in relations:
            rel_embedding = semantic_model.encode(rel, convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(question_embedding, rel_embedding, dim=0).item()
            rel_texts.append(f"{rel}\n{similarity:.2f}")
            f=1
            for item in result_dict['similarity']:
                if item['relation'] == rel:
                    f=0
            if f:
                result_dict['similarity'].append({"relation": rel, "score": similarity})

        edge_labels[(u, v)] = '\n'.join(rel_texts)

    
    with open('','a') as f:
        f.write(json.dumps(result_dict)+'\n')

def analyze_instance(instance: Dict, instance_id: str) -> Dict:
    triples = extract_triples(instance['reasoning_chains'])
    G = create_graph(triples)
    
    cal_sim(
        G,
        instance['question'],
        instance_id,
    )

def main():
    instances = read_reasoning_chains(file_path)
    print(f"\nTotal instances: {len(instances)}")

    for instance in instances:
        instance_id = instance['ID']
        print(f"\nAnalyzing instance {instance_id}...")
        analyze_instance(instance, instance_id)

if __name__ == "__main__":
    main()