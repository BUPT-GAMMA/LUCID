import torch
from torch.nn import Module
from torch_geometric.nn import GINConv
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--use_edge_emb', type=bool, default=True)
parser.add_argument('--use_edge_weight', type=bool, default=True)
parser.add_argument('--use_sim_loss', type=bool, default=True)
args = parser.parse_args()


def build_graph_from_embeddings(entity_embeddings, relation_embeddings, entity_ids, relation_ids, reasoning_chains, sim_scores):
    edge_index = []
    edge_type = []
    edge_weights = []
    
    for chain in reasoning_chains:
        if chain["id"] not in sim_scores:
            continue
            
        scores = sim_scores[chain["id"]]
        
        for l in chain:
            head, relation, tail = l[0], l[1], l[2]
            
            if head in entity_ids and tail in entity_ids and relation in relation_ids:
                head_idx = entity_ids[head]
                tail_idx = entity_ids[tail]
                rel_id = relation_ids[relation]
                
                edge_index.append([head_idx, tail_idx])
                edge_type.append(rel_id)
                
                weight = scores.get(relation, 1.0)
                edge_weights.append(weight)
    
    if not edge_index:
        raise ValueError("No valid edges found in reasoning chains")
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    return Data(x=entity_embeddings, 
               edge_index=edge_index, 
               edge_type=edge_type,
               edge_attr=edge_weights)
   

def load_reasoning_chains(file_path):
    all_chains = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'reasoning_chains' in data:
                chains = data['reasoning_chains']
                all_chains.extend(chains)
    return all_chains

def load_reasoning_chains_and_embeddings(jsonl_path, embedding_path, sim_path):
    data_list = []
    embeddings_data = {}
    with open(embedding_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'id' in data and 'label' in data:
                if data['label'] == "":
                    continue
                embeddings_data[data['id']] = data
    
    sim_scores={}
    if sim_path:
        with open(sim_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                scores = {}
                for rel in data.get("similarity", []):
                    scores[rel["relation"]] = rel["score"]
                sim_scores[data["id"]] = scores
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'reasoning_chains' in data and 'ID' in data :
                data_id = data['ID']
                if sim_scores:
                    if data_id in embeddings_data and data_id in sim_scores:
                        data_list.append({
                            'chains': data['reasoning_chains'],
                            'embeddings': embeddings_data[data_id],
                            'similarity': sim_scores[data_id],
                        })
                else:
                    if data_id in embeddings_data:
                        data_list.append({
                            'chains': data['reasoning_chains'],
                            'embeddings': embeddings_data[data_id]
                        })
    return data_list

def build_graph_from_single_data(entity_embeddings, relation_embeddings, entity_ids, relation_ids, reasoning_chains, relation_sims):
    edge_index = []
    edge_type = []
    node_embeddings = []  
    used_entities = set()  
    
    kg_triple_set = []
    entity_set = set()
    for chain in reasoning_chains:
        for l in chain:
            triple = [l[0], l[1], l[2]]
            if triple not in kg_triple_set:
                kg_triple_set.append(triple)
                if l[0] in entity_ids:
                    entity_set.add(l[0])
                if l[2] in entity_ids:
                    entity_set.add(l[2])

    new_entity_ids = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
    
    for entity in sorted(entity_set):
        if entity in entity_ids:
            original_idx = entity_ids[entity]
            node_embeddings.append(entity_embeddings[original_idx])
            used_entities.add(entity)

    edge_embeddings = []  
    edge_sim=[]
    node_to_edges = {idx: [] for idx in range(len(node_embeddings))} 
    
    for triple in kg_triple_set:
        head, relation, tail = triple
        if head in new_entity_ids and tail in new_entity_ids and relation in relation_ids:
            head_idx = new_entity_ids[head]
            tail_idx = new_entity_ids[tail]
            edge_index.append([head_idx, tail_idx])
            rel_idx = relation_ids[relation]
            edge_type.append(rel_idx)
            
            edge_embeddings.append(relation_embeddings[rel_idx])
            edge_sim.append(relation_sims[rel_idx])
            
            edge_idx = len(edge_embeddings) - 1
            node_to_edges[head_idx].append(edge_idx)
            node_to_edges[tail_idx].append(edge_idx)
    
    if not edge_index or not node_embeddings:
        return None
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    node_embeddings = torch.stack(node_embeddings)
    edge_embeddings = torch.stack(edge_embeddings) if edge_embeddings else torch.zeros((0, node_embeddings.size(1), node_embeddings.size(2)))
    
    avg_edge_embeddings = []
    for node_idx in range(len(node_embeddings)):
        if node_to_edges[node_idx]:

            connected_edge_embeddings = edge_embeddings[node_to_edges[node_idx]]

            avg_edge_emb = torch.mean(connected_edge_embeddings, dim=0)
        else:
            avg_edge_emb = torch.zeros_like(edge_embeddings[0]) if len(edge_embeddings) > 0 else torch.zeros_like(node_embeddings[0])
        avg_edge_embeddings.append(avg_edge_emb)
    

    avg_edge_embeddings = torch.stack(avg_edge_embeddings)

    node_embeddings_flat = node_embeddings.view(node_embeddings.size(0), -1)
    avg_edge_embeddings_flat = avg_edge_embeddings.view(avg_edge_embeddings.size(0), -1)
    
    if args.use_edge_emb:
        combined_embeddings = torch.cat([node_embeddings_flat, avg_edge_embeddings_flat], dim=1)
    else:
        combined_embeddings = node_embeddings_flat
    
    return Data(x=combined_embeddings, 
                edge_index=edge_index, 
                edge_type=edge_type,
                edge_sim=edge_sim)

def process_single_data(embeddings_data, tag, sim_scores=None):
    entity_embeddings = []
    relation_embeddings = []
    relation_sims = []  
    entity_ids = {}
    relation_ids = {}
    current_entity_id = 0
    current_relation_id = 0
    
    for entity in embeddings_data['entity_content']:
        entity_ids[entity['content']] = current_entity_id
        entity_embeddings.append(torch.tensor(entity['matrix']))
        current_entity_id += 1
    
    for relation in embeddings_data['relation_content']:
        relation_content = relation['content']
        relation_ids[relation_content] = current_relation_id
        relation_emb = torch.tensor(relation['matrix'])
        relation_embeddings.append(relation_emb)
        
        if sim_scores and relation_content in sim_scores:
            weight = sim_scores[relation_content]
            relation_sims.append(weight)
        
        current_relation_id += 1
    
    entity_embeddings = torch.stack(entity_embeddings) if entity_embeddings else None
    relation_embeddings = torch.stack(relation_embeddings) if relation_embeddings else None
    
    label = 1.0 if tag == "right" else 0.0
    label_tensor = torch.tensor(label)
    
    return entity_embeddings, relation_embeddings, entity_ids, relation_ids, label_tensor, relation_sims

def custom_collate(batch):
    return batch

class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]

def load_similarity_scores(file_path):
    sim_scores = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            scores = {}
            for rel in data.get("similarity", []):
                scores[rel["relation"]] = rel["score"]
            sim_scores[data["id"]] = scores
    return sim_scores

def train_gin(jsonl_path):
    batch_size = 32
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    
    embedding_path = ''

    sim_file_path = ''
   
    data_list = load_reasoning_chains_and_embeddings(jsonl_path, embedding_path,sim_file_path)
    
    train_data=data_list
    
    train_dataset = GraphDataset(train_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=custom_collate
    )

        
    for batch_data in train_loader:
        batch_graphs = []
        batch_labels = []
        batch_sim_scores = []
        
        for data_item in batch_data:
            sim_scores = data_item.get('similarity', None)
            entity_embeddings, relation_embeddings, entity_ids, relation_ids, label, relation_sims = process_single_data(
                data_item['embeddings'],
                data_item['embeddings']['label'],
                sim_scores
            )
            
            if entity_embeddings is None or relation_embeddings is None:
                continue
                
            graph_data = build_graph_from_single_data(
                entity_embeddings,
                relation_embeddings,
                entity_ids,
                relation_ids,
                data_item['chains'],
                relation_sims
            )
            
            
            if graph_data is not None:
                batch_graphs.append(graph_data)
                batch_labels.append(label)
                
            
                with open('', 'a') as f:
                    dict={}
                    dict['id']=data_item['embeddings']['id']
                    dict['similarity_mean']=np.mean(graph_data.edge_sim)
                    dict['similarity_std']=np.std(graph_data.edge_sim)
                    f.write(json.dumps(dict) + '\n')

        if not batch_graphs:
            continue
            
        batched_graphs = Batch.from_data_list(batch_graphs).to(device)
        batch_labels = torch.stack(batch_labels).to(device)


if __name__ == '__main__':
    jsonl_path = ''
    train_gin(jsonl_path)
