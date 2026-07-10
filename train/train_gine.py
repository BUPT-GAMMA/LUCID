import os
import random
import torch
from torch.nn import Module
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_edge_weight', type=bool, default=True)
parser.add_argument('--margin', type=float, default=0.3, help='margin for loss computation')
args = parser.parse_args()

class GINE(Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        self.convs.append(GINEConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, 256)
            ),
            edge_dim=edge_dim  
        ))
        
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(
                nn=torch.nn.Sequential(
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64)
                ),
                edge_dim=edge_dim  
            ))
            
        self.classifier = torch.nn.Linear(64, 1)
        
    def forward(self, x, edge_index, edge_attr, edge_sim, batch):
        if len(x.shape) == 3:
            x = x.view(x.size(0), -1)
        if len(edge_attr.shape) == 3:
            edge_attr = edge_attr.view(edge_attr.size(0), -1)    
        if args.use_edge_weight:
            edge_attr = edge_attr * edge_sim.unsqueeze(-1)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        
        graph_embeddings = global_add_pool(x, batch)  
        logits = self.classifier(graph_embeddings).squeeze(-1)
        return torch.sigmoid(logits)
        
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

def compute_loss(pred, target, margin=0.2):
    return torch.nn.BCELoss()(pred, target.float())

def load_reasoning_chains(file_path):
    all_chains = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'reasoning_chains' in data:
                chains = data['reasoning_chains']
                all_chains.extend(chains)
    return all_chains

def load_reasoning_chains_and_embeddings(jsonl_path, embedding_path, sim_path, label_path):
    label_dict={}
    with open(label_path,'r')as f:
        for line in f:
            data=json.loads(line)
            label_dict[str(data['ID'])]=data['label']

    data_list = []
    embeddings_data = {}
    with open(embedding_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'id' in data:
                if str(data['id']) not in label_dict:
                    continue
                embeddings_data[str(data['id'])] = data
                embeddings_data[str(data['id'])]['label']=label_dict[str(data['id'])]
    
    sim_scores={}
    if sim_path:
        with open(sim_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                scores = {}
                for rel in data.get("similarity", []):
                    scores[rel["relation"]] = rel["score"]
                sim_scores[str(data["id"])] = scores
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'reasoning_chains' in data and 'ID' in data :
                data_id = str(data['ID'])
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

def build_graph_from_single_data(entity_embeddings, relation_embeddings, entity_ids, relation_ids, reasoning_chains, sim_scores=None):
    edge_index = []
    edge_type = []
    node_embeddings = []
    edge_sim_scores = [] 
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
    
    # for l in reasoning_chains:
    #     triple = [l[0], l[1], l[2]]
    #     if triple not in kg_triple_set:
    #         kg_triple_set.append(triple)
    #         if l[0] in entity_ids:
    #             entity_set.add(l[0])
    #         if l[2] in entity_ids:
    #             entity_set.add(l[2])

    new_entity_ids = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
    
    for entity in sorted(entity_set):
        if entity in entity_ids:
            original_idx = entity_ids[entity]
            node_embeddings.append(entity_embeddings[original_idx])
            used_entities.add(entity)

    edge_features = []
    
    for triple in kg_triple_set:
        head, relation, tail = triple
        if head in new_entity_ids and tail in new_entity_ids and '('+head+', '+relation+', '+tail+')' in relation_ids:
            head_idx = new_entity_ids[head]
            tail_idx = new_entity_ids[tail]
            edge_index.append([head_idx, tail_idx])
            rel_idx = relation_ids['('+head+', '+relation+', '+tail+')']
            
            edge_features.append(relation_embeddings[rel_idx])
            
            if sim_scores and relation in sim_scores:
                edge_sim_scores.append(sim_scores[relation])
            else:
                edge_sim_scores.append(1.0)
    
    if not edge_index or not node_embeddings:
        return None
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_embeddings = torch.stack(node_embeddings)
    edge_features = torch.stack(edge_features) if edge_features else torch.zeros((0, relation_embeddings.size(1)))
    edge_sim_scores = torch.tensor(edge_sim_scores, dtype=torch.float)
    
    node_embeddings_flat = node_embeddings.view(node_embeddings.size(0), -1)
    edge_features_flat = edge_features.view(edge_features.size(0), -1)
    
    return Data(x=node_embeddings_flat,
               edge_index=edge_index,
               edge_attr=edge_features_flat,
               edge_sim=edge_sim_scores) 

def process_single_data(embeddings_data, tag, sim_scores=None):
    entity_embeddings = []
    relation_embeddings = []
    entity_ids = {}
    relation_ids = {}
    current_entity_id = 0
    current_relation_id = 0
    
    for entity in embeddings_data['entity_content']:
        entity_ids[entity['content']] = current_entity_id
        entity_embeddings.append(torch.tensor(entity['matrix']))
        current_entity_id += 1
        
    for relation in embeddings_data['relation_content']:
        relation_content = '('+relation['content'][0]+', '+relation['content'][1]+', '+relation['content'][2]+')'
        relation_ids[relation_content] = current_relation_id
        relation_emb = torch.tensor(relation['matrix'])
        relation_embeddings.append(relation_emb)
        current_relation_id += 1
    
    entity_embeddings = torch.stack(entity_embeddings) if entity_embeddings else None
    relation_embeddings = torch.stack(relation_embeddings) if relation_embeddings else None
    
    if tag == 1 or tag == 2:
        label=1
    else:
        label=0
    
    label_tensor = torch.tensor(label)
    
    return entity_embeddings, relation_embeddings, entity_ids, relation_ids, label_tensor

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

def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt  

def train_gin(jsonl_path):
    learning_rate = 1e-4
    epochs = 300
    hidden_channels = 512
    batch_size = 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    embedding_path = ''
    label_path = ''
    
    if args.use_edge_weight:
        sim_file_path = ''
    else:
        sim_file_path = None
    data_list = load_reasoning_chains_and_embeddings(jsonl_path, embedding_path, sim_file_path, label_path)
    
    random.shuffle(data_list)
    
    train_dataset = GraphDataset(data_list)
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=custom_collate
    )
    
    sample_data = data_list[0]
    sim_scores = sample_data.get('similarity', None)
    entity_embeddings, relation_embeddings, entity_ids, relation_ids, label = process_single_data(
        sample_data['embeddings'],
        sample_data['embeddings']['label'],
        sim_scores
    )

    edge_dim = relation_embeddings.size(1) * relation_embeddings.size(2)

    in_channels = entity_embeddings.size(1) * entity_embeddings.size(2)
    
    model = GINE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        edge_dim=edge_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    best_f1 = 0.0
    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        all_probs = []
        all_labels = []
        
        for batch_data in train_loader:
            batch_graphs = []
            batch_labels = []

            for data_item in batch_data:
                sim_scores = data_item.get('similarity', None)
                entity_embeddings, relation_embeddings, entity_ids, relation_ids, label = process_single_data(
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
                    sim_scores
                )
                
                if graph_data is not None:
                    batch_graphs.append(graph_data)
                    batch_labels.append(label)

            if not batch_graphs:
                continue

            batched_graphs = Batch.from_data_list(batch_graphs).to(device)
            batch_labels = torch.stack(batch_labels).to(device)
            
            optimizer.zero_grad()

            pred = model(
                x=batched_graphs.x,
                edge_index=batched_graphs.edge_index,
                edge_attr=batched_graphs.edge_attr,
                edge_sim=batched_graphs.edge_sim,
                batch=batched_graphs.batch
            )

            loss = compute_loss(pred, batch_labels, args.margin)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1

            all_probs.extend(pred.detach().cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = get_threshold(thresholds, tpr, fpr)

        predicted = (np.array(all_probs) > optimal_threshold).astype(float)
        accuracy = np.mean(predicted == all_labels)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Train Accuracy: {accuracy:.4f}, Train AUC: {auc:.4f}")

        scheduler.step(avg_epoch_loss)

        if epoch%10==0:
            save_dir = ""
            os.makedirs(save_dir, exist_ok=True)
            if args.use_edge_weight:
                save_path = os.path.join(save_dir, f'best_gine_model_sim_epoch{epoch+1}.pt')
            else:
                save_path = os.path.join(save_dir, f'best_gine_model_nosim_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': best_acc,
            }, save_path)

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_state = model.state_dict()

            save_dir = ""
            os.makedirs(save_dir, exist_ok=True)
            if args.use_edge_weight:
                save_path = os.path.join(save_dir, f'best_gine_model_sim_epoch{epoch+1}.pt')
            else:
                save_path = os.path.join(save_dir, f'best_gine_model_nosim_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': best_acc,
            }, save_path)
            
            print(f"***New Best Model - Train Loss: {avg_epoch_loss:.4f}, Train Accuracy: {accuracy:.4f}, Train AUC: {auc:.4f}***")

    model.load_state_dict(best_model_state)
    print(f"Training finished. Best training accuracy: {best_acc:.4f}")
    return model

if __name__ == '__main__':
    jsonl_path = ''
    trained_models = train_gin(jsonl_path)
