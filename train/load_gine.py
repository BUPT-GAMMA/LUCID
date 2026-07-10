import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.metrics import roc_auc_score
import numpy as np
from train_gine import GINE, GraphDataset, process_single_data,  custom_collate, build_graph_from_single_data
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--use_edge_weight', type=bool, default=True)
args = parser.parse_args()

def load_and_evaluate_gine(model_path, test_data_path, embedding_path, label_path, sim_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_data_list = []
    with open(test_data_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            test_data_list.append(data)

    label_dict={}
    with open(label_path,'r')as f:
        for line in f:
            data=json.loads(line)
            label_dict[str(data['ID'])]=data['label']
    
    embeddings_data = {}
    with open(embedding_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'id' in data:
                if str(data['id']) not in label_dict:
                    continue
                embeddings_data[str(data['id'])] = data
                embeddings_data[str(data['id'])]['label']=label_dict[str(data['id'])]

    sim_scores = {}
    if sim_path:
        with open(sim_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                scores = {}
                for rel in data.get("similarity", []):
                    if "relation" in rel:
                        scores[rel["relation"]] = rel["score"]
                    elif "entity" in rel:
                        scores[rel["entity"]] = rel["score"]
                sim_scores[data["id"]] = scores
    
    processed_data = []
    for data in test_data_list:
        if 'ID' in data and data['ID'] in embeddings_data:
            data_dict = {
                'chains': data.get('reasoning_chains', []),
                'embeddings': embeddings_data[data['ID']]
            }
            if sim_path and data['ID'] in sim_scores:
                data_dict['similarity'] = sim_scores[data['ID']]
            processed_data.append(data_dict)
    
    test_dataset = GraphDataset(processed_data)
    print(f"Test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    print(os.path.basename(model_path))
    
    sample_data = processed_data[0]
    sim_scores_sample = sample_data.get('similarity', None)
    entity_embeddings, relation_embeddings, _, _, _ = process_single_data(
        sample_data['embeddings'],
        sample_data['embeddings']['label'],
        sim_scores_sample
    )
    
    in_channels = entity_embeddings.size(1) * entity_embeddings.size(2)
    hidden_channels=512
    edge_dim = relation_embeddings.size(1) * relation_embeddings.size(2)
    
    model = GINE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,  
        edge_dim=edge_dim,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}, Train Acc: {checkpoint['train_acc']:.4f}")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []
    results = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            batch_graphs = []
            batch_labels = []
            batch_ids = [] 
            
            for data_item in batch_data:
                if args.use_edge_weight:
                    sim_scores = data_item.get('similarity', None)
                else:
                    sim_scores = None
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
                    data_id = data_item['embeddings'].get('id', 'unknown')
                    all_ids.append(data_id)
                    batch_ids.append(data_id)  # 保存ID
            
            if not batch_graphs:
                continue
            
            batched_graphs = Batch.from_data_list(batch_graphs).to(device)
            batch_labels = torch.stack(batch_labels).to(device)
            
            pred = model(
                x=batched_graphs.x,
                edge_index=batched_graphs.edge_index,
                edge_attr=batched_graphs.edge_attr,
                edge_sim=batched_graphs.edge_sim,
                batch=batched_graphs.batch
            )
            
            for i, (data_id, pred_value) in enumerate(zip(batch_ids, pred.cpu().numpy())):
                results.append({
                    "ID": data_id,
                    "prediction": float(pred_value),
                    "label": int(batch_labels[i].cpu().numpy())
                })
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
   
    
    from sklearn.metrics import roc_curve
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    predicted = (np.array(all_preds) > thresh).astype(float)
        
    accuracy = np.mean(predicted == all_labels)
    
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, predicted)
    recall = recall_score(all_labels, predicted)
    f1 = f1_score(all_labels, predicted)
    print(f"\nTest Results for threshold {thresh:.2f}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    print(f'AVG: {(accuracy+auc+f1)/3.0:.4f}')
    
    dataset_name=os.path.basename(label_path).split('_')[1]
    method_name=os.path.basename(label_path).split('_')[0]
    
    return {
        "dataset": dataset_name,
        "method": method_name,
        "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

if __name__ == '__main__':
   
    dataset_names=['graliqa','WebQSP','qald']
    # method_names=['Readi','StructGPT']
    method_names=['ToG']
    
    all_results = []
    for dataset in dataset_names:
        for method in method_names:
            model_path = f''
            test_data_path = f''
            embedding_path = f''
            label_path= f''
            sim_path = f''
            
            print(f"{dataset}-{method}********************")
            eval_result=load_and_evaluate_gine(model_path, test_data_path, embedding_path, label_path, sim_path)
            if eval_result: 
                all_results.append(eval_result)
            print("\n")
