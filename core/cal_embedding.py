import json
from collections import defaultdict
import os

input_path = ''
output_path = ''
right_path=''
error_path=''

saved=[]
if os.path.exists(output_path):
    with open(output_path,'r') as f:
        for line in f:
            data = json.loads(line.strip()) 
            if data:
                saved.append(data['id'])
                
right_ids=[]
error_ids=[]
with open(right_path,'r') as f:
    for line in f:
        data = json.loads(line.strip()) 
        if data:
            right_ids.append(data['ID'])
with open(error_path,'r') as f:
    for line in f:
        data = json.loads(line.strip()) 
        if data:
            error_ids.append(data['ID'])
            
  
emb_id=[]
with open(input_path, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        
        if data['id'] in right_ids or data['id'] in error_ids:
            emb_id.append(data['id'])


with open(input_path, 'r') as f:
    for line in f:
        entity_attention_dict = defaultdict(lambda: defaultdict(list)) 
        relation_attention_dict = defaultdict(lambda: defaultdict(list)) 
        data = json.loads(line.strip())
        if data['id'] in saved:
            continue
        
        if data['attention_scores'][0]['span_scores']['content']==[]:
            continue
        
        tag=""
        if data['id'] in right_ids:
            tag='right'
        elif data['id'] in error_ids:
           tag='error'
        else:
            continue
           
        
        for attn_info in data['attention_scores']:
            i=0
            layer_id = attn_info['layer_id']
            head_id = attn_info['head_id']
            scores = attn_info['span_scores']['content']
            
            layers=[23,24,25,26,27]
            if layer_id not in layers:
                continue
            
            for item in scores:
                entity_content = item['content']
                attention_score = item['attention_score']
                key = (layer_id, head_id)
                if item['type'] == 'entity':
                    entity_attention_dict[entity_content][key].append(attention_score)
                if item['type'] == 'relation':
                    relation_attention_dict[(scores[i]['content'],entity_content,scores[i+1]['content'])][key].append(attention_score)
                    i+=2

        entity_avg_matrix = []
        relation_avg_matrix = []

        all_keys = set()
        for v in entity_attention_dict.values():
            all_keys.update(v.keys())

        max_layers = max(k[0] for k in all_keys) + 1
        max_heads = max(k[1] for k in all_keys) + 1

        for entity, scores_dict in entity_attention_dict.items():
            matrix = [[0.0 for _ in range(max_heads)] for _ in range(max_layers)]
            count_matrix = [[0 for _ in range(max_heads)] for _ in range(max_layers)]

            for (layer, head), scores in scores_dict.items():
                avg_score = sum(scores) / len(scores)
                matrix[layer][head] = avg_score
                count_matrix[layer][head] += 1

            entity_avg_matrix.append({"content":entity, "matrix":matrix[-len(layers):]})
        
        for relation, scores_dict in relation_attention_dict.items():
            matrix = [[0.0 for _ in range(max_heads)] for _ in range(max_layers)]
            count_matrix = [[0 for _ in range(max_heads)] for _ in range(max_layers)]

            for (layer, head), scores in scores_dict.items():
                avg_score = sum(scores) / len(scores)
                matrix[layer][head] = avg_score
                count_matrix[layer][head] += 1

            relation_avg_matrix.append({"content":relation, "matrix":matrix[-len(layers):]})
        
        
        out_dict={}
        out_dict['id']=data['id']
        out_dict['label']=tag
        out_dict['entity_content']=entity_avg_matrix
        out_dict['relation_content']=relation_avg_matrix
        with open(output_path, 'a') as out_f:
            outstr = json.dumps(out_dict, ensure_ascii=False)
            out_f.write(outstr+'\n')