import json


method=['Readi','ToG','StructGPT']
dataset=['cwq', 'graliqa', 'WebQSP', 'qald']

for m in method:
    for d in dataset:
        res_path=f''
        input_path=f''
        output_path=f''

        label={}
        with open(input_path,'r') as f:
            for line in f:
                data = json.loads(line)
                if data['label']==1 or data['label']==2:
                    label[data['ID']]=1
                else:
                    label[data['ID']]=0
                
        data=[]
        with open(res_path,'r') as f:
            for line in f:
                d=json.loads(line)
                if d['ID'] in label:
                    data.append(d)
                    
        with open(output_path,'w') as f:
            for d in data:
                dict_res={}
                dict_res['ID']=d['ID']
                dict_res['question']=d['question']
                dict_res['results']=d['results']
                dict_res['reasoning_chains']=d['reasoning_chains']
                dict_res['label']=label[d['ID']]
                f.write(json.dumps(dict_res)+'\n')
        