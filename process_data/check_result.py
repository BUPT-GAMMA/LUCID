import json
import re
from data.KGQA import KGQADataset
import argparse
from SPARQLWrapper import SPARQLWrapper, JSON

def align(dataset_name, question_string, data, ground_truth_datas, answer_class = None):
    answer_list= []
    origin_data = [j for j in ground_truth_datas if j[question_string] == data[question_string]]
    if len(origin_data) == 0:
        return [], []
    origin_data = origin_data[0]
    if dataset_name == 'cwq':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'graliqa':
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'qald_10-en':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
        
    elif dataset_name == 'webquestions':
        answer_list = origin_data["answers"]

    elif dataset_name == 'trex' or dataset_name == 'zeroshotre':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'creak':
        answer = origin_data['label']
        answer_list.append(answer)
    elif 'poisoned' in dataset_name:
        if answer_class is None:
            answers = origin_data["answer"]
            for answer in answers.values(): 
                answer_list.append(answer)
        else:
            for index in answer_class:
                answer = origin_data["poisoned_answer"+str(index + 1)].popitem()
                answer_list.append(answer[1])
    return origin_data["ID"], list(set(answer_list))
    
def check_string(string):
    return "{" in string

def clean_results(string):
    if "{" in string:
        start = string.find("{") + 1
        end = string.find("}")
        content = string[start:end]
        return content
    else:
        return "NULL"
    

def check_refuse(string):
    refuse_words = ["however", "sorry"]
    return any(word in string.lower() for word in refuse_words)


def exact_match(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False

def save_result2json(dataset_name, num_right, num_error, total_nums, method, model):
    results_data = {
        'dataset': dataset_name,
        'Exact Match': float(num_right/total_nums),
        'Right Samples': num_right,
        'Error Sampels': num_error
    }
    with open('{}_{}_{}_results.json'.format(method, dataset_name, model), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
                     
def extract_content(s):
    matches = re.findall(r'\{(.*?)\}', s)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[0]
    else:
        return 'NULL'

def get_tag(data, ground_truth_datas, link_data):
    question = data["question"]
    origin_data = [j for j in ground_truth_datas if j["question"] == question]
    data_id = origin_data[0]["ID"]
    for j in link_data:
        if j["ID"] == data_id and "tag" in j["relation"]:
            return j["relation"]["tag"]
        
def count_paths(data):
    paths = data.get("reasoning_chains", [])
    return len(paths)

count_results = []

SPARQLPATH = "http://localhost:8890/sparql"
def execute_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--dataset', type=str, default='poisoned_graliqa')
    parser.add_argument('--method', type=str, default='ToG')
    parser.add_argument('--answer_class', type=list, default=None)

    args = parser.parse_args()
    
    dataset = args.dataset
    answer_class = args.answer_class
    model = args.model
    output_datas = []
    with open("{}_{}_{}.jsonl".format(args.method, dataset, model), encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            output_datas.append(data)
    ground_truth_datas = KGQADataset(dataset)

    num_right = 0
    num_error = 0
    statics = 0
    statics_error = 0

    id_list = []
    id_list_error = []
    answer_list = {}
    
    for data in output_datas:
        ID, answers = align(dataset.lower(), ground_truth_datas.question_string, data, ground_truth_datas, answer_class)
        if len(answers) == 0:
            continue
        answer_list[ID] = answers
        results = data['results']
        if check_string(results):
            response = extract_content(results)
            if response=="NULL":
                response = results
            else:
                if exact_match(response, answers):
                    num_right += 1
                    id_list.append(ID)
                else:
                    num_error += 1
                    id_list_error.append(ID)
        else:
            response = results
            if check_string(response):
                continue
            if exact_match(response, answers):
                num_right+=1
                id_list.append(ID)
            else:
                num_error+=1
                id_list_error.append(ID)

    print("Exact Match: {}".format(float(num_right/len(output_datas))))
    print("right: {}, error: {}".format(num_right, num_error))

    save_result2json(dataset, num_right, num_error, len(output_datas), args.method, args.model)
    
    for data in output_datas:
        if int(data["ID"]) in id_list:
            count = count_paths(data)
            if count > 0 and count < 50:
                count_results.append({
                    "ID": data["ID"],
                    "question": data["question"],
                    "results": data["results"],
                    "answer": answer_list[int(data["ID"])],
                    "reasoning_chains": data["reasoning_chains"],
                    "path_count": count
                })

    output_file_path = "{}_{}_{}_right.jsonl".format(args.method, dataset, model)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for result in count_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Path counts saved to {output_file_path}")
    
    for data in output_datas:
        if int(data["ID"]) in id_list_error:
            count = count_paths(data)
            if count > 0 and count < 50:
                count_results.append({
                    "ID": data["ID"],
                    "question": data["question"],
                    "results": data["results"],
                    "answer": answer_list[int(data["ID"])],
                    "reasoning_chains": data["reasoning_chains"],
                    "path_count": count
                })
            
    output_file_path = "{}_{}_{}_error.jsonl".format(args.method, dataset, model)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for result in count_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Path counts saved to {output_file_path}")