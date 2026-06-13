from sklearn.model_selection import train_test_split
import pandas as pd
import transformers, sys
from typing import Literal 
import torch, os, json
import argparse
import subprocess
import time
import ollama
import os
from ollama import chat, Client
from pydantic import BaseModel
from ollama import chat
from pydantic import BaseModel
import json
import subprocess
import time
import os


clarity_mapping = {
    'Explicit': 'Clear Reply',
    'Implicit': 'Ambivalent',
    'Dodging': "Ambivalent",
    'Deflection': "Ambivalent",
    'Partial/half-answer': "Ambivalent",
    'General': "Ambivalent",
    'Declining to answer': "Clear Non-Reply",
    'Claims ignorance': "Clear Non-Reply",
    'Clarification': "Clear Non-Reply",
}

class StructuredOutput(BaseModel):
    Taxonomy_code:int
    #Taxonomy_code:str
    certainty: int


def get_model_prediction(file_path, input_data, add_specific_labels,mappinf):
    texts = []
    idx_n = []
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
        if add_specific_labels:
            all_labels = [mappinf[row["evasion_label"]] for row in data]
        else:
            all_labels = [mappinf[row["clarity_label"]] for row in data]

        for row in data:
            idx = row.get('id', '')
            interview_question = row.get('interview_question', '').strip()
            if input_data == "full":
                Answer = " ".join(row.get('sentenced_answer', [])).strip()
            elif input_data == "all" or input_data == "multi":
                chosen_sentence = row.get('chosen_sentence')
                if chosen_sentence and len(chosen_sentence) > 0:
                    Answer = " ".join([item.get('sentence', '') for item in chosen_sentence]).strip()
                else:
                    print("no answer", input_data)
                    Answer = " ".join(row.get('sentences', [])).strip()

            question = row.get('question', '').strip()

            if add_specific_labels:
                text = f"""Based on a segment of the interview in which the interviewer poses a series of questions, classify the type of response provided by the interviewee for the following question using the following taxonomy:
    Explicit - The information requested is explicitly stated (in the requested form)
    Implicit - The information requested is given, but without being explicitly stated (not in the requested form)
    Dodging - Ignoring the question altogether
    Deflection - Starts on topic but shifts the focus and makes a different point than what is asked
    Partial/half-answer - Offers only a specific component of the requested information.
    General - The information provided is too general/lacks the requested specificity.
    Declining to answer - Acknowledge the question but directly or indirectly refusing to answer at the moment
    Claims ignorance - The answerer claims/admits not to know the answer themselves.
    Clarification - Does not provide the requested information and asks for clarification.

    Here is one small example for each term of the taxonony:
    Question: Do you have your own views about PR at Westminster don’t you? 
    Answer: I do. 
    Label: Explicit 
    Explanation: The answer directly gives the info requested.

    Question: Are you going to watch television? 
    Answer: What else is there to do? 
    Label: Implicit 
    Explanation: They suggest planning to watch TV, despite not explicitly stating it.
    
    Question: Do you like my new dress? 
    Answer: We are late. 
    Label: Dodging 
    Explanation: Does not even acknowledge the question and goes straight to another topic.
    
    Question: Did you eat the last piece of pie? 
    Answer: I have to admit that this was a great recipe, I always like it when there are chocolate chips in the dough. 
    Label: Deflection 
    Explanation: Acknowledges the question but goes on a tangent about the chips, without answering.
    
    Question: Did you enjoy the film? 
    Answer: The directing was great. 
    Label: Partial/half-answer 
    Explanation: Directing is only part of what constitutes a film.
    
    Question: What’s your favorite film? 
    Answer: Fight Club, Filth, and Hereditary. 
    Label: General 
    Explanation: The reply gives three movies instead of one, which makes the desired information unclear. 
    
    Question: The hypothesis I was discussing, wouldn’t you regard that as a defeat? 
    Answer: I am not going to prophesy what will happen. 
    Label: Declining to answer 
    Explanation: Directly stating they won’t answer.
    
    Question: On what precise date did the government order the refit of the HMAS Kanimbla in preparation for its forward deployment to a possible war against Iraq? 
    Answer: I do not know that date. I will find out and let the House know. 
    Label: Claims ignorance 
    Explanation: Claims/admits they don’t have the information.
    
    Question: Was it your decision to release the fund? 
    Answer: You mean the public fund? 
    Label: Clarification 
    Explanation: Gives no data, asks for clarification.

    ### Part of the interview ###
    Interview Question: {interview_question}
    Answer: {Answer}
                
    ### Question ###
    {question}

    Return JSON with: Taxonomy_code: integer (0=Explicit, 1=Implicit, 2=Dodging, 3=Deflection, 
  4=Partial/half-answer, 5=General, 6=Declining to answer, 
  7=Claims ignorance, 8=Clarification), certainty (0 = not at all certain, 100 = completely certain) 
."""
            else:
                text = f"""Based on a segment of the interview in which the interviewer poses a series of questions, classify the type of response provided by the interviewee for the following question using the following taxonomy:
    Clear Reply - The information requested is explicitly stated (in the requested form)
    Ambivalent - The information requested is given in an incomplete way e.g. the answer is too general, partial, implicit, contradictory, diffused, dodging or deflection
    Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification or declining to answer

    Here is one small example for each term of the taxonony:
    Question: Do you have your own views about PR at Westminster don’t you? 
    Answer: I do. 
    Label: Clear Reply 
    Explanation: The answer directly gives the info requested.

    Question: Are you going to watch television? 
    Answer: What else is there to do? 
    Label: Ambivalent 
    Explanation: They suggest planning to watch TV, despite not explicitly stating it.

    Question: Do you like my new dress? 
    Answer: We are late. 
    Label: Ambivalent 
    Explanation: Does not even acknowledge the question and goes straight to another topic.

    Question: Did you eat the last piece of pie? 
    Answer: I have to admit that this was a great recipe, I always like it when there are chocolate chips in the dough. 
    Label: Ambivalent 
    Explanation: Acknowledges the question but goes on a tangent about the chips, without answering.

    Question: Did you enjoy the film? 
    Answer: The directing was great. 
    Label: Ambivalent 
    Explanation: Directing is only part of what constitutes a film.

    Question: What’s your favorite film? 
    Answer: Fight Club, Filth, and Hereditary. 
    Label: Ambivalent 
    Explanation: The reply gives three movies instead of one, which makes the desired information unclear.

    Question: The hypothesis I was discussing, wouldn’t you regard that as a defeat? 
    Answer: I am not going to prophesy what will happen. 
    Label: Clear Non-Reply 
    Explanation: Directly stating they won’t answer.

    Question: On what precise date did the government order the refit of the HMAS Kanimbla in preparation for its forward deployment to a possible war against Iraq? 
    Answer: I do not know that date. I will find out and let the House know. 
    Label: Clear Non-Reply 
    Explanation: Claims/admits they don’t have the information.

    Question: Was it your decision to release the fund? 
    Answer: You mean the public fund? 
    Label: Clear Non-Reply 
    Explanation: Gives no data, asks for clarification. 
                
    ### Part of the interview ###
    Interview Question: {interview_question}
    Answer: {Answer}
                
    ### Question ###
    {question}

    Return JSON with: Taxonomy_code: integer (0 = Clear Reply, 1 = Ambivalent, 2 = Clear Non-Reply), certainty (0 = not at all certain, 100 = completely certain) 
."""

            texts.append(text)
            idx_n.append(idx)
    return texts, all_labels, idx_n


def get_llm_reponse(input, model_id, input_label, input_id, out_file_path, out_file,mapping):
    llm_result = []
    client = Client(host='http://..../', timeout=3600000) #website account that shared by RISE
    
    processed_ids = set()
    if os.path.exists(out_file_path):
        with open(out_file_path, 'r', encoding='utf-8') as f: 
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except:
                    pass
    print(f"processed {len(processed_ids)} item, pass")

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    #for i, l, idx in zip(input, input_label, input_id):
    for i, l, idx in zip([input[106]], [input_label[106]], [input_id[106]]):
        if idx in processed_ids:
            continue  
        pass
      
        retries = 0  
        while True:
            try:
                response = client.chat(
                    model=model_id,
                    messages=[{'role': 'user', 'content': i}],
                    think=True,
                    stream=False,
                    format=StructuredOutput.model_json_schema(),
                    options={'temperature': 0, 'top_k': 1, 'seed': 42, 'repeat_penalty': 1.0}
                )
                thinking = response.message.thinking or ""

                if not response.message.content or response.message.content.strip() == "":
                    thinking = response.message.thinking or "" 
                    print(f"[SKIP] id={idx} empty！alertt")
                    each_item = {
                        "id": idx,
                        "label": l,
                        "prediction": {"Taxonomy_code": -1, "certainty": -1},
                        "thinking": thinking
                    }
                    llm_result.append(each_item)
                    out_file.write(json.dumps(each_item, ensure_ascii=False) + '\n')
                    out_file.flush()
                    break

                predicted = StructuredOutput.model_validate_json(response.message.content)
                result = predicted.model_dump()
                #print("model response ",result)

                #predicted_int_result = mapping[result["Taxonomy_code"]]
                predicted_int_result = result["Taxonomy_code"]

                #print("transfered",predicted_int_result)
                each_item = {
                    "id": idx,
                    "label": l,
                    "prediction": {
                    "Taxonomy_code":predicted_int_result,  
                    "certainty": result["certainty"]
                    },
                    "thinking": thinking
                }
                llm_result.append(each_item)
                out_file.write(json.dumps(each_item, ensure_ascii=False) + '\n')
                out_file.flush()
                print(f"[{idx}/307] id={idx} prediction={result}") 
                break
           
            except Exception as e:
                retries += 1
                print(f"[ERROR] id={idx}: {e} | Retrying {retries}/{MAX_RETRIES}")
                if retries >= MAX_RETRIES:
                    print(f"[SKIP] id={idx} exceed the trial")
                    break
                time.sleep(RETRY_DELAY)

    return llm_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero Shot deepseek family")
    parser.add_argument("--add_specific_labels", action="store_true",
                        help="Include this flag to indicate whether specific labels should be added or not.")

    args = parser.parse_args()
    add_specific_labels = args.add_specific_labels

    if add_specific_labels:
        num_labels = 9
        mapping_labels = {
            'Explicit': 0, 'Implicit': 1, 'Dodging': 2,
            'Deflection': 3, 'Partial/half-answer': 4, 'General': 5,
            'Declining to answer': 6, 'Claims ignorance': 7, 'Clarification': 8   }
    else:
        num_labels = 3
        mapping_labels = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}

    experiments_input = [ "full", "all", "multi"]
    all_results = []

    for i in experiments_input:
        if i == "full":
           test =   "/raid/jiawen/proj_master/HF_data/runningdata/test_results_full_updated.jsonl"
        if i == "all":
            test =   "/raid/jiawen/proj_master/HF_data/runningdata/test_results_all_updated.jsonl"
        if i == "multi":
            test =   "/raid/jiawen/proj_master/HF_data/runningdata/test_results_multi_updated.jsonl"

        test_texts, test_labels, test_id = get_model_prediction(test, i, add_specific_labels,mapping_labels)



        models = [
            'deepseek-r1:1.5b',
            'deepseek-r1:7b',
            'deepseek-r1:8b',
            'deepseek-r1:14b',
            'deepseek-r1:32b',
            'deepseek-r1:70b'
        ]
        print(f"Loaded {len(test_texts)} samples")

        for model_id in models:
            print(f"  Running {model_id}...")
            model_name_safe = model_id.replace('/', '_').replace(':', '_')
            output_file = f"{i}_{add_specific_labels}_{model_name_safe}_result1.jsonl"
            with open(output_file, 'a', encoding='utf-8') as f:
                result_from_deepseek = get_llm_reponse(test_texts, model_id, test_labels, test_id,output_file,f,mapping_labels)
                import time
                time.sleep(2)
            print(f"    Saved to {output_file}")