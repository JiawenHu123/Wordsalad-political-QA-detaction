from sklearn.model_selection import train_test_split
import pandas as pd
import transformers, sys
import torch,os, json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import time
import os
from ollama import chat, Client
from pydantic import BaseModel
from ollama import chat
from pydantic import BaseModel
import json
import subprocess
import time
import os

CLIENT_32B = Client(host='http://....../', timeout=3600) # the account that shared by RISE
CLIENT_70B = Client(host='http://...../', timeout=3600)
 
 
MODEL_WORKER = "deepseek-r1:32b"   # paralleled worker
MODEL_JUDGE  = "deepseek-r1:70b"   # judger

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
    
LABELS_9 = """
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
    Explanation: Gives no data, asks for clarification."""

LABELS_3 = """
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
    Explanation: Gives no data, asks for clarification. """

class Worker(BaseModel):
    Taxonomy_code:int
    certainty: int
    reason: str 



class Judger(BaseModel):
    Final_Taxonomy_code: int
    final_certainty: int

#collect the data
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
                    Answer = " ".join(row.get('sentences', [])).strip()

            question = row.get('question', '').strip()
        
            texts.append({
                "interview_question": interview_question,
                "answer": Answer,
                "question": question,
            })
            idx_n.append(idx)
    return texts,all_labels,idx_n

#construct debate prompt
# method 1
def political_prompt(interview_question, answer, question, add_specific_labels):
    taxonomy = LABELS_9 if add_specific_labels else LABELS_3
    codes = """Taxonomy_code (int (0=Explicit, 1=Implicit, 2=Dodging, 3=Deflection, 
  4=Partial/half-answer, 5=General, 6=Declining to answer, 
  7=Claims ignorance, 8=Clarification)), certainty (0-100), reason (explaining your key deciding factor).""" if add_specific_labels else \
    "Taxonomy_code (int (0=Clear Reply, 1=Ambivalent, 2=Clear Non-Reply)), certainty (0-100), reason (explaining your key deciding factor)."

    return f"""You are a POLITICAL DISCOURSE ANALYST EXPERT specializing in evasion, deflection, framing, and accountability strategies in political Q&A.

### Taxonomy:
{taxonomy}

### Interview Question (context): {interview_question}
### Answer to classify: {answer}
### Specific question being asked: {question}

### Instructions:
Step 1 - Find the KEY SENTENCE: Identify the single sentence in the answer that most directly responds to the specific question from a political accountability perspective. If no such sentence exists, note that.
Step 2 - Classify: Base your classification primarily on that key sentence, not the overall tone of the answer.

Return JSON with: {codes}"""


def Economist_prompt(interview_question, answer, question, add_specific_labels):
    taxonomy = LABELS_9 if add_specific_labels else LABELS_3
    codes = """Taxonomy_code (int (0=Explicit, 1=Implicit, 2=Dodging, 3=Deflection, 
  4=Partial/half-answer, 5=General, 6=Declining to answer, 
  7=Claims ignorance, 8=Clarification)), certainty (0-100), reason (explaining your key deciding factor).""" if add_specific_labels else \
    "Taxonomy_code (int (0=Clear Reply, 1=Ambivalent, 2=Clear Non-Reply)), certainty (0-100), reason (explaining your key deciding factor)."

    return f"""You are an ECONOMIST EXPERT specializing in cost-benefit framing, resource allocation, and market logic in policy Q&A.

### Taxonomy:
{taxonomy}

### Interview Question (context): {interview_question}
### Answer to classify: {answer}
### Specific question being asked: {question}

### Instructions:
Step 1 - Find the KEY SENTENCE: Identify the single sentence in the answer that most directly responds to the specific question from an economic reasoning perspective. If no such sentence exists, note that.
Step 2 - Classify: Base your classification primarily on that key sentence, not the overall framing of the answer.

Return JSON with: {codes}"""


def psychologist_prompt(interview_question, answer, question, add_specific_labels):
    taxonomy = LABELS_9 if add_specific_labels else LABELS_3
    codes = """Taxonomy_code (int (0=Explicit, 1=Implicit, 2=Dodging, 3=Deflection, 
  4=Partial/half-answer, 5=General, 6=Declining to answer, 
  7=Claims ignorance, 8=Clarification)), certainty (0-100), reason (explaining your key deciding factor).""" if add_specific_labels else \
    "Taxonomy_code (int (0=Clear Reply, 1=Ambivalent, 2=Clear Non-Reply)), certainty (0-100), reason (explaining your key deciding factor)."

    return f"""You are a PROFESSIONAL PSYCHOLOGIST EXPERT specializing in cognitive biases, emotional framing, motivation, defensiveness, and social identity in Q&A settings.

### Taxonomy:
{taxonomy}

### Interview Question (context): {interview_question}
### Answer to classify: {answer}
### Specific question being asked: {question}

### Instructions:
Step 1 - Find the KEY SENTENCE: Identify the single sentence in the answer that most directly responds to the specific question from a psychological motivation perspective. If no such sentence exists, note that.
Step 2 - Classify: Base your classification primarily on that key sentence, not the emotional tone of the answer.

Return JSON with: {codes}"""


def judge_prompt(interview_question, answer, question, political, economist, psychologist, add_specific_labels):
    codes = """Final_Taxonomy_code (int (0=Explicit, 1=Implicit, 2=Dodging, 3=Deflection, 
  4=Partial/half-answer, 5=General, 6=Declining to answer, 
  7=Claims ignorance, 8=Clarification)), final_certainty (0-100).""" if add_specific_labels else \
    "Final_Taxonomy_code (int (0=Clear Reply, 1=Ambivalent, 2=Clear Non-Reply)), final_certainty (0-100)."

    return f"""You are the JUDGE. Your role is to critically evaluate the three experts' analyses and render a final verdict.

### Interview Question (context): {interview_question}
### Answer to classify: {answer}
### Specific question being asked: {question}

### Political Expert result: {json.dumps(political)}
### Economist result: {json.dumps(economist)}
### Psychologist result: {json.dumps(psychologist)}

### Aggregation rules:
1. Identify which expert found the most decisive key sentence and whether it was correctly interpreted.
2. Use majority vote as a starting point.
3. A well-justified minority reason can override the majority — evaluate reason quality, not just vote count.
4. Do NOT average certainties. Set final_certainty based on your own confidence in the final verdict.

Return JSON with: {codes}"""

def call_worker(prompt):
    response = CLIENT_32B.chat(
            model=MODEL_WORKER,
            messages=[{'role': 'user', 'content': prompt}],
            think=True,
            stream=False,
            format=Worker.model_json_schema(),
            options={'temperature': 0, 'top_k': 1, 'seed': 42, 'repeat_penalty': 1.0}
        )
    thinking = response.message.thinking or ""
    if not response.message.content or response.message.content.strip() == "":
        return None, thinking

    predicted = Worker.model_validate_json(response.message.content)
    result = predicted.model_dump()      
    return result, thinking

    
 
def call_judge(prompt):
    response = CLIENT_70B.chat(
            model=MODEL_JUDGE,
            messages=[{'role': 'user', 'content': prompt}],
            think=True,
            stream=False,
            format=Judger.model_json_schema(),
            options={'temperature': 0, 'top_k': 1, 'seed': 42, 'repeat_penalty': 1.0}
        )
    thinking = response.message.thinking or ""
    if not response.message.content or response.message.content.strip() == "":
        return None, thinking

    predicted = Judger.model_validate_json(response.message.content)
    result = predicted.model_dump()   
    
    return result, thinking
 

def debate_classify(interview_question, answer, question,add_specific_labels):
    # Step 1: three agents run in parallel
    prompts = {
        "political":  political_prompt(interview_question, answer, question,add_specific_labels),
        "economist": Economist_prompt(interview_question, answer, question, add_specific_labels),
        "psychologist": psychologist_prompt(interview_question, answer, question, add_specific_labels)}
    agent_results = {}
    with ThreadPoolExecutor(max_workers=3) as executor: #Used for executing multiple tasks concurrently
        futures = {}
        for name, prompt in prompts.items():
            future = executor.submit(call_worker, prompt)  # submit one mission
            futures[future] = name 

        for fut in as_completed(futures):
            name = futures[fut] # agent name
            result, thinking = fut.result()
            if result is None:
                print(f"    [{name}] pass the empty")
                agent_results[name] = None
            else:
                agent_results[name] = {**result, "thinking": thinking}
                print(f"Taxonomy_code:   {result.get('Taxonomy_code')}, certainty: {result.get('certainty')}")
    
    for checkpoint in agent_results.values(): #what if the answer from agent is null，jump
        if checkpoint is None:
            return None        

    # Step 2: judge aggregates
    print("    [Judge] Aggregating...")
    judge_result, judge_thinking = call_judge(
        judge_prompt(interview_question, answer, question,
            {k: v for k, v in agent_results["political"].items()  if k != "thinking"},#exclude the token of reasoning
            {k: v for k, v in agent_results["economist"].items() if k != "thinking"},
            {k: v for k, v in agent_results["psychologist"].items() if k != "thinking"},
            add_specific_labels)
    )
    if judge_result is None:
        return None

    if add_specific_labels:
        return {
            "Taxonomy_code":   judge_result.get("Final_Taxonomy_code"),
            "final_certainty":        judge_result.get("final_certainty"),
            "judge_thinking":         judge_thinking,
            "agent_political":        agent_results["political"],
            "agent_economist":        agent_results["economist"],
            "agent_psychologist":     agent_results["psychologist"],}
    else:
        return {
            "Taxonomy_code":   judge_result.get("Final_Taxonomy_code"),
            "final_certainty":        judge_result.get("final_certainty"),
            "judge_thinking":         judge_thinking,
            "agent_political":        agent_results["political"],
            "agent_economist":        agent_results["economist"],
            "agent_psychologist":     agent_results["psychologist"],}

 
def run(texts, labels, ids, output_file, mapping, add_specific_labels):
    # lost in the process
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f: 
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except:
                    pass
    print(f"processed {len(processed_ids)} item，pass")

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for text, label, idx in zip([texts[106]], [labels[106]], [ids[106]]):  
            if idx in processed_ids:
                continue
            pass

            retries = 0  
            while True:
                try:
                    result = debate_classify(
                        text["interview_question"],
                        text["answer"],
                        text["question"],add_specific_labels) # get the answer from 3 agents
                    
                    if add_specific_labels:  
                        if result is None:
                            each_item = {
                                "id": idx, "label": label,
                                "prediction": {"Taxonomy_code": -1, "certainty": -1},
                                }
                        
                        else:
                            each_item = {
                                "id": idx, "label": label,
                                "prediction": {
                                "Taxonomy_code":  result["Taxonomy_code"],
                                "final_certainty": result["final_certainty"],
                                },
                                "judge_thinking": result["judge_thinking"],
                                "agents": {
                                    "political":   result["agent_political"],
                                    "economist":   result["agent_economist"],
                                    "psychologist":result["agent_psychologist"],
                                    }
                                }

                        out_file.write(json.dumps(each_item, ensure_ascii=False) + '\n')
                        out_file.flush()
                        print(f"[{idx}] label={label} | pred={each_item['prediction']}")
                        break

                    else:
                        if result is None:
                            each_item = {
                                "id": idx, "label": label,
                                "prediction": {"Taxonomy_code": -1, "certainty": -1},
                                }
                        
                        else:
                            each_item = {
                                "id": idx, "label": label,
                                "prediction": {
                                "Taxonomy_code": result["Taxonomy_code"],
                                "final_certainty": result["final_certainty"],
                                },
                                "judge_thinking": result["judge_thinking"],
                                "agents": {
                                    "political":   result["agent_political"],
                                    "economist":   result["agent_economist"],
                                    "psychologist":result["agent_psychologist"],
                                    }
                                }

                        out_file.write(json.dumps(each_item, ensure_ascii=False) + '\n')
                        out_file.flush()
                        print(f"[{idx}] label={label} | pred={each_item['prediction']}")
                        break

                except Exception as e:
                    retries += 1
                    print(f"[ERROR] id={idx}: {e} | Retrying {retries}/{MAX_RETRIES}")
                    if retries >= MAX_RETRIES:
                        print(f"[SKIP] id={idx} exceed，failed toooo many times")
                        break
                    time.sleep(RETRY_DELAY)

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
 
if __name__ == "__main__":
    input_data = "multi"
    test_file = "/raid/jiawen/proj_master/HF_data/runningdata/test_results_multi_updated.jsonl"
    
    # two label modes
    label_modes = [
        (True, "9class"),
          (False, "3class"),  # (add_specific_labels, name)
        ]
    
    for add_specific_labels, label_name in label_modes:
        print(f"{'1' if add_specific_labels else '2'} start {label_name} exper")        
        if add_specific_labels:
            mapping_labels = {
                'Explicit': 0, 'Implicit': 1, 'Dodging': 2,
                'Deflection': 3, 'Partial/half-answer': 4, 'General': 5,
                'Declining to answer': 6, 'Claims ignorance': 7, 'Clarification': 8,
            }
        else:
            mapping_labels = {
                "Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2
            }

        test_texts, test_labels, test_ids = get_model_prediction(
            test_file, input_data, add_specific_labels, mapping_labels) 
        print(f"Loaded {len(test_texts)} samples")
 
        model_name_safe = MODEL_WORKER.replace('/', '_').replace(':', '_')
        output_file = f"debate_{input_data}_{add_specific_labels}_{model_name_safe}_result_0temp.jsonl"
 
        run(test_texts, test_labels, test_ids, output_file, mapping_labels,add_specific_labels)
        print(f"Saved to {output_file}")
        time.sleep(2)