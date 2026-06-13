import torch
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser(description='')
    
# Adding arguments
parser.add_argument('--experiment', type=str)
args = parser.parse_args()

experiment = args.experiment


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
    
    

def get_data(file_path, input_data, experiment,mappinf):
    all_texts = []
    all_labels = []
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
        if experiment == "evasion_based_clarity":
            all_labels = [mappinf[row["evasion_label"]] for row in data]
            
        elif experiment == "direct_clarity":
            all_labels = [mappinf[row["clarity_label"]] for row in data]

        for row in data:
            Question = row.get('interview_question', '').strip()
            Subquestion = row.get('question', '').strip()
            if input_data == "full":
                Answer = " ".join(row.get('sentenced_answer', [])).strip()
            elif input_data == "all" or input_data == "multi":
                chosen_sentence = row.get('chosen_sentence')
                if chosen_sentence and len(chosen_sentence) > 0:
                    Answer = " ".join([item.get('sentence', '') for item in chosen_sentence]).strip()
                else:
                    Answer = " ".join(row.get('sentences', [])).strip()
            combined = f"[QUESTION] {Question}\n[ANSWER] {Answer}\n[SUBQUESTION] {Subquestion}"
            all_texts.append(combined)
    return all_texts, all_labels


class ModelTrain_logisticREgression:
    
    def __init__(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        self.X_train = x_train
        self.y_train = y_train
        self.X_val = x_dev
        self.y_val = y_dev
        self.X_test = x_test
        self.y_test = y_test
 
    def train_with_validation(self, params):
        model = LogisticRegression(**params)
        model.fit(self.X_train, self.y_train)
        
        # train
        y_train_pred = model.predict(self.X_train)
        train_report = classification_report(self.y_train, y_train_pred, output_dict=True)

        # dev
        y_val_pred = model.predict(self.X_val)
        dev_report = classification_report(self.y_val, y_val_pred, output_dict=True)

        
        return {
            'model': model,
            'train_report':train_report,
            'dev_report':dev_report
        }
    
    def parameter_search(self, param_search):
        best_score = 0
        best_model = None
        best_params = None
        all_results = []
        combinations = []
        
        print("start searching...")
        for C in param_search['C']:
            for solver in param_search['solver']:
                combinations.append((C, solver))
                
        for C, solver in tqdm(combinations, desc="Searching Hyperparameters"):
            params = {'solver': solver,
                'C': C,
                'max_iter': 3000,
                'class_weight': 'balanced',
                'random_state': 42
            }
            result = self.train_with_validation(params)
            result_info = {
                'C': C,
                'solver': solver,
                'val_f1': result['dev_report']['macro avg']['f1-score'],
                'val_acc': result['dev_report']['accuracy'],
                'train_f1': result['train_report']['macro avg']['f1-score'],
                'train_acc': result['train_report']['accuracy'],

            }
                    
            all_results.append(result_info)
            
            if result_info['val_f1'] > best_score:
                best_score = result_info['val_f1']
                best_model = result['model']
                best_params = params
        
        print(f"\Best parameter: {best_params}")
        print(f"Best F1: {best_score:.4f}")
        self.all_results = all_results
        return best_model, best_params
    
    def final_evaluation(self, model,label_names=None):
        """Evalusation on train"""
        y_test_pred = model.predict(self.X_test)
        test_report = classification_report(self.y_test, y_test_pred, target_names=label_names, output_dict=True)

        return {'test_report':test_report}

def run(experiment, input_data_type, train_path, test_path):

    print("Experiment starting!!!")
    if experiment == "evasion_based_clarity":
        num_labels = 9
        mapping_labels = {'Explicit': 0, 'Implicit': 1, 'Dodging': 2, 'Deflection': 3, 'Partial/half-answer': 4, 'General': 5, 'Declining to answer': 6, 'Claims ignorance': 7, 'Clarification': 8}
        label_names = list(mapping_labels.keys())

    elif experiment == "direct_clarity":
        num_labels = 3
        mapping_labels = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}
        label_names = list(mapping_labels.keys())

    
    all_train_texts, all_train_labels = get_data(train_path, input_data_type, experiment,mapping_labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(all_train_texts, all_train_labels, test_size=0.2, random_state=42, stratify=all_train_labels)  
    test_texts, test_labels = get_data(test_path, input_data_type, experiment,mapping_labels)

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer( max_features=10000,  min_df=5, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    trainer = ModelTrain_logisticREgression(X_train, train_labels, 
                                            X_val, val_labels, 
                                            X_test, test_labels)
    # searching params
    param_search = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga']
    }

    best_model, best_params = trainer.parameter_search(param_search)

    #testing!!!
    result = trainer.final_evaluation(best_model, label_names=label_names)
    report_dict = result['test_report']
    test_acc       = report_dict['accuracy']
    test_macro_f1  = report_dict['macro avg']['f1-score']
    test_macro_p   = report_dict['macro avg']['precision']
    test_macro_r   = report_dict['macro avg']['recall']
    test_weighted_f1 = report_dict['weighted avg']['f1-score']
    test_weighted_p  = report_dict['weighted avg']['precision']
    test_weighted_r  = report_dict['weighted avg']['recall']

    print(f"Test Accuracy:          {test_acc:.4f}")
    print(f"Test Macro    P/R/F1:   {test_macro_p:.4f} / {test_macro_r:.4f} / {test_macro_f1:.4f}")
    print(f"Test Weighted P/R/F1:   {test_weighted_p:.4f} / {test_weighted_r:.4f} / {test_weighted_f1:.4f}")


    row = {
        'model': 'TF-IDF+LR', 'experiment': experiment, 'input_type': input_data_type,
        'accuracy':           test_acc,
        'macro_precision':    test_macro_p,
        'macro_recall':       test_macro_r,
        'macro_f1':           test_macro_f1,
        'weighted_precision': test_weighted_p,
        'weighted_recall':    test_weighted_r,
        'weighted_f1':        test_weighted_f1,
        'train_samples': len(train_texts), 'test_samples': len(test_texts),
        'num_features': X_train.shape[1]
    }

    for label in label_names:
        row[f'{label}_precision'] = round(report_dict[label]['precision'], 4)
        row[f'{label}_recall']    = round(report_dict[label]['recall'],    4)
        row[f'{label}_f1']        = round(report_dict[label]['f1-score'],  4)
        row[f'{label}_support']   = int(report_dict[label]['support'])
    csv_path = f"detailed_results_{experiment}.csv"
    row_df = pd.DataFrame([row])
    if os.path.isfile(csv_path):
        row_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        row_df.to_csv(csv_path, mode='w', header=True, index=False)
    print(f"Results saved to {csv_path}")


    import joblib
    model_save_path = f"best_tfidf_lr_{experiment}_{input_data_type}.joblib"

    joblib.dump({'vectorizer': vectorizer, 'model': best_model,'test_accuracy': test_acc, 'test_macro_f1': test_macro_f1, 'best_params': best_params}, model_save_path)
    print(f"Model saved to {model_save_path}")


    if hasattr(trainer, 'all_results'):
        results_df = pd.DataFrame(trainer.all_results)
        plt.figure(figsize=(12, 5))   
        plt.ylim(0, 1.0)              
        plt.xlim(min(param_search['C']) * 0.5, max(param_search['C']) * 2)


        
        # drow line
        for solver in results_df['solver'].unique():
            solver_data = results_df[results_df['solver'] == solver]
            plt.plot(solver_data['C'], solver_data['val_f1'], 
                    marker='o', label=f'solver={solver}')
        
        plt.xscale('log')
        plt.xlabel('Regularization Strength (C)')
        plt.ylabel('Validation Macro F1')
        plt.title(f'Parameter Search Results - {experiment} ({input_data_type})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        # save pic
        plot_path = f"param_search_{experiment}_{input_data_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight') 
        print(f"Plot saved to {plot_path}")
        plt.close()

    return {
        'experiment': experiment, 'input_data_type': input_data_type,
        'test_acc':        test_acc,
        'test_macro_f1':   test_macro_f1,
        'test_macro_p':    test_macro_p,
        'test_macro_r':    test_macro_r,
        'test_weighted_f1': test_weighted_f1,
        'test_weighted_p':  test_weighted_p,
        'test_weighted_r':  test_weighted_r,
        'best_params': best_params,
        'train_samples': len(train_texts), 'test_samples': len(test_texts),
        'num_features': X_train.shape[1]
    }

    

def main():
    experiments_input = ["full","all","multi"]
    all_results = []
    for i in experiments_input:
        if i == "full":
            train = "/raid/jiawen/proj_master/HF_data/runningdata/train_results_full.jsonl"
            test =   "/raid/jiawen/proj_master/HF_data/runningdata/test_results_full.jsonl"
            result = run(experiment, i, train, test)

        elif i == "all":
            train = "/raid/jiawen/proj_master/HF_data/runningdata/train_results_all.jsonl"
            test =   "/raid/jiawen/proj_master/HF_data/runningdata/test_results_all.jsonl"
            result = run(experiment,i, train, test)

        elif i == "multi":
            train = "/raid/jiawen/proj_master/HF_data/runningdata/train_results_multi.jsonl"
            test =   "/raid/jiawen/proj_master/HF_data/runningdata/test_results_multi.jsonl"
            result = run(experiment, i, train, test)

        all_results.append(result)
        import time
        time.sleep(2)
        
        
        
    comparison_df = pd.DataFrame(all_results)
    print(comparison_df[['input_data_type', 'test_acc',
                          'test_macro_p', 'test_macro_r', 'test_macro_f1',
                          'test_weighted_p', 'test_weighted_r', 'test_weighted_f1']])    
    comparison_df.to_csv(f"comparison_{experiment}.csv", index=False)
    print(f"comparison save as: {experiment}_comparison.csv")
    
    
    if len(all_results) == 3:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(all_results))
        width = 0.35
        
        accuracies = [r['test_acc'] for r in all_results]
        f1_scores = [r['test_macro_f1'] for r in all_results]
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        plt.bar(x + width/2, f1_scores, width, label='Macro F1', color='lightcoral')
        plt.xlim(-0.5, len(all_results) - 0.5)  # fix x 
        plt.ylim(0, 1.0)                         # fix y

        plt.xlabel('Experiment')
        plt.ylabel('Score')
        plt.title('Comparison of Experiments')
        plt.xticks(x, [r['input_data_type'] for r in all_results], rotation=15)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Adding number label
        for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
            plt.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{experiment}_comparison.png", dpi=300, bbox_inches='tight')
        print(f"comparison figure: {experiment}_comparison.png")
        plt.close()
    
    print("\n Finished")

if __name__ == "__main__":
    main()



