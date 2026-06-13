import torch
import torch._dynamo
from torch.utils.data import DataLoader, Dataset
#from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import time

import os
import json
from torch import autograd
import wandb


#Environmental setup
torch._dynamo.config.disable = True
os.environ["TORCHINDUCTOR_DISABLE"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# empty CUDA cache
torch.cuda.empty_cache()
torch._dynamo.config.suppress_errors = True

# Adding arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment', type=str)
args = parser.parse_args()
experiment = args.experiment

# Set the wandb config
WANDB_CONFIG = {
    "lr": 3e-5,
    "epochs": 5,
    "accumulation_steps": 4,
    "max_length": 2800
}


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


#construct data
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


#check the length of the texts 
def check_text_lengths(texts, tokenizer):
    lengths = [len(tokenizer.encode(t)) for t in texts]
    print(f"Texts: {len(lengths)}  |  Avg: {np.mean(lengths):.1f}  |  "
          f"Max: {max(lengths)}  |  Min: {min(lengths)}")
    return lengths

# Load pre-trained BERT model and tokenizer
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_scheduler
model_name = "answerdotai/ModernBERT-base"


def prepare_model(label_number):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=label_number,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    return tokenizer,model


# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):  # can set max_length to an appropriate value
        self.max_length = max_length
        self.texts, self.labels = [], []
        self.tokenizer = tokenizer

        for text, label in zip(texts, labels):
            inputs = self.tokenizer( text, truncation=True, max_length=max_length)

            # Check if the input has more tokens than the max_length
            if len(inputs['input_ids']) <= max_length:
                self.texts.append(inputs)
                self.labels.append(label)
            
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    


## pad to the same length in each batch and transfer to tensor
def build_collate_fn(tokenizer):
    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        inputs, labels = zip(*batch)
        padded = tokenizer.pad(inputs, padding=True, return_tensors='pt')
        return {
            'input_ids': padded['input_ids'],
            'attention_mask': padded['attention_mask'],
            'labels': torch.tensor(labels)
        }
    return collate_fn



from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# calculate the class weights
"""Use sklearn to automatically calculate the weight of each class; 
the fewer the samples, the higher the weight, 
and the more samples, the lower the weight."""

def get_class_weights(train_labels, num_of_labels, device, boost_classes=None):
    classes = np.array(sorted(set(train_labels)))
    weights = compute_class_weight(class_weight='balanced',classes=classes,y=train_labels)

    #Give all classes a default weight of 1.0
    full_weights = np.ones(num_of_labels)
    for cls, w in zip(classes, weights):
        full_weights[cls] = w

    if boost_classes:
        for cls in boost_classes:
            full_weights[cls] *= 1.5

    """Convert it to a PyTorch tensor, use bfloat16 for dtype to keep it consistent with the model, 
    move it to the GPU, and then pass it to CrossEntropyLoss for use."""
    print("Adjusted class weights:", full_weights)
    class_weights = torch.tensor(full_weights, dtype=torch.bfloat16).to(device)
    return class_weights

####Training time!!!
def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, accum_steps, epoch):
    model.train()
    total_loss = 0
    accum_loss = 0  
    all_preds, all_true = [], []

    optimizer.zero_grad() 

    for i, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}')):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=inputs, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels) / accum_steps  # outputs.loss
        loss.backward()

        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(labels.cpu().numpy())

        #keep tracking of each step of loss
        total_loss += loss.item() * accum_steps  #
        accum_loss += loss.item() * accum_steps

        last_in_epoch = (i + 1) == len(dataloader)
        if (i + 1) % accum_steps == 0 or last_in_epoch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps_in_this_accum = (i % accum_steps) + 1  
            wandb.log({"batch_loss": accum_loss / steps_in_this_accum, "lr": scheduler.get_last_lr()[0]})  
            accum_loss = 0
        
    train_report = classification_report(all_true, all_preds, output_dict=True, zero_division=0) 
    return total_loss / len(dataloader), train_report



####Validation time!!!
# Inside the validation loop
def eval_epoch(model, dataloader, loss_fn, device, epoch, num_epochs):
    model.eval()
    val_loss = 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)  # using weighted loss
            val_loss += loss.item()  

    
            # Calculate accuracy
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            
    avg_loss = val_loss / len(dataloader)
    report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
    accuracy = report['accuracy'] 

    print(f'Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_loss:.4f} - Accuracy: {accuracy * 100:.2f}%')
    return avg_loss, report

# Example data
def run(experiment, input_data_type, train_path, test_path):
    print("Experiment starting!!!")

    
    wandb.init(
        entity="hu13260427527-uppsala-universitet",
        project="updated_ModernBert9classe",
        name=f"{experiment}-{input_data_type}",
        config=WANDB_CONFIG)
    
    config = wandb.config
    
    if experiment == "evasion_based_clarity":
        num_labels = 9
        mapping_labels = {'Explicit': 0, 'Implicit': 1, 'Dodging': 2, 'Deflection': 3, 'Partial/half-answer': 4, 'General': 5, 'Declining to answer': 6, 'Claims ignorance': 7, 'Clarification': 8}
        boost_classes = [4, 6, 7]
    elif experiment == "direct_clarity":
        num_labels = 3
        mapping_labels = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}
        boost_classes = None

    using_tokenizer, using_model = prepare_model(num_labels)

    # Load and  split data 
    all_train_texts, all_train_labels = get_data(train_path, input_data_type, experiment, mapping_labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(all_train_texts, all_train_labels, test_size=0.2, random_state=42, stratify=all_train_labels)  
    test_texts, test_labels = get_data(test_path, input_data_type, experiment, mapping_labels)
    assert all(0 <= l < num_labels for l in all_train_labels), "warning: werid label！"

    print (f"The set of labels of train and dev dataset:{set(all_train_labels)}")
    print("Label Distribution:", pd.Series(all_train_labels).value_counts(normalize=True))
    lengths = check_text_lengths(all_train_texts, using_tokenizer)
    print(lengths)


    # Create datasets and dataloaders
    max_length = config.max_length
    collate_fn = build_collate_fn(using_tokenizer)   # fix tokenizer
    train_dataset = CustomDataset(train_texts, train_labels, using_tokenizer, max_length=max_length)
    val_dataset = CustomDataset(val_texts, val_labels, using_tokenizer, max_length=max_length)
    test_dataset = CustomDataset(test_texts, test_labels, using_tokenizer, max_length=max_length)
    print(f"Filtered validation length: {len(val_dataset)}, Filtered train length: {len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    print(f"Train DataLoader: {len(train_dataloader)} ，Dev: {len(val_dataloader)} ")

    # check the first batch
    first_batch = next(iter(train_dataloader))
    print(f"First batch shape:")
    print(f"  input_ids: {first_batch['input_ids'].shape}")
    print(f"  attention_mask: {first_batch['attention_mask'].shape}")
    print(f"  labels: {first_batch['labels'].shape}")
    print(f"  labels value: {first_batch['labels']}")

    # prepare Optimizer / Scheduler
    accum_steps = config.accumulation_steps
    num_epochs = config.epochs
    num_training_steps = num_epochs * len(train_dataloader)
    total_steps = (len(train_dataloader) //accum_steps) * num_epochs


    optimizer = AdamW(using_model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = get_scheduler("linear",
                optimizer=optimizer,
                num_warmup_steps=int(total_steps * 0.06),
                num_training_steps=total_steps)


    """Use DataParallel to copy the model onto 4 cards. 
    Each batch will be automatically split into 4 parts and computed on 4 cards respectively. 
    The gradients are then aggregated onto the main card.
    """
    model = torch.nn.DataParallel(using_model)
    device = torch.device("cuda")
    model = model.to(device)

    # Class weights & loss 
    class_weights = get_class_weights(all_train_labels, num_labels, device, boost_classes=boost_classes)  # recall=0 classes
    loss_fn  = torch.nn.CrossEntropyLoss(weight=class_weights)
    out_file = f"{model_name.split('/')[-1]}-qaevasion-{experiment}-{input_data_type}"
    best_val_loss = float('inf')
    best_val_macro_f1 = 0.0   


    #  Training loop 
    for epoch in range(num_epochs):
        avg_train_loss,train_report  = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            loss_fn, device, accum_steps, epoch,
        )
        avg_val_loss, val_report = eval_epoch(
            model, val_dataloader, loss_fn, device, epoch, num_epochs,
        )

        wandb.log({
            "epoch":             epoch + 1,
            "avg_train_loss":    avg_train_loss,
            "train_accuracy":  train_report["accuracy"],
            "train_macro_f1":  train_report["macro avg"]["f1-score"],
            "train_macro_p":   train_report["macro avg"]["precision"],
            "train_macro_r":   train_report["macro avg"]["recall"],
            # ── val ──
            "val_loss":          avg_val_loss,
            "val_accuracy":      val_report["accuracy"],
            "val_macro_f1":      val_report["macro avg"]["f1-score"],
            "val_macro_p":       val_report["macro avg"]["precision"],
            "val_macro_r":       val_report["macro avg"]["recall"],
            "val_weighted_f1":   val_report["weighted avg"]["f1-score"],
            "val_weighted_p":    val_report["weighted avg"]["precision"],
            "val_weighted_r":    val_report["weighted avg"]["recall"],
        })
    
        current_macro_f1 = val_report['macro avg']['f1-score']
        if current_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = current_macro_f1
            model.module.save_pretrained(f"{out_file}-best")
            using_tokenizer.save_pretrained(f"{out_file}-best")
            print(f"Yeah! The Best model epoch {epoch+1}, val_macro_f1={best_val_macro_f1:.4f}")
        
    # Save the fine-tuned model
    model.module.save_pretrained(out_file)
    using_tokenizer.save_pretrained(out_file)
    print(f"Saved: {out_file}")

    # Test evaluation (load best model) 
    print("Loading best model for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        f"{out_file}-best",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    best_model = torch.nn.DataParallel(best_model).to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
 
    test_loss, test_report = eval_epoch(best_model, test_dataloader, loss_fn, device, epoch=0, num_epochs=1)
    wandb.log({
        "test_loss":          test_loss,
        "test_accuracy":      test_report["accuracy"],
        "test_macro_f1":      test_report["macro avg"]["f1-score"],
        "test_macro_p":       test_report["macro avg"]["precision"],
        "test_macro_r":       test_report["macro avg"]["recall"],
        "test_weighted_f1":   test_report["weighted avg"]["f1-score"],
        "test_weighted_p":    test_report["weighted avg"]["precision"],
        "test_weighted_r":    test_report["weighted avg"]["recall"],
    })
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_report['accuracy'] * 100:.2f}%")  
 
    wandb.finish()
    

    return {"experiment": experiment, "input_type": input_data_type}  




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
        time.sleep(2)
if __name__ == "__main__":
    main()

        
    
