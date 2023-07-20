import numpy as np
import evaluate
import json
from datetime import datetime
import os

def compute_metrics(eval_pred):
   load_accuracy = evaluate.load("accuracy")
   load_f1 = evaluate.load("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

def save_metics(metrics_dict, model_name):
    file_path = '../reports/model_metrics.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8' ) as f:
            metrics = json.load(f)
        if model_name not in metrics:
            metrics[model_name] = {}
        metrics[model_name][str(datetime.now())] =  metrics_dict
    else:
        first_entry = {str(datetime.now()): metrics_dict}
        metrics = {model_name: first_entry}

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    return 'metrics saved'