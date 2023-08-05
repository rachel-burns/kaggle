# train and predict (break into two separate modules?)
import data
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
import metrics

def train_doc2vec(df, fname='doc2vec.model',vector_size=5, dm=1, window=2, min_count=1, workers=4, epochs=10):
    tagged_docs = data.get_tagged_docs(df)
    model_doc2vec = Doc2Vec(tagged_docs, vector_size=vector_size, dm=dm, window=window,  min_count=min_count, workers=workers, epochs=epochs)
    model_doc2vec.save('../models/'+fname)
    return model_doc2vec

def preprocess_function(tokenizer, ds):
        return tokenizer(ds["text"], truncation=True)

def train_roberta(train_ds, test_ds, model_name): 
    roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
    roberta_tokenized_train = train_ds.map(lambda x: preprocess_function(roberta_tokenizer,x), batched=True)
    roberta_tokenized_test = test_ds.map(lambda x: preprocess_function(roberta_tokenizer,x), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

    repo_name = f"../models/{model_name}"
 
    training_args = TrainingArguments(
    output_dir=repo_name,
    use_mps_device=True,
    # learning_rate=2e-5,
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    # num_train_epochs=2,
    # weight_decay=0.01,
    # save_strategy="epoch",
    )
    
    trainer = Trainer(
    model=roberta_model,
    args=training_args,
    train_dataset=roberta_tokenized_train,
    eval_dataset=roberta_tokenized_test,
    tokenizer=roberta_tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics.compute_metrics,
    )
    trainer.train()
    metric_output = trainer.evaluate()
    metrics.save_metics(metric_output, model_name)
    return metric_output