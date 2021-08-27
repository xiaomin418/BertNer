#Author: xiaomin
#Create Time: 2021.08.23
# The example of how to use pretrained bert in NER task
import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForTokenClassification,\
    TrainingArguments,Trainer,DataCollatorForTokenClassification

#Load ConLL-2003 dataset
dataset=load_dataset('conll2003')
tokenizer=BertTokenizerFast.from_pretrained('bert-base-cased')
def tokenize_and_align_labels(examples):
    tokenized_inputs=tokenizer(examples['tokens'],truncation=True,is_split_into_words=True)
    labels=[]
    for i,label in enumerate(examples["ner_tags"]):
        print("{}/{}".format(i,len(examples["ner_tags"])))
        word_ids=tokenized_inputs.word_ids(batch_index=1)
        print("labels:{} word_ids: {}".format(label,word_ids))
        previous_word_idx=None
        label_ids=[]
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx!=previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx=word_idx
        labels.append(label_ids)

    tokenized_inputs['labels']=labels
    return tokenized_inputs



#define evaluating indicators.
def compute_metrics(p):
    predictions,labels=p
    predictions=np.argmax(predictions,axis=2)
    true_predictions=[
        [label_list[p] for (p,l) in zip(prediction,label) if l!=-100]
        for prediction,label in zip(predictions,labels)
    ]
    true_labels=[
        [label_list[p] for (p,l) in zip(prediction,label) if l!=-100]
        for prediction,label in zip(predictions,labels)
    ]
    results=metric.compute(predictions=true_predictions,
                          references=true_labels)
    return {
        "precision":results["overall_precision"],
        "recall":results["overall_recall"],
        "f1":results["overall_f1"],
        "accuracy":results["overall_accuracy"]
    }



tokenized_datasets=dataset.map(tokenize_and_align_labels,batched=True,
                               load_from_cache_file=False)

#Get label list, and laod the pretrained model.
label_list=dataset['train'].feature["ner_tags"].feature.names
model=BertTokenizerFast.from_pretrained('bert-base-cased',num_labels=len(label_list))

#define data_collator, and use the seqeval evaluation
data_collator=DataCollatorForTokenClassification(tokenizer)
metric=load_metric('seqeval')

#define training args: TrainingArguments and Trainer.
args=TrainingArguments(
    "tf-conll2003",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3
)
trainer=Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#Begin Training!
trainer.train()
trainer.evaluate()
