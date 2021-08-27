#Author: xiaomin
#Create Time: 2021.08.26
# Defined in Section 7.4.5.2

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
import joblib
# 加载CoNLL-2003数据集、分词器
# dataset = load_dataset('conll2003')
# import pdb
# pdb.set_trace()
# dataset = load_dataset('conll2003', data_files={'txt': ['./raw_dataset/train.txt']})
dataset = load_dataset('csv', data_files={"test":['./raw_dataset/test.txt']})
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')#'bert-base-cased'
label2int={'O': 0, 'B-GPE': 1, 'M-GPE': 2, 'E-GPE': 3, 'B-PER': 4, 'M-PER': 5, 'E-PER': 6, 'B-LOC': 7, 'M-LOC': 8, 'E-LOC': 9, 'B-ORG': 10, 'M-ORG': 11, 'E-ORG': 12, 'S-GPE': 13, 'S-LOC': 14, 'S-PER': 15, 'S-ORG': 16}
int2label={value:key for key,value in label2int.items()}

def tokenize_and_align_labels_onto(examples):
    examples_tokens=[]
    examples_labels=[]
    for onedata in examples["data"]:
        cur_token=onedata.split('\t')[0]
        cur_token=cur_token.split(' ')
        cur_label=onedata.split('\t')[1]
        cur_label=cur_label.split(' ')
        cur_label=[label2int[lb] for lb in cur_label]

        examples_tokens.append(cur_token)
        examples_labels.append(cur_label)


    tokenized_inputs = tokenizer(examples_tokens, truncation=True,  is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 将特殊符号的标签设置为-100，以便在计算损失函数时自动忽略
            if word_idx is None:
                label_ids.append(-100)
            # 把标签设置到每个词的第一个token上
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # 对于每个词的其他token也设置为当前标签
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels_onto, batched=True, load_from_cache_file=False)

# 获取标签列表，并加载预训练模型
# label_list = dataset["train"].features["ner_tags"].feature.names
label_list=['O', 'B-GPE', 'M-GPE', 'E-GPE', 'B-PER', 'M-PER', 'E-PER', 'B-LOC', 'M-LOC', 'E-LOC', 'B-ORG', 'M-ORG', 'E-ORG', 'S-GPE', 'S-LOC', 'S-PER', 'S-ORG']
model = BertForTokenClassification.from_pretrained('./ft-tmp/checkpoint-500', num_labels=len(label_list))

# 定义data_collator，并使用seqeval进行评价
data_collator = DataCollatorForTokenClassification(tokenizer)
# metric = load_metric("seqeval")

# 定义评价指标
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除需要忽略的下标（之前记为-100）
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    a=[]
    b=[]
    for i in true_predictions:
        a=a+i
    true_predictions=[a]#
    for i in true_labels:
        b=b+i
    true_labels=[b]
    results = classification_report(true_predictions, true_labels)
    res_precision=precision_score(true_predictions, true_labels)
    res_recall=recall_score(true_predictions, true_labels)
    res_f1=f1_score(true_predictions, true_labels)
    res_accuracy=accuracy_score(true_predictions, true_labels)
    return {
        "precision": res_precision,
        "recall": res_recall,
        "f1": res_f1,
        "accuracy": res_accuracy,
    }
    # return {
    #     "precision": results["overall_precision"],
    #     "recall": results["overall_recall"],
    #     "f1": results["overall_f1"],
    #     "accuracy": results["overall_accuracy"],
    # }

# 定义训练参数TrainingArguments和Trainer
args = TrainingArguments(
    "ft-tmp",                     # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=8,     # 定义训练批次大小
    per_device_eval_batch_size=8,      # 定义测试批次大小
    num_train_epochs=3,                 # 定义训练轮数
)

trainer = Trainer(
    model,
    args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

predict=trainer.predict(tokenized_datasets["test"])

def compute_result(predict):
    ftest_file = open('./raw_dataset/test.txt', 'r')
    result_file = open('./raw_dataset/result.txt', 'w')
    ftest_file.readline()
    error = 0
    joblib.dump(predict.predictions,'predictions.json')
    for i in range(predict.predictions.shape[0]):
        line = ftest_file.readline()
        cur_tokens = line.split('\t')[0]
        len_cur_tokens=len(cur_tokens.split(' '))
        cur_labels = []
        for j in range(1,len_cur_tokens+1):
            lb=np.argmax(predict.predictions[i][j])
            cur_labels.append(int2label[lb])

        cur_labels = " ".join(cur_labels)

        result_file.write(cur_tokens + "\t" + cur_labels + '\n')

    return error

compute_result(predict)