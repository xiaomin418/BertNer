import joblib
import numpy as np

label2int={'O': 0, 'B-GPE': 1, 'M-GPE': 2, 'E-GPE': 3, 'B-PER': 4, 'M-PER': 5, 'E-PER': 6, 'B-LOC': 7, 'M-LOC': 8, 'E-LOC': 9, 'B-ORG': 10, 'M-ORG': 11, 'E-ORG': 12, 'S-GPE': 13, 'S-LOC': 14, 'S-PER': 15, 'S-ORG': 16}
int2label={value:key for key,value in label2int.items()}

def compute_result():
    ftest_file = open('./raw_dataset/test.txt', 'r')
    result_file = open('./raw_dataset/result2.txt', 'w')
    ftest_file.readline()
    error = 0
    predictions=joblib.load('predictions.json')
    len_prediction=predictions[0].shape[0]
    for i in range(predictions.shape[0]):
        line = ftest_file.readline()
        cur_tokens = line.split('\t')[0]
        len_cur_tokens=len(cur_tokens.split(' '))
        cur_labels = []
        for j in range(1,len_cur_tokens+1):
            if j<len_prediction:
                lb = np.argmax(predictions[i][j])
                cur_labels.append(int2label[lb])
            else:
                cur_labels.append('O')

        cur_labels = " ".join(cur_labels)

        result_file.write(cur_tokens + "\t" + cur_labels + '\n')

    return error

def compute_submit_with_tokens():
    ftest_file = open('./raw_dataset/test.txt', 'r')
    result_file = open('./raw_dataset/submit.txt', 'w')
    result_file.write('id,tag\n')
    ftest_file.readline()
    error = 0
    predictions=joblib.load('predictions.json')
    len_prediction=predictions[0].shape[0]
    count=0
    for i in range(predictions.shape[0]):
        # import pdb
        # pdb.set_trace()
        line = ftest_file.readline()
        cur_tokens = line.split('\t')[0]
        cur_tokens=cur_tokens.split(' ')
        len_cur_tokens=len(cur_tokens)
        cur_labels = []
        for j in range(1,len_cur_tokens+1):
            if j<len_prediction:
                lb = np.argmax(predictions[i][j])
                cur_labels.append(int2label[lb])
                result_file.write(str(count)+','+cur_tokens[j-1]+','+int2label[lb]+'\n')
                count=count+1
            else:
                cur_labels.append('O')
                result_file.write(str(count) + ',' + cur_tokens[j - 1] + ',' + 'O' + '\n')
                count = count + 1
        result_file.write(',\n')

        cur_labels = " ".join(cur_labels)


    return error

def compute_submit():
    ftest_file = open('./raw_dataset/test.txt', 'r')
    result_file = open('./raw_dataset/submit.txt', 'w')
    result_file.write('id,tag\n')
    ftest_file.readline()
    error = 0
    predictions=joblib.load('predictions.json')
    len_prediction=predictions[0].shape[0]
    count=0
    for i in range(predictions.shape[0]):
        # import pdb
        # pdb.set_trace()
        line = ftest_file.readline()
        cur_tokens = line.split('\t')[0]
        cur_tokens=cur_tokens.split(' ')
        len_cur_tokens=len(cur_tokens)
        cur_labels = []
        for j in range(1,len_cur_tokens+1):
            if j<len_prediction:
                lb = np.argmax(predictions[i][j])
                cur_labels.append(int2label[lb])
                result_file.write(str(count)+','+int2label[lb]+'\n')
                count=count+1
            else:
                cur_labels.append('O')
                result_file.write(str(count) +',' + 'O' + '\n')
                count = count + 1
        result_file.write(',\n')

        cur_labels = " ".join(cur_labels)


    return error

if __name__=='__main__':
    compute_submit()