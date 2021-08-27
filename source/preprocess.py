
def convert_to(train_path,new_train_path):

    old_file = open(train_path, 'r')
    new_file = open(new_train_path, 'w')
    new_file.write('tokens,pos,ner,ner_tags\n')

    line = old_file.readline().strip()
    line = old_file.readline().strip()
    while line:
        if len(line)==0:
            new_file.write('\n')
        token = line.split(',')[0]
        ner_tag = line.split(',')[1]
        new_file.write(token + ',' + ner_tag + ',' + ner_tag + ',' + ner_tag + '\n')
        line = old_file.readline().strip()
    old_file.close()
    new_file.close()

def convert_to_train(train_path,new_train_path):

    old_file = open(train_path, 'r')
    new_file = open(new_train_path, 'w')
    new_file.write('data\n')

    line = old_file.readline().strip()
    line = old_file.readline().strip()
    cur_token=[]
    cur_label=[]
    while line:
        if len(line)==1 and line[0]==',':
            curline=" ".join(cur_token)+'\t'+" ".join(cur_label)+'\n'
            new_file.write(curline)
            cur_token = []
            cur_label = []
        token = line.split(',')[0]
        ner_tag = line.split(',')[1]
        
        if len(token)!=0:
            cur_token.append(token)
            cur_label.append(ner_tag)

        line = old_file.readline().strip()
    if len(cur_label)!=0:
        curline = " ".join(cur_token) + '\t' + " ".join(cur_label) + '\n'
        new_file.write(curline)

    old_file.close()
    new_file.close()


def convert_to_test(test_path, new_test_path):
    old_file = open(test_path, 'r')
    new_file = open(new_test_path, 'w')
    new_file.write('data\n')

    line = old_file.readline().strip()
    line = old_file.readline().strip()
    cur_token = []
    cur_label = []
    while line:
        if len(line) == 1 and line[0] == ',':
            curline = " ".join(cur_token) + '\t' + " ".join(cur_label) + '\n'
            new_file.write(curline)
            cur_token = []
            cur_label = []
        token = line.split(',')[1]
        ner_tag = line.split(',')[0]

        if len(token) != 0:
            cur_token.append(token)
            cur_label.append('O')

        line = old_file.readline().strip()
    if len(cur_label) != 0:
        curline = " ".join(cur_token) + '\t' + " ".join(cur_label) + '\n'
        new_file.write(curline)

    old_file.close()
    new_file.close()

def countLabels(train_path):
    old_file = open(train_path, 'r')

    line = old_file.readline().strip()
    line = old_file.readline().strip()
    labels=[]
    while line:
        ner_tag = line.split(',')[1]
        if ner_tag not in labels:
            labels.append(ner_tag)
        line = old_file.readline().strip()
    old_file.close()
    return labels

train_path = './raw_dataset/evaluation_public.csv'
new_train_path = './raw_dataset/test.txt'
convert_to_test(train_path,new_train_path)
# res=countLabels(train_path)
# print(res)