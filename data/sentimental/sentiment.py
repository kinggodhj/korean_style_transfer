import pdb

read_file=['./ratings_train.txt', './ratings_test.txt']
write_file_1='./positive'
write_file_2='./negative'

review=[]
for i in range(len(read_file)):
    f=open(read_file[i], 'r', encoding='utf-8')
    lines=f.readlines()
    for line in lines:
        review.append(line.rstrip())
    f.close()

negative=[]
positive=[]
for i in range(1, len(review)):
    tmp=review[i].split('\t')
    if tmp[-1]=='0':
        if len(tmp[1])>5:
            negative.append(tmp[1])
    else:
        if len(tmp[1])>5:
            positive.append(tmp[1])

f=open(write_file_1, 'w', encoding='utf-8')
for i in range(len(positive)):
    f.write(positive[i])
    f.write('\n')
f.close()

g=open(write_file_2, 'w', encoding='utf-8')
for i in range(len(negative)):
    g.write(negative[i])
    g.write('\n')
g.close()
