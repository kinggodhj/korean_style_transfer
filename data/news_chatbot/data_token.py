import sentencepiece as spm
import pdb

model_name='newchat'
sp=spm.SentencePieceProcessor()
sp.Load('%s.model'%model_name)
input_file=['./news_train', './news_dev', './news_test', './chat_train', './chat_dev', './chat_test']
output_file=['./processed_files/sentiment.train.0', './processed_files/sentiment.dev.0', './processed_files/sentiment.test.0', './processed_files/sentiment.train.1', './processed_files/sentiment.dev.1', './processed_files/sentiment.test.1']

for i in range(len(input_file)):
    f=open(input_file[i], 'r', encoding='utf-8')
    lines=f.readlines()

    g=open(output_file[i], 'w')
    for line in lines:
        tmp=sp.EncodeAsIds(line)
        for j in range(len(tmp)):
            g.write(str(tmp[j])+' ')
        g.write('4')
        g.write('\n')

    f.close()
    g.close()
#
#f=open(input_file_2, 'r', encoding='utf-8')
#lines=f.readlines()
#
#g=open(output_file_2, 'w')
#for line in lines:
#    tmp=sp.EncodeAsIds(line)
#    for j in range(len(tmp)):
#        g.write(str(tmp[j])+' ')
#    g.write('4')
#    g.write('\n')
#
#f.close()
#g.close()
