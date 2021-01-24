import sentencepiece as spm
import pdb

model_name='sentiment'
input_file='./data/sentimental/negative_test'
output_file='./data/sentimental/negative_test_id'

input_file_2='./data/sentimental/positive_test'
output_file_2='./data/sentimental/positive_test_id'

sp=spm.SentencePieceProcessor()
sp.Load('%s.model'%model_name)

#print(sp.EncodeAsIds('안녕!  <PAD>'))
#pdb.set_trace()

f=open(input_file, 'r', encoding='utf-8')
lines=f.readlines()

g=open(output_file, 'w')
for line in lines:
    tmp=sp.EncodeAsIds(line)
    for j in range(len(tmp)):
        g.write(str(tmp[j])+' ')
    g.write('4')
    g.write('\n')

f.close()
g.close()

f=open(input_file_2, 'r', encoding='utf-8')
lines=f.readlines()

g=open(output_file_2, 'w')
for line in lines:
    tmp=sp.EncodeAsIds(line)
    for j in range(len(tmp)):
        g.write(str(tmp[j])+' ')
    g.write('4')
    g.write('\n')

f.close()
g.close()
