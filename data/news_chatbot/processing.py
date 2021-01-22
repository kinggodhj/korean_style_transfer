from pykospacing import spacing
from hanspell import spell_checker

read_file=['negative_train', 'negative_dev', 'negative_test', 'positive_train', 'positive_dev', 'positive_test']
output_file=['negative_train_processed', 'negative_dev_processed', 'negative_test_processed', 'positive_train_processed', 'positive_dev_processed', 'positive_test_processed']

for file in range(len(read_file)):
    sentence=[]
    f=open(read_file[file], 'r', encoding='utf-8')
    lines=f.readlines()
    for line in lines:
        if line.rstrip() != '':
            sentence.append(line.rstrip())
    f.close()
                                     
    wo_space=[]
    for i in range(len(sentence)):
        wo_space.append(sentence[i].replace(" ", ''))
    
    w_space=[]
    for i in range(len(wo_space)):
        w_space.append(spacing(wo_space[i]))

    w_spell=[]
    for i in range(len(w_space)):
        w_spell.append(spell_checker.check(w_space[i]).checked)

    f=open(output_file[file], 'w', encoding='utf-8')
    for i in range(len(w_spell)):
        f.write(w_spell[i])
        f.write('\n')
    f.close()
