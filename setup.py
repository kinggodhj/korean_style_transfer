import os
import torch

from data import prepare_data

def read_file(path):
    text=list()
    f=open(path, 'r')
    lines=f.readlines()
    for line in lines:
        text.append(line.rstrip())
    f.close()
    return text


def write_file(text_list, path):
    p=path.split('/')
    di=''
    for i in range(len(p)-1):
        di=di+p[i]+'/'
    if not (os.path.exists(di)):
        os.makedirs(di)

    f=open(p[-1], 'w')
    for i in range(len(text_list)):
        f.write(str(text_list[i])+'\n')
    f.close()
    return


def add_output(ss, path):
    with open(path, 'a') as f:
        f.write(str(ss) + '\n')
    return


def sortInput(input):
    input_lengths=torch.LongTensor([torch.max(input[i, :].data.nonzero())+1 for i in range(input.size(0))])
    input_lengths, sorted_idx=input_lengths.sort(0, descending=True)
    input=input[sorted_idx]
    input=input.unsqueeze(2)
    return input, input_lengths, sorted_idx
   
def preparation(args):
    args.current_save_path='./save/{}'.format(args.name)

    if not (os.path.exists(args.current_save_path)):
        os.makedirs(args.current_save_path)
    print('dir:{}'.format(args.current_save_path))

    args.id_to_word, args.vocab_size, args.train_file_list, args.train_label_list=prepare_data(data_path=args.data_path, max_num=args.word_dict_max_num, task_type='yelp')

    return args


