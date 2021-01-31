# coding: utf-8
import argparse
import math
import numpy as np
import os
import pdb
import time
import torch
from torch import optim
import sentencepiece as spm

from data import get_cuda, pad_batch_sequences, non_pair_data_loader, to_var
from model_norm import Classifier, make_model
from setup import add_output, preparation 

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################################################################
#  Environmental parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--id_pad', type=int, default=0, help='')
parser.add_argument('--id_unk', type=int, default=1, help='')
parser.add_argument('--id_bos', type=int, default=2, help='')
parser.add_argument('--id_eos', type=int, default=3, help='')

######################################################################################
#  File parameters
######################################################################################
parser.add_argument('--task', type=str, default='yelp', help='Specify datasets.')
parser.add_argument('--word_to_id_file', type=str, default='', help='')
parser.add_argument('--data_path', type=str, default='./data/sentimental/processed_files/', help='')
parser.add_argument('--sm_path', type=str, default='./data/sentimental/', help='')
parser.add_argument('--name', type=str, default='sentimental')
parser.add_argument('--beam_size', type=int, default=10)

######################################################################################
#  Model parameters
######################################################################################
parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--max_sequence_length', type=int, default=60)
parser.add_argument('--num_layers_AE', type=int, default=2)
parser.add_argument('--transformer_model_size', type=int, default=256)
parser.add_argument('--transformer_ff_size', type=int, default=1024)

parser.add_argument('--latent_size', type=int, default=256)
parser.add_argument('--word_dropout', type=float, default=1.0)
parser.add_argument('--embedding_dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--label_size', type=int, default=1)

parser.add_argument('--iter', type=str, default=108,)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--weight', type=float, default=1.5)
parser.add_argument('--mode', type=str, default='add')

parser.add_argument('--if_load_from_checkpoint', type=bool)
args = parser.parse_args()

device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
######################################################################################
#  End of hyper parameters
######################################################################################

def piece2text(pieces, sm):
    tmp=[]
    for i in range(len(pieces)):
        if pieces[i]==4:
            break
        else:
            tmp.append(pieces[i])
    return sm.DecodeIds(tmp)

def text2piece(sentence, sm):
    tmp=sm.EncodeAsIds(sentence)
    return [tmp]

def generation(ae_model, sm, test_sentence, label, epoch, args):
    for it in range(len(test_sentence)):
        ####################
        #####load data######
        ####################
        batch_encoder_input, batch_decoder_input, batch_decoder_target, batch_encoder_len, \
        batch_decoder_len=pad_batch_sequences(test_sentence, args.id_bos, args.id_eos, \
        args.id_unk, args.max_sequence_length, args.vocab_size)

        tensor_src=get_cuda(torch.tensor(batch_encoder_input, dtype=torch.long), args.gpu)
        tensor_tgt_y=get_cuda(torch.tensor(batch_decoder_target, dtype=torch.long), args.gpu)
        tensor_src_mask=(tensor_src!=0).unsqueeze(-2)
        tensor_labels=get_cuda(torch.tensor(label, dtype=torch.long), args.gpu)
        
        latent=ae_model.getLatent(tensor_src, tensor_src_mask)
        style, similarity=ae_model.getSim(latent)
        sign=2*(tensor_labels.long())-1
        t_sign=2*(1-tensor_labels.long())-1

        trans_emb=style.clone()[torch.arange(style.size(0)), (1-tensor_labels).long().item()] 
        own_emb=style.clone()[torch.arange(style.size(0)), tensor_labels.long().item()] 
        #batch, dim = 1,256
        w=args.weight
        out_1=ae_model.beam_decode(latent+sign*w*(trans_emb+own_emb), args.beam_size, args.max_sequence_length, args.id_bos)
        add_output(piece2text(out_1[1].tolist(), sm), './generation/{}/test.txt'.format(args.name, epoch, args.beam_size, args.weight))
        
        print("-------------------------------")
        print('original:', sm.DecodeIds(tensor_tgt_y.tolist()[0]))
        print('transferred:', piece2text(out_1[1].tolist(), sm))
        print("-------------------------------")
            

if __name__ == '__main__':
    if not os.path.exists('./generation/{}'.format(args.name)):
        os.makedirs('./generation/{}'.format(args.name))
    
    preparation(args)

    ae_model = get_cuda(make_model(d_vocab=args.vocab_size,
                                   N=args.num_layers_AE,
                                   d_model=args.transformer_model_size,
                                   latent_size=args.latent_size,
                                   gpu=args.gpu, 
                                   d_ff=args.transformer_ff_size), args.gpu)

    iters=args.iter.split(',')

    for idx, i in enumerate(iters):
        ae_model.load_state_dict(torch.load(args.current_save_path + '/{}_ae_model_params.pkl'.format(i), map_location=device))
        sm=spm.SentencePieceProcessor()
        sm.Load(args.sm_path+'%s.model'%args.name)
        print('Type the style of original sentence')
        print('Negative:0 Positive:1')
        label=int(input())
        print('Type the sentence wanted to transferred')
        input_sentence=str(input())
        test_sentence=text2piece(input_sentence, sm)
        generation(ae_model, sm, test_sentence, label, i, args)

    print("Done!")

