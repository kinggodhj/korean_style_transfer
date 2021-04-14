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

from data import get_cuda, id2list, non_pair_data_loader, to_var
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
parser.add_argument('--word_to_id', type=str, default='./data/sentimental/processed_files/word_to_id.txt', help='')
parser.add_argument('--data_path', type=str, default='./data/sentimental/processed_files/', help='')
parser.add_argument('--name', type=str, default='norm')
parser.add_argument('--dis_step', type=int)

######################################################################################
#  Model parameters
######################################################################################
parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
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

parser.add_argument('--iter', type=str, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--weight', type=float, default=10)

parser.add_argument('--if_load_from_checkpoint', type=bool)
args = parser.parse_args()

device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
######################################################################################
#  End of hyper parameters
######################################################################################

def generation(ae_model, sm, args):
    eval_data_loader=non_pair_data_loader(
        batch_size=1, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size,
        gpu=args.gpu
    )

    eval_file_list=[
        args.data_path+'sentiment.test.0',
        args.data_path+'sentiment.test.1'
    ]

    eval_label_list=[
        [0],
        [1]
    ]
    eval_data_loader.create_batches(eval_file_list, eval_label_list, if_shuffle=False)
    ae_model.eval()

    sent_dic={0:'chatbot', 1:'bible'}
   
    trans_pos=list()
    dif_pos=list()
    trans_neg=list()
    dif_neg=list() 

    pred=list()
    true=list()
    for it in range(eval_data_loader.num_batch):
        ####################
        #####load data######
        ####################
        batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = eval_data_loader.next_batch()

        latent=ae_model.getLatent(tensor_src, tensor_src_mask)
        style, similarity=ae_model.getSim(latent)
        sign=2*(tensor_labels.long())-1
        t_sign=2*(1-tensor_labels.long())-1
        #style_prob=ae_model.sigmoid(similarity)

        trans_emb=style.clone()[torch.arange(style.size(0)), (1-tensor_labels).long().item()] 
        own_emb=style.clone()[torch.arange(style.size(0)), tensor_labels.long().item()] 
        w=args.weight
        
        out_1=ae_model.greedy_decode(latent+sign*w*(trans_emb+own_emb), args.max_sequence_length, args.id_bos)
        style_1=sm.DecodeIds(id2list(out_1[0].tolist()))
        add_output(style_1, './generation/{}/sign_{}_add.txt'.format(args.name,args.weight))
        out_3=ae_model.greedy_decode(latent+sign*w*(own_emb), args.max_sequence_length, args.id_bos)
        style_3=sm.DecodeIds(id2list(out_3[0].tolist()))
        add_output(style_3, './generation/{}/sign_{}_own.txt'.format(args.name, args.weight))
       
        sent=sent_dic[tensor_labels.item()]
        trans=sent_dic[1-tensor_labels.item()]
        print("------------%d------------" % it)
        print('original {}:'.format(sent), sm.DecodeIds(tensor_tgt_y[0].tolist()))
        print('s:{} w:{} {}:'.format(t_sign.item(), args.weight, trans), style_3)


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

    #iters=args.iter.split(',')
    #for idx, i in enumerate(iters):
    #    ae_model.load_state_dict(torch.load(args.current_save_path + '/{}_ae_model_params.pkl'.format(i), map_location=device))
    #    dis_model.load_state_dict(torch.load(args.current_save_path + '/{}_dis_model_params.pkl'.format(i), map_location=device))
    #    sm=spm.SentencePieceProcessor()
    #    sm.Load('all.model')
    #
    #    generation(ae_model, sm, i, args)
    ae_model.load_state_dict(torch.load('model.pkl', map_location=device))
    sm=spm.SentencePieceProcessor()
    sm.Load('sentimental.model')
    generation(ae_model, sm, args)


    print("Done!")

