import sentencepiece as spm

input_file='./data/sentimental/all.txt'
vocab_size=32000
model_name='sentiment'
model_type='bpe'

#user_defined_symbols='[PAD]'

input_argument='--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --pad_id=1 --unk_id=2 --bos_id=3 --eos_id=4'

cmd=input_argument%(input_file, model_name, vocab_size, model_type)

spm.SentencePieceTrainer.Train(cmd)

