from models.T5 import TransformerModel
from utils.data_load import data_load
from utils.dataset import Seq2Seq_CustomDataset
from transformers import AutoTokenizer


def training(args):
    if args.task == 'fine_tune':
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        train_src_list, train_trg_list = data_load(dataset_path=args.dataset_path, huggingface_data=False, mode='train')
        valid_src_list, valid_trg_list = data_load(dataset_path=args.dataset_path, huggingface_data=False, mode='tune')
        train_dataset = Seq2Seq_CustomDataset(src_tokenizer = tokenizer, trg_tokenizer= tokenizer, src_list=train_src_list, \
                                                trg_list=train_trg_list)
        valid_dataset = Seq2Seq_CustomDataset(src_tokenizer = tokenizer, trg_tokenizer= tokenizer, src_list=valid_src_list, \
                                                trg_list=valid_src_list)

        
        model = TransformerModel(encoder_model_type=args.model, decoder_model_type=args.model)


        
        pass
    if args.task == 'pretrain_with_taco':
        pass
    

