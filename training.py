from model.T5 import TransformerModel
from utils.data_load import data_load
from transformers import AutoTokenizer


def training(args):
    if args.task == 'fine_tune':
        data_load(dataset_path=args.dataset_path, huggingface_data=False, mode='train')
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = TransformerModel(encoder_model_type=args.model, decoder_model_type=args.model)
        print(model)
        pass
    if args.task == 'pretrain_with_taco':
        pass
    

