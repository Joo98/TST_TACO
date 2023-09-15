from model.T5 import TransformerModel
from utils.data_load import data_load


def training(args):
    if args.task == 'fine_tune':
        data_load(dataset_path=args.dataset_path, huggingface_data=False, mode='test')
        
        pass
    if args.task == 'pretrain_with_taco':
        pass
    

