from models.T5 import TransformerModel
from utils.data_load import data_load
from utils.optimizer import get_optimizer
from utils.scheduler import get_linear_schedule_with_warmup
from utils.utils import get_logger
from torch import nn
import torch
from torch.utils.data import DataLoader
from utils.dataset import Seq2Seq_CustomDataset
from transformers import AutoTokenizer
from tqdm import tqdm

logger = get_logger("Training")

def training(args):
    if args.task == 'fine_tune':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        train_src_list, train_trg_list = data_load(dataset_path=args.dataset_path, huggingface_data=False, mode='train')
        valid_src_list, valid_trg_list = data_load(dataset_path=args.dataset_path, huggingface_data=False, mode='tune')
        args.num_classes = len(set(train_trg_list))

        train_dataset = Seq2Seq_CustomDataset(src_tokenizer = tokenizer, trg_tokenizer= tokenizer, src_list=train_src_list, \
                                                trg_list=train_trg_list)
        valid_dataset = Seq2Seq_CustomDataset(src_tokenizer = tokenizer, trg_tokenizer= tokenizer, src_list=valid_src_list, \
                                                trg_list=valid_src_list)
        train_DL = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        valid_DL = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        dataloader_dict = {
        'train': train_DL,
        'valid': valid_DL
    }

        logger.info(f'{args.dataset_path} data_load finish')
        
        model = TransformerModel(encoder_model_type=args.model, decoder_model_type=args.model)
        model.to(device)

         # Optimizer setting
        optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay, optim_type=args.optim_type)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset) * args.epochs)
        criterion = nn.CrossEntropyLoss()
        
        Best_loss = 0
    for epoch in range(args.epochs):
        val_loss = 0
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                # write_log(logger, 'Validation start...')
                model.eval()
                logger.info(f"validation...")

            for batch in tqdm(dataloader_dict[phase]):

                # Optimizer gradient setting
                optimizer.zero_grad()

                # Input setting
                src_sequence = batch['src_sequence'].to(device)
                src_attention_mask = batch['src_attention_mask'].to(device)
                trg_sequence = batch['trg_sequence'].to(device)
                # trg_attention_mask = batch['trg_attention_mask'].to(device)

                # Model processing
                with torch.set_grad_enabled(phase == 'train'):
                    encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_attention_mask)
                    decoder_out = model.decode(trg_input_ids=trg_sequence, encoder_hidden_states=encoder_out, encoder_attention_mask=src_attention_mask)

                # Loss back-propagation
                decoder_out = decoder_out.view(-1, decoder_out.size(-1))
                trg_sequence = trg_sequence.view(-1)
                loss = criterion(decoder_out, trg_sequence)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            val_loss /= len(valid_dataset['valid'])
            if val_loss < Best_loss:
                torch.save(model.state_dict(), args.model_path)
                Best_loss = val_loss
        logger.info(f"validation finish! Loss : {val_loss} / best loss : {Best_loss}")


    if args.task == 'pretrain_with_taco':
        pass
    

