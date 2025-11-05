import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, data, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_length=128):
        self.data = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length

        self.sos_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.long)
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_target_pair = self.data[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_length - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_length - len(dec_input_tokens) - 1

        if enc_input_tokens < 0 or dec_input_tokens < 0:
            raise ValueError("Sentence is too long.")
        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length
         
        return {
            "encoder_input": encoder_input, # Seq_len
            "decoder_input": decoder_input, # Seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)) # (1, Seq_len) & (1, Seq_len, Seq_len)
        }