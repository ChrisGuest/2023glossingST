import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqEncoder, FairseqDecoder
import torch


class SimpleLSTMEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_dim=128, hidden_dim=128, dropout=0.1):
        super().__init__(dictionary)
        self.args = args
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=dictionary.pad())
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)

    def forward(self, src_tokens, src_lengths):
        if self.args.left_pad_source:
            src_tokens = utils.convert_padding_direction(src_tokens, padding_idx=self.dictionary.pad(), left_to_right=True)
        x = self.embed_tokens(src_tokens)
        x = self.dropout(x)
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)
        _outputs, (final_hidden, _final_cell) = self.lstm(x)
        return {'final_hidden': final_hidden.squeeze(0)}

    def reorder_encoder_out(self, encoder_out, new_order):
        final_hidden = encoder_out['final_hidden']
        return {'final_hidden': final_hidden.index_select(0, new_order)}


class SimpleLSTMDecoder(FairseqDecoder):
    def __init__(self, dictionary, encoder_hidden_dim=128, embed_dim=128, hidden_dim=128, dropout=0.1):
        super().__init__(dictionary)
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=dictionary.pad())
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(encoder_hidden_dim + embed_dim, hidden_dim, num_layers=1, bidirectional=False)
        self.output_projection = nn.Linear(hidden_dim, len(dictionary))

    def forward(self, prev_output_tokens, encoder_out):
        bsz, tgt_len = prev_output_tokens.size()
        final_encoder_hidden = encoder_out['final_hidden']
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout(x)
        x = torch.cat([x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)], dim=2)
        initial_state = (final_encoder_hidden.unsqueeze(0), torch.zeros_like(final_encoder_hidden).unsqueeze(0))
        output, _ = self.lstm(x.transpose(0, 1), initial_state)
        x = output.transpose(0, 1)
        x = self.output_projection(x)
        return x, None

# from fairseq.models import FairseqEncoderDecoderModel, register_model

class SimpleLSTMModel(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N')
        parser.add_argument('--encoder-hidden-dim', type=int, metavar='N')
        parser.add_argument('--encoder-dropout', type=float, default=0.1)
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N')
        parser.add_argument('--decoder-hidden-dim', type=int, metavar='N')
        parser.add_argument('--decoder-dropout', type=float, default=0.1)

    @classmethod
    def build_model(cls, args, task):
        encoder = SimpleLSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout,
        )
        decoder = SimpleLSTMDecoder(
            dictionary=task.target_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            dropout=args.decoder_dropout,
        )
        model = SimpleLSTMModel(encoder, decoder)
        print(model)
        return model
