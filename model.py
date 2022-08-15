import numpy as np
from transformers import AutoModel

from config import ConfigArgs as args
import torch
import torch.nn as nn
from network import TextEncoder, StyleTokenLayer, AudioDecoder, AttentionLayer
import module as mm


class Empty:
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class QTacotron(nn.Module):
    """
    GST-Tacotron
   
    """

    def __init__(self):
        super(QTacotron, self).__init__()
        self.name = 'QTacotron'
        self.embed = nn.Embedding(119547, args.Ce, padding_idx=0)  # len(tokenizer)
        self.encoder = TextEncoder(hidden_dims=args.Cx)  # bidirectional
        self.GST = StyleTokenLayer(embed_size=args.Cx, n_units=args.Cx)
        self.tpnet = TPSENet(text_dims=args.Cx * 2, style_dims=args.Cx)
        self.decoder = AudioDecoder(enc_dim=args.Cx * 3, dec_dim=args.Cx)
        self.attention = AttentionLayer(embed_size=args.Cx, n_units=args.Cx)

        self.Bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased").eval()
        self.bert_linear = nn.Linear(768, args.Cx)
        self.WordLabelsPredictor = Empty()  # find later

    def forward(self, texts, prev_mels, refs=None, synth=False, ref_mode=True):
        """
        :param texts: (N, Tx) Tensor containing texts
        :param prev_mels: (N, Ty/r, n_mels*r) Tensor containing previous audio
        :param refs: (N, Ty, n_mels) Tensor containing reference audio
        :param synth: Boolean. whether it synthesizes.
        :param ref_mode: Boolean. whether it is reference mode

        Returns:
            :mels_hat: (N, Ty/r, n_mels*r) Tensor. mel spectrogram
            :mags_hat: (N, Ty, n_mags) Tensor. magnitude spectrogram
            :attentions: (N, Ty/r, Tx) Tensor. seq2seq attention
            :style_attentions: (N, n_tokens) Tensor. Style token layer attention
            :ff_hat: (N, Ty/r, 1) Tensor for binary final prediction
            :style_emb: (N, 1, E) Tensor. Style embedding
        """
        if refs is None:
            mel_lengths = [len(mel) for mel in prev_mels]
            refs = torch.zeros(len(prev_mels), max(mel_lengths), 1)
            for idx in range(len(prev_mels)):
                mel_end = mel_lengths[idx]
                refs[idx, mel_end - 1:] = 1.0

        x = self.embed(texts)  # (bsize, Tx, Ce)
        text_emb, enc_hidden = self.encoder(x)  # (bsize, Tx, Cx*2)
        tp_style_emb = self.tpnet(text_emb)
        bert_embeds = self.bert_linear(self.Bert(texts)["pooler_output"]).unsqueeze(1).repeat((1, args.n_tokens, 1))
        # replicated_embeds = self.WordLabelsPredictor(bert_embeds + text_emb)
        replicated_embeds = 0
        if synth:
            style_emb, style_attentions = tp_style_emb + bert_embeds + replicated_embeds, None
        else:
            token_embedding = self.GST(refs)  # (N, 1, E), (N, n_tokens)
            token_embedding = token_embedding + replicated_embeds

            style_emb, style_attentions = self.attention(token_embedding + bert_embeds, refs)

        tiled_style_emb = style_emb.expand(-1, text_emb.size(1), -1)  # (N, Tx, E)
        memory = torch.cat([text_emb, tiled_style_emb], dim=-1)  # (N, Tx, Cx*2+E)
        mels_hat, mags_hat, attentions, ff_hat = self.decoder(prev_mels, memory, synth=synth)
        return mels_hat, mags_hat, attentions, style_attentions, ff_hat, style_emb, tp_style_emb

    def save(self, path, optimizer=None):
        if optimizer is not None:
            torch.save({
                "model_state": self.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, str(path))
        else:
            torch.save({
                "model_state": self.state_dict(),
            }, str(path))

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        # assignment to parameters or buffers is overloaded, updates internal dict entry
        self.step = self.step.data.new_tensor(1)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters


class TPSENet(nn.Module):
    """
    Predict speakers from style embedding (N-way classifier)

    """

    def __init__(self, text_dims, style_dims):
        super(TPSENet, self).__init__()
        self.conv = nn.Sequential(
            mm.Conv1d(text_dims, style_dims, 3, activation_fn=torch.relu, bn=True, bias=False),
            # mm.Conv1d(style_dims, style_dims, 3, activation_fn=torch.relu, bn=True, bias=False)
        )
        self.gru = nn.GRU(style_dims, style_dims, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(style_dims * 2, style_dims)
        # self.net = nn.Linear(args.Cx, args.n_speakers)

    def forward(self, text_embedding):
        """
        :param text_embedding: (N, Tx, E)

        Returns:
            :y_: (N, 1, n_speakers) Tensor.
        """
        te = text_embedding.transpose(1, 2)  # (N, E, Tx)
        h = self.conv(te)
        h = h.transpose(1, 2)  # (N, Tx, C)
        out, _ = self.gru(h)
        se = self.fc(out[:, -1:, :])
        se = torch.tanh(se)
        return se
