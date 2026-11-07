"""Stanza(PyTorch) → ONNX 변환과 사전(vocab.json · lemma_dict.json) 내보내기.

stanza 1.15 의 순전파를 한 문장(batch 1) 기준으로 그대로 옮깁니다. 한 문장이면 PackedSequence 와 채움 없는 텐서가 같으므로
전역 몽키패치(torch.zeros · torch.max · pack_padded_sequence 바꿔치기) 없이 각 모듈의 nn.LSTM · Linear · Embedding 을 직접 부릅니다.
결과가 PyTorch 와 같은지는 verify_onnx.py 가 vocab.json 만으로 입력을 만들어 대조합니다.

이전 판(2026-07 변환본)과 달라진 점 (verify_onnx.py 로 잰 차이는 README.md):
- POS: 문자 모델 출력에 trans_char 투영을 거치고(이전에는 원시 은닉 400칸 중 앞 125칸을 잘라 씀), 영어 문자 언어모델(charlm)은
  문장 전체 글자열에서 낱말 끝 자리를 읽습니다(이전에는 낱말마다 따로, 역방향도 정방향 순서로 돌림). 학습된 LSTM 초기 상태(h_init · c_init)를
  씁니다(이전에는 0). 문자 모델의 주의 합에서 채움 칸을 뺍니다. xpos · feats 머리를 함께 내보냅니다.
- 사전: 사전학습 단어 사전(pretrain, 소문자로 찾음)과 문자 언어모델 사전을 넣습니다. 이전에는 없어서 pre 입력이 늘 0 이었습니다.
- 원형: 교사 강제 한 번(tgt_in) 대신 인코더(lemma_encoder.onnx)와 한 걸음 디코더(lemma_decoder.onnx)로 나눠 탐욕 디코딩을 하게 합니다.
  이전 판은 디코더 반복이 추적 시점 길이(6)로 굳어 다른 길이 낱말에서 실행이 실패했습니다. 빈도 사전(lemma_dict.json)과 편집 분류도 함께 씁니다.
- 의존 구문: ROOT 자리, xpos(구성형) 임베딩, linearization · distance 점수, 대각 -inf 를 원본대로 넣습니다.
- 토큰화: 채움 길이 입력(l) 없이 x · f 만 받습니다 (한 문장). 특징 칸 순서(feat_layout)를 사전에 적고, 일본어 · 중국어처럼
  사전 특성(use_dictionary)을 쓰는 모델은 낱말 목록(tokenize_lexicon.json)을 함께 내보냅니다. 이전 판은 특징을 모두 0 으로 넣었습니다.
- MWT: 여러 낱말 토큰(du → de le, zum → zu dem, del → de el …)을 펼치는 처리기를 내보냅니다. 이전 판에는 없었습니다.
  seq2seq 판은 원형과 같은 인코더(mwt_encoder.onnx) · 한 걸음 디코더(mwt_decoder.onnx), 영어는 글자 분류기(mwt_classifier.onnx),
  학습 사전(mwt_dict.json)은 모델보다 먼저 봅니다.

입력 이름은 이전 판과 겹치지 않게 바꿨습니다 (word_ids · pretrain_ids · char_ids …). 이전 판 입력 규약(logis-center stanza.rs)으로 새 파일을 열면
'모르는 입력' 으로 실패해 품사 없이 진행하도록 하기 위해서입니다 (옛 규약으로 잘못된 값을 넣고 돌지 않게).

    python final_definitive_conversion.py                 # 12개 언어 전부, ./onnx_final/<lang>/
    python final_definitive_conversion.py --langs en,ko   # 일부만
"""
import argparse
import json
import os
import subprocess
import sys

for _mod, _pkg in (("torch", "torch"), ("stanza", "stanza"), ("onnx", "onnx")):
    try:
        __import__(_mod)
    except ImportError:
        print(f"📦 {_pkg} 모듈이 없어 자동 설치를 진행합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

import torch
import torch.nn as nn
import torch.nn.functional as F
import stanza
from stanza.models.common.char_model import CHARLM_END, CHARLM_START
from stanza.models.common.vocab import CompositeVocab
from stanza.models.pos.tag_columns import TAG_LINK_EMB
from stanza.models.tokenization.data import PER_CHAR_FEAT_FUNCS, STRUCTURAL_FEATURES

FORMAT = 2
OPSET = 17
# 원형 디코더의 복사 칸 수. 학습 사전에 없는 글자(DeltaVocab 확장)를 낱말 하나에 이만큼까지 복사할 수 있습니다.
LEMMA_OOV_SLOTS = 64
LANGS = ["en", "cs", "ko", "de", "es", "fr", "ja", "pt", "ar", "it", "nl", "zh-hans"]
SPECIAL = {"PAD": 0, "UNK": 1, "EMPTY": 2, "ROOT": 3}


# ---------------------------------------------------------------------------------------------
# 공통 조각: stanza 모듈을 한 문장 기준으로 그대로 부릅니다.
# ---------------------------------------------------------------------------------------------

def log_sigmoid(x):
    """F.logsigmoid 의 안정된 식. ONNX 로는 Log(Sigmoid(x)) 로 나가 x 가 아주 작으면 -inf 가 되고,
    원형 디코더의 복사 확률에서 -inf - (-inf) = NaN 이 됐습니다 (verify_onnx.py 가 '고착된' → '고착+되<PAD>+ㄴ' 으로 찾음)."""
    return torch.minimum(x, torch.zeros_like(x)) - torch.log(1 + torch.exp(-torch.abs(x)))


def softplus(x):
    """F.softplus 의 안정된 식 (큰 x 에서 넘치지 않게)"""
    return torch.maximum(x, torch.zeros_like(x)) + torch.log(1 + torch.exp(-torch.abs(x)))


def logsumexp2(a, b):
    """torch.logsumexp(stack([a, b]), 0). a 가 늘 유한(≥ -1e12)이라 최댓값을 빼는 식이 안전합니다."""
    m = torch.maximum(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m))


def plain_lstm(packed_lstm):
    """PackedLSTM 안의 nn.LSTM. rec_dropout 이 있는 LSTMwRecDropout 은 다른 순전파라 지원하지 않습니다."""
    lstm = packed_lstm.lstm
    if not isinstance(lstm, nn.LSTM):
        raise NotImplementedError(f"nn.LSTM 이 아닌 순환층은 지원하지 않습니다: {type(lstm).__name__}")
    if not lstm.batch_first:
        raise NotImplementedError("batch_first 가 아닌 LSTM 은 지원하지 않습니다")
    return lstm


def highway_lstm(hl, x, h_init, c_init):
    """stanza HighwayLSTM (common/hlstm.py) 한 문장 판. h_init · c_init 는 학습된 (2·층, 1, H) 매개변수입니다."""
    nd = hl.num_directions
    inp = x
    for layer in range(hl.num_layers):
        lstm = plain_lstm(hl.lstm[layer])
        h0 = h_init[layer * nd:(layer + 1) * nd].contiguous()
        c0 = c_init[layer * nd:(layer + 1) * nd].contiguous()
        h, _ = lstm(inp, (h0, c0))
        hw = hl.highway[layer](inp)
        if hl.highway_func is not None:
            hw = hl.highway_func(hw)
        inp = h + torch.sigmoid(hl.gate[layer](inp)) * hw
    return inp


def bilinear(bl, x1, x2):
    """nn.Bilinear 를 MatMul · Mul · ReduceSum 으로. x1 (1, n, I), x2 (1, n, J) → (1, n, O)"""
    o, i, j = bl.weight.shape
    t = torch.matmul(x1, bl.weight.permute(1, 0, 2).reshape(i, o * j))
    t = t.reshape(x1.size(0), x1.size(1), o, j)
    return (t * x2.unsqueeze(2)).sum(-1) + bl.bias


def with_ones(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], -1)


def biaffine_scorer(bs, x1, x2):
    """stanza BiaffineScorer: 양쪽 끝에 1 을 붙인 뒤 Bilinear"""
    return bilinear(bs.W_bilin, with_ones(x1), with_ones(x2))


def pairwise_bilinear(pb, x1, x2):
    """stanza PairwiseBilinear (einsum NLI,IJO->NLJO · NLJO,NMJ->NLMO). x1 (1, N, I), x2 (1, M, J) → (1, N, M, O)"""
    i, j, o = pb.weight.shape
    inter = torch.matmul(x1, pb.weight.reshape(i, j * o)).reshape(x1.size(0), x1.size(1), j, o)
    inter = inter.permute(0, 1, 3, 2).reshape(x1.size(0), x1.size(1) * o, j)
    out = torch.matmul(inter, x2.transpose(1, 2)).reshape(x1.size(0), x1.size(1), o, x2.size(1))
    return out.permute(0, 1, 3, 2) + pb.bias


def deep_biaffine(dbs, x1, x2):
    """stanza DeepBiaffineScorer (쌍마다). 드롭아웃은 평가 때 없습니다."""
    h1 = dbs.hidden_func(dbs.W1(x1))
    h2 = dbs.hidden_func(dbs.W2(x2))
    return pairwise_bilinear(dbs.scorer.W_bilin, with_ones(h1), with_ones(h2))


def char_model_rep(cm, char_ids, char_lens):
    """stanza CharacterModel (common/char_model.py). 낱말마다 글자 LSTM → 주의 가중 합.
    char_ids (n, C) 는 뒤를 0(PAD)으로 채웁니다. 단방향이라 뒤 채움이 앞 칸 출력을 바꾸지 않고, 합에서는 채움 칸을 뺍니다
    (stanza 는 PackedSequence 라 채움 칸이 애초에 없습니다)."""
    if cm.num_dir != 1:
        raise NotImplementedError("양방향 문자 모델은 채움 칸이 역방향 출력을 바꾸므로 지원하지 않습니다")
    if not cm.attn:
        raise NotImplementedError("주의 없는 문자 모델(마지막 상태)은 아직 지원하지 않습니다")
    lstm = plain_lstm(cm.charlstm)
    layers, hidden = cm.args["char_num_layers"], cm.args["char_hidden_dim"]
    n = char_ids.size(0)
    emb = cm.char_emb(char_ids)
    h0 = cm.charlstm_h_init.expand(layers, n, hidden).contiguous()
    c0 = cm.charlstm_c_init.expand(layers, n, hidden).contiguous()
    out, _ = lstm(emb, (h0, c0))
    weights = torch.sigmoid(cm.char_attn(out))
    positions = torch.arange(char_ids.size(1), dtype=torch.long).unsqueeze(0)
    mask = (positions < char_lens.unsqueeze(1)).to(out.dtype).unsqueeze(2)
    return (out * weights * mask).sum(1)


def charlm_rep(lm, ids, offsets):
    """stanza CharacterLanguageModel.build_char_representation 한 문장 판.
    ids (1, L): 문장 글자열 ('\\n' + 낱말1 + ' ' + 낱말2 + ' ' …, 역방향 모델은 낱말 순서와 글자를 모두 뒤집음)
    offsets (n,): 낱말 뒤 ' ' 자리 (역방향은 원래 낱말 순서로 다시 뒤집은 자리)"""
    lstm = plain_lstm(lm.charlstm)
    layers, hidden = lm.args["char_num_layers"], lm.args["char_hidden_dim"]
    emb = lm.char_emb(ids)
    h0 = lm.charlstm_h_init.expand(layers, 1, hidden).contiguous()
    c0 = lm.charlstm_c_init.expand(layers, 1, hidden).contiguous()
    out, _ = lstm(emb, (h0, c0))
    return out[0].index_select(0, offsets)


# ---------------------------------------------------------------------------------------------
# 처리기별 그래프
# ---------------------------------------------------------------------------------------------

class PosGraph(nn.Module):
    """stanza pos/model.py Tagger.forward 의 평가 경로 (한 문장).
    입력: word_ids (1, n), [pretrain_ids (1, n)], [char_ids (n, C), char_lens (n,)] 또는
          [charlm_f_ids (1, Lf), charlm_f_offsets (n,), charlm_b_ids (1, Lb), charlm_b_offsets (n,)]
    출력: upos (1, n, U), 그 밖의 태그 열(xpos · feats) 마다 로짓. 구성형(CompositeVocab)은 칸마다 <열>_<k>."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        if m.args.get("bert_model"):
            raise NotImplementedError("트랜스포머 임베딩을 쓰는 POS 모델은 지원하지 않습니다")
        self.charlm = bool(m.args.get("charlm"))
        self.has_char = bool(m.args["char"] and m.args["char_emb_dim"] > 0)
        names = ["word_ids"]
        if m.args["pretrain"]:
            names.append("pretrain_ids")
        if self.has_char:
            names += ["charlm_f_ids", "charlm_f_offsets", "charlm_b_ids", "charlm_b_offsets"] if self.charlm else ["char_ids", "char_lens"]
        self.input_names = names
        outs = ["upos"]
        for name in m.tag_names[1:]:
            vocab = m.vocab[name]
            outs += [f"{name}_{k}" for k in range(len(vocab))] if isinstance(vocab, CompositeVocab) else [name]
        self.output_names = outs

    def forward(self, *xs):
        m = self.m
        a = dict(zip(self.input_names, xs))
        word_ids = a["word_ids"]
        inputs = []
        if m.args["word_emb_dim"] > 0:
            inputs.append(m.word_emb(word_ids))
        if m.args["pretrain"]:
            inputs.append(m.trans_pretrained(m.pretrained_emb(a["pretrain_ids"])))
        if self.has_char:
            if self.charlm:
                fwd = charlm_rep(m.charmodel_forward, a["charlm_f_ids"], a["charlm_f_offsets"])
                bwd = charlm_rep(m.charmodel_backward, a["charlm_b_ids"], a["charlm_b_offsets"])
                if m.charmodel_forward_transform is not None:
                    fwd = m.charmodel_forward_transform(fwd)
                    bwd = m.charmodel_backward_transform(bwd)
                inputs += [fwd.unsqueeze(0), bwd.unsqueeze(0)]
            else:
                inputs.append(m.trans_char(char_model_rep(m.charmodel, a["char_ids"], a["char_lens"])).unsqueeze(0))
        x = torch.cat(inputs, 2)
        out = highway_lstm(m.taggerlstm, x, m.taggerlstm_h_init, m.taggerlstm_c_init)

        upos_hid = F.relu(m.upos_hid(out))
        upos = m.upos_clf(upos_hid)
        hids = {"upos": upos_hid}
        scores = {"upos": upos}
        results = {}
        for name in m.eval_order:
            hid = upos_hid if m.share_hid else F.relu(m.tag_hid[name](out))
            hids[name] = hid
            vocab = m.vocab[name]
            if m.share_hid:
                clf = lambda c: c(hid)
            else:
                parent = self.parent(name, hids, scores)
                clf = lambda c: biaffine_scorer(c, hid, parent)
            if isinstance(vocab, CompositeVocab):
                results[name] = [clf(m.tag_clf[name][k]) for k in range(len(vocab))]
            else:
                pred = clf(m.tag_clf[name])
                scores[name] = pred
                results[name] = [pred]
        outs = [upos]
        for name in m.tag_names[1:]:
            outs += results[name]
        return tuple(outs)

    def parent(self, name, hids, scores):
        """Tagger.parent_representation 의 평가 경로: 부모 열의 예측 태그 임베딩(기본) 또는 부모 은닉층"""
        m = self.m
        reps = []
        for p in m.column_parents[name]:
            if m.link_mode == TAG_LINK_EMB:
                ids = scores[p].argmax(-1)
                reps.append(m.upos_emb(ids) if p == "upos" else m.tag_emb[p](ids))
            else:
                reps.append(hids[p])
        return reps[0] if len(reps) == 1 else torch.cat(reps, -1)


class LemmaEncoder(nn.Module):
    """stanza Seq2SeqModel.embed + encode + edit_clf (common/seq2seq_model.py).
    입력: src_ids (1, L) = [SOS] 글자… [EOS] (학습 사전에 없는 글자는 V 부터 매긴 번호, 임베딩은 UNK), pos_id (1,)
    출력: h_in (1, L[+1], H), hn (1, H), cn (1, H), edit_logits (1, E)"""

    input_names = ["src_ids", "pos_id"]
    output_names = ["h_in", "hn", "cn", "edit_logits"]

    def __init__(self, m):
        super().__init__()
        if m.contextual_embedding is not None:
            raise NotImplementedError("문맥 임베딩(charlm)을 쓰는 원형 모델은 지원하지 않습니다")
        self.m = m

    def forward(self, src_ids, pos_id):
        m = self.m
        ids = torch.where(src_ids >= m.vocab_size, torch.full_like(src_ids, 1), src_ids)
        enc = m.embedding(ids)
        if m.use_pos:
            enc = torch.cat([m.pos_embedding(pos_id).unsqueeze(1), enc], 1)
        h0 = torch.zeros(m.encoder.num_layers * 2, 1, m.enc_hidden_dim)
        c0 = torch.zeros(m.encoder.num_layers * 2, 1, m.enc_hidden_dim)
        h_in, (hn, cn) = m.encoder(enc, (h0, c0))
        hn = torch.cat((hn[-1], hn[-2]), 1)
        cn = torch.cat((cn[-1], cn[-2]), 1)
        edit = m.edit_clf(hn) if m.edit else torch.zeros(1, 1)
        return h_in, hn, cn, edit


class MwtEncoder(nn.Module):
    """MWT seq2seq (stanza mwt/trainer.py 의 Seq2SeqModel, 품사 · 편집 없음) 인코더.
    입력: src_ids (1, L) = [SOS] 글자… [EOS] (학습 사전에 없는 글자는 V 부터 매긴 번호, 임베딩은 UNK)
    출력: h_in (1, L, H), hn (1, H), cn (1, H). 디코더는 원형과 같은 LemmaDecoder 그래프입니다."""

    input_names = ["src_ids"]
    output_names = ["h_in", "hn", "cn"]

    def __init__(self, m):
        super().__init__()
        if m.use_pos or m.edit or m.contextual_embedding is not None:
            raise NotImplementedError("품사 · 편집 · 문맥 임베딩을 쓰는 MWT 모델은 지원하지 않습니다")
        self.m = m

    def forward(self, src_ids):
        m = self.m
        ids = torch.where(src_ids >= m.vocab_size, torch.full_like(src_ids, 1), src_ids)
        enc = m.embedding(ids)
        h0 = torch.zeros(m.encoder.num_layers * 2, 1, m.enc_hidden_dim)
        c0 = torch.zeros(m.encoder.num_layers * 2, 1, m.enc_hidden_dim)
        h_in, (hn, cn) = m.encoder(enc, (h0, c0))
        return h_in, torch.cat((hn[-1], hn[-2]), 1), torch.cat((cn[-1], cn[-2]), 1)


class MwtClassifierGraph(nn.Module):
    """영어 MWT (stanza mwt/character_classifier.py CharacterClassifier, force_exact_pieces).
    입력: src_ids (1, L) = [SOS] 글자… [EOS]. 출력: logits (1, L, 2) — [나누지 않음, 이 글자 앞에서 나눔]"""

    input_names = ["src_ids"]
    output_names = ["logits"]

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, src_ids):
        m = self.m
        ids = torch.where(src_ids >= m.vocab_size, torch.full_like(src_ids, 1), src_ids)
        h0 = torch.zeros(m.encoder.num_layers * 2, 1, m.enc_hidden_dim)
        c0 = torch.zeros(m.encoder.num_layers * 2, 1, m.enc_hidden_dim)
        enc, _ = m.encoder(m.embedding(ids), (h0, c0))
        return m.output_layer(enc)


class LemmaDecoder(nn.Module):
    """stanza Seq2SeqModel.decode 한 걸음 (LSTMAttention + 복사). 탐욕 디코딩 반복은 실행기가 합니다.
    입력: token (1, 1) 직전 출력 (처음은 SOS, V 이상이면 UNK 임베딩), hn · cn (1, H), h_in (1, L[+1], H), src_ids (1, L)
    출력: log_probs (1, V + 복사 칸), hn_out, cn_out
    복사 칸: 학습 사전에 없는 글자는 V, V+1 … 번호로 넣습니다. 원본은 V 를 max(src)+1 까지 늘리고 늘린 칸에 UNK 점수를 넣는데,
    여기서는 늘 V + LEMMA_OOV_SLOTS 칸으로 둡니다. 복사되지 않은 늘린 칸은 UNK 와 점수가 같아 argmax(첫 최댓값)는 같습니다."""

    input_names = ["token", "hn", "cn", "h_in", "src_ids"]
    output_names = ["log_probs", "hn_out", "cn_out"]

    def __init__(self, m):
        super().__init__()
        self.m = m
        att = m.decoder.attention_layer
        if type(att).__name__ != "SoftDotAttention":
            raise NotImplementedError(f"soft 주의만 지원합니다: {type(att).__name__}")

    def forward(self, token, hn, cn, h_in, src_ids):
        m = self.m
        V = m.vocab_size
        tok = torch.where(token >= V, torch.full_like(token, 1), token)
        dec_in = m.embedding(tok)[:, 0]
        hy, cy = m.decoder.lstm_cell(dec_in, (hn, cn))
        att = m.decoder.attention_layer
        target = att.linear_in(hy).unsqueeze(2)
        attn = torch.bmm(h_in, target).squeeze(2)
        if m.copy:
            log_attn = torch.log_softmax(attn, 1)
            attn_w = torch.exp(log_attn)
        else:
            attn_w = torch.softmax(attn, 1)
        weighted = torch.bmm(attn_w.unsqueeze(1), h_in).squeeze(1)
        h_tilde = torch.tanh(att.linear_out(torch.cat((weighted, hy), 1)))
        log_probs = torch.log_softmax(m.dec2vocab(h_tilde), -1)
        if not m.copy:
            return log_probs, hy, cy
        copy_logit = m.copy_gate(h_tilde)
        la = log_attn[:, 1:] if m.use_pos else log_attn
        la = torch.log_softmax(la, -1)
        lcp = log_sigmoid(copy_logit) + la
        mx = lcp.max(-1, keepdim=True)[0]
        cp = torch.exp(lcp - mx)
        # scatter_add 대신 원-핫 곱: 칸마다 같은 글자 자리의 복사 확률 합 (ScatterElements 없이)
        slots = torch.arange(V + LEMMA_OOV_SLOTS, dtype=torch.long).view(1, 1, -1)
        onehot = (src_ids.unsqueeze(-1) == slots).to(cp.dtype)
        copied = torch.matmul(cp.unsqueeze(1), onehot).squeeze(1)
        zero = copied == 0
        log_copied = (torch.log(copied.masked_fill(zero, 1e-12)) + mx).masked_fill(zero, -1e12)
        log_nocopy = -torch.log(1 + torch.exp(copy_logit))
        ext = torch.cat([log_probs, log_probs[:, 1:2].expand(1, LEMMA_OOV_SLOTS)], -1)
        out = logsumexp2(log_copied, ext + log_nocopy)
        return out, hy, cy


class DepGraph(nn.Module):
    """stanza depparse/model.py GraphParser.embed + forward_scores 평가 경로 (한 문장, 0번은 ROOT).
    입력: word_ids · lemma_ids · upos_ids · pretrain_ids (1, n+1), xpos_ids (1, n+1, P) 또는 (1, n+1),
          문자: char_ids (n+1, C) · char_lens (n+1,) 또는 charlm_* (ROOT 는 '\\n' 낱말)
    출력: unlabeled (1, n+1, n+1) [i, j] = j 가 i 의 머리일 점수 (대각 -inf), deprel (1, n+1, n+1, R)"""

    def __init__(self, m):
        super().__init__()
        self.m = m
        if m.args.get("bert_model"):
            raise NotImplementedError("트랜스포머 임베딩을 쓰는 의존 구문 모델은 지원하지 않습니다")
        if m.args.get("use_arc_embedding"):
            raise NotImplementedError("arc_embedding 판은 지원하지 않습니다")
        self.charlm = bool(m.args.get("charlm"))
        self.has_char = bool(m.args["char"] and m.args["char_emb_dim"] > 0)
        self.composite_xpos = isinstance(m.vocab["xpos"], CompositeVocab)
        names = []
        if m.args["pretrain"]:
            names.append("pretrain_ids")
        if m.args["word_emb_dim"] > 0:
            names += ["word_ids", "lemma_ids"]
        if m.args["tag_emb_dim"] > 0:
            names += ["upos_ids", "xpos_ids"]
        if self.has_char:
            names += ["charlm_f_ids", "charlm_f_offsets", "charlm_b_ids", "charlm_b_offsets"] if self.charlm else ["char_ids", "char_lens"]
        self.input_names = names
        self.output_names = ["unlabeled", "deprel"]

    def forward(self, *xs):
        m = self.m
        a = dict(zip(self.input_names, xs))
        inputs = []
        if m.args["pretrain"]:
            inputs.append(m.trans_pretrained(m.pretrained_emb(a["pretrain_ids"])))
        if m.args["word_emb_dim"] > 0:
            inputs += [m.word_emb(a["word_ids"]), m.lemma_emb(a["lemma_ids"])]
        if m.args["tag_emb_dim"] > 0:
            use_upos, use_xpos = m.args.get("use_upos", True), m.args.get("use_xpos", True)
            pos_emb = m.upos_emb(a["upos_ids"]) if use_upos else 0
            if use_xpos:
                xpos = a["xpos_ids"]
                if self.composite_xpos:
                    for k in range(len(m.vocab["xpos"])):
                        pos_emb = pos_emb + m.xpos_emb[k](xpos[:, :, k])
                else:
                    pos_emb = pos_emb + m.xpos_emb(xpos)
            if use_upos or use_xpos:
                inputs.append(pos_emb)
            if m.args.get("use_ufeats", True):
                # 원본 그대로: feats 임베딩을 만들고도 pos_emb 를 한 번 더 넣습니다 (학습된 모델의 입력 폭이 이 모양)
                inputs.append(pos_emb)
        if self.has_char:
            if self.charlm:
                inputs += [charlm_rep(m.charmodel_forward, a["charlm_f_ids"], a["charlm_f_offsets"]).unsqueeze(0),
                           charlm_rep(m.charmodel_backward, a["charlm_b_ids"], a["charlm_b_offsets"]).unsqueeze(0)]
            else:
                inputs.append(m.trans_char(char_model_rep(m.charmodel, a["char_ids"], a["char_lens"])).unsqueeze(0))
        x = torch.cat(inputs, 2)
        out = highway_lstm(m.parserlstm, x, m.parserlstm_h_init, m.parserlstm_c_init)

        unlabeled = deep_biaffine(m.unlabeled, out, out).squeeze(3)
        deprel = deep_biaffine(m.deprel, out, out)
        n = out.size(1)
        if m.args["linearization"] or m.args["distance"]:
            r = torch.arange(n, dtype=torch.long)
            offset = (r.view(1, 1, -1) - r.view(1, -1, 1)).to(out.dtype)
        if m.args["linearization"]:
            lin = deep_biaffine(m.linearization, out, out).squeeze(3)
            unlabeled = unlabeled + log_sigmoid(lin * torch.sign(offset))
        if m.args["distance"]:
            dist = deep_biaffine(m.distance, out, out).squeeze(3)
            pred = 1 + softplus(dist)
            unlabeled = unlabeled - torch.log((offset.abs() - pred) ** 2 / 2 + 1)
        # torch.eye(bool) 는 EyeLike(bool) 로 나가 onnxruntime 에 구현이 없어 자리 비교로 만듭니다
        r = torch.arange(n, dtype=torch.long)
        diag = (r.view(1, -1, 1) == r.view(1, 1, -1))
        unlabeled = unlabeled.masked_fill(diag, float("-inf"))
        return unlabeled, deprel


class TokGraph(nn.Module):
    """stanza tokenization/model.py Tokenizer.forward 한 문단 (채움 없음).
    입력: x (1, T) 글자 번호, f (1, T, F) 특징 (feat_funcs 순서, 0/1). 출력: output (1, T, 3 또는 5) 로그 확률 [비경계, 토큰, 문장(, MWT …)]"""

    input_names = ["x", "f"]
    output_names = ["output"]

    def __init__(self, m):
        super().__init__()
        if m.charmodel is not None:
            raise NotImplementedError("charlm 을 쓰는 토크나이저는 지원하지 않습니다")
        self.m = m

    def forward(self, x, f):
        m = self.m
        args = m.args
        emb = torch.cat([m.embeddings(x), f], 2)
        conv_input = emb.transpose(1, 2).contiguous() if args["conv_res"] is not None else None
        inp, _ = m.rnn(emb)
        if args["conv_res"] is not None:
            if not args.get("hier_conv_res", False):
                for layer in m.conv_res:
                    inp = inp + layer(conv_input).transpose(1, 2).contiguous()
            else:
                hid = F.relu(torch.cat([layer(conv_input) for layer in m.conv_res], 1))
                inp = inp + m.conv_res2(hid).transpose(1, 2).contiguous()
        tok0 = m.tok_clf(inp)
        sent0 = m.sent_clf(inp)
        mwt0 = m.mwt_clf(inp) if args["use_mwt"] else None
        if args["hierarchical"]:
            inp2 = inp
            if args["hier_invtemp"] > 0:
                inp2 = inp2 * (1 - torch.sigmoid(-tok0 * args["hier_invtemp"]))
            inp2, _ = m.rnn2(inp2)
            tok0 = tok0 + m.tok_clf2(inp2)
            sent0 = sent0 + m.sent_clf2(inp2)
            if args["use_mwt"]:
                mwt0 = mwt0 + m.mwt_clf2(inp2)
        nontok, tok = log_sigmoid(-tok0), log_sigmoid(tok0)
        nonsent, sent = log_sigmoid(-sent0), log_sigmoid(sent0)
        if args["use_mwt"]:
            nonmwt, mwt = log_sigmoid(-mwt0), log_sigmoid(mwt0)
            return torch.cat([nontok, tok + nonsent + nonmwt, tok + sent + nonmwt, tok + nonsent + mwt, tok + sent + mwt], 2)
        return torch.cat([nontok, tok + nonsent, tok + sent], 2)


# ---------------------------------------------------------------------------------------------
# 입력 만들기 (verify_onnx.py 와 실행기가 따르는 규약과 같음)
# ---------------------------------------------------------------------------------------------

def charlm_inputs(words, vocab, forward):
    """build_char_representation 과 같은 글자열 · 자리"""
    seq = words if forward else [w[::-1] for w in reversed(words)]
    chars, offsets = [CHARLM_START], []
    for w in seq:
        chars.extend(w)
        chars.append(CHARLM_END)
        offsets.append(len(chars) - 1)
    if not forward:
        offsets.reverse()
    return torch.tensor([vocab.map(chars)], dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def char_inputs(words, vocab):
    width = max(len(w) for w in words)
    ids = torch.zeros(len(words), width, dtype=torch.long)
    for i, w in enumerate(words):
        if w:
            ids[i, :len(w)] = torch.tensor(vocab.map(list(w)), dtype=torch.long)
    return ids, torch.tensor([len(w) for w in words], dtype=torch.long)


SAMPLE = {"en": ["The", "pumps", "were", "running", "hot", "."], "ko": ["급수펌프의", "토출", "압력이", "높은", "밸브를", "찾아줘"]}


def sample_words(lang):
    return SAMPLE.get(lang, ["a", "bc", "def", "ghij", "kl", "m"])


def pos_example(m, words, pretrain):
    feeds = [torch.tensor([m.vocab["word"].map(words)], dtype=torch.long)]
    if m.args["pretrain"]:
        feeds.append(torch.tensor([pretrain.vocab.map([w.lower() for w in words])], dtype=torch.long))
    if m.args["char"] and m.args["char_emb_dim"] > 0:
        if m.args.get("charlm"):
            f_ids, f_off = charlm_inputs(words, m.charmodel_forward.char_vocab(), True)
            b_ids, b_off = charlm_inputs(words, m.charmodel_backward.char_vocab(), False)
            feeds += [f_ids, f_off, b_ids, b_off]
        else:
            feeds += list(char_inputs(words, m.vocab["char"]))
    return tuple(feeds)


def dep_example(m, words, pretrain):
    from stanza.models.common.vocab import ROOT_ID
    a = {}
    n1 = len(words) + 1
    if m.args["pretrain"]:
        a["pretrain_ids"] = torch.tensor([[ROOT_ID] + pretrain.vocab.map([w.lower() for w in words])], dtype=torch.long)
    if m.args["word_emb_dim"] > 0:
        a["word_ids"] = torch.tensor([[ROOT_ID] + m.vocab["word"].map(words)], dtype=torch.long)
        a["lemma_ids"] = torch.tensor([[ROOT_ID] + m.vocab["lemma"].map([w.lower() for w in words])], dtype=torch.long)
    if m.args["tag_emb_dim"] > 0:
        a["upos_ids"] = torch.full((1, n1), ROOT_ID, dtype=torch.long)
        if isinstance(m.vocab["xpos"], CompositeVocab):
            a["xpos_ids"] = torch.full((1, n1, len(m.vocab["xpos"])), ROOT_ID, dtype=torch.long)
        else:
            a["xpos_ids"] = torch.full((1, n1), ROOT_ID, dtype=torch.long)
    if m.args["char"] and m.args["char_emb_dim"] > 0:
        if m.args.get("charlm"):
            rooted = ["\n"] + list(words)
            f_ids, f_off = charlm_inputs(rooted, m.charmodel_forward.char_vocab(), True)
            b_ids, b_off = charlm_inputs(rooted, m.charmodel_backward.char_vocab(), False)
            a.update(charlm_f_ids=f_ids, charlm_f_offsets=f_off, charlm_b_ids=b_ids, charlm_b_offsets=b_off)
        else:
            ids, lens = char_inputs(list(words), m.vocab["char"])
            root = torch.zeros(1, ids.size(1), dtype=torch.long)
            root[0, 0] = ROOT_ID
            a["char_ids"] = torch.cat([root, ids], 0)
            a["char_lens"] = torch.cat([torch.tensor([1]), lens])
    return a


DYNAMIC = {
    "word_ids": {1: "n"}, "pretrain_ids": {1: "n"}, "lemma_ids": {1: "n"}, "upos_ids": {1: "n"}, "xpos_ids": {1: "n"},
    "char_ids": {0: "n", 1: "c"}, "char_lens": {0: "n"},
    "charlm_f_ids": {1: "lf"}, "charlm_f_offsets": {0: "n"}, "charlm_b_ids": {1: "lb"}, "charlm_b_offsets": {0: "n"},
    "src_ids": {1: "l"}, "h_in": {1: "lp"},
    "x": {1: "t"}, "f": {1: "t"}, "output": {1: "t"}, "logits": {1: "l"},
    "unlabeled": {1: "n", 2: "n"}, "deprel": {1: "n", 2: "n"},
}


def export(module, example, path, input_names, output_names):
    dynamic = {k: v for k, v in DYNAMIC.items() if k in input_names or k in output_names}
    for name in output_names:
        if name == "upos" or name.startswith(("xpos", "feats")):
            dynamic[name] = {1: "n"}
    with torch.no_grad():
        torch.onnx.export(module, example, path, input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic, opset_version=OPSET, do_constant_folding=True, dynamo=False)


# ---------------------------------------------------------------------------------------------
# 사전
# ---------------------------------------------------------------------------------------------

def vocab_units(v):
    """BaseVocab → 번호 순 단위 목록, CompositeVocab → {sep, keyed, parts: [(키, 목록)]}"""
    if isinstance(v, CompositeVocab):
        return {"composite": True, "sep": v.sep, "keyed": bool(v.keyed),
                "parts": [[str(k), list(units)] for k, units in v._id2unit.items()]}
    return list(v._id2unit)


def vocab_ignores(multi):
    """WordVocab.ignore (pos/vocab.py): 이 단위는 입력에서 EMPTY 번호로, EMPTY 번호는 출력에서 '_' 로 바뀝니다.
    xpos 를 학습하지 않은 fr · pt 모델은 xpos 가 늘 EMPTY → '_' 이고, 의존 구문 입력 '_' 는 UNK 가 아니라 EMPTY 입니다."""
    out = {}
    for name in multi._vocabs:
        ig = getattr(multi[name], "ignore", None)
        if ig:
            out[name] = list(ig)
    return out


def package_of(proc):
    path = proc.config.get("model_path") or ""
    return os.path.splitext(os.path.basename(path))[0] if path else None


def convert_lang(lang, out_root, processors):
    out = os.path.join(out_root, lang)
    os.makedirs(out, exist_ok=True)
    print(f"\n🚀 {lang}")
    nlp = None
    # MWT 가 있는 언어(en · de · es · fr · pt · ar · cs · it · nl)는 함께, 없는 언어(ko · ja · zh)는 빼고 엽니다
    for procs in ("tokenize,mwt,pos,lemma,depparse", "tokenize,pos,lemma,depparse"):
        try:
            nlp = stanza.Pipeline(lang, processors=procs, download_method=stanza.DownloadMethod.REUSE_RESOURCES, verbose=False)
            break
        except Exception as e:
            err = e
    if nlp is None:
        print(f"    ❌ 파이프라인을 열 수 없습니다: {err}")
        return False
    words = sample_words(lang)
    vocab = {"format": FORMAT, "lang": lang, "stanza": stanza.__version__, "special": SPECIAL,
             "packages": {k: package_of(p) for k, p in nlp.processors.items()}, "files": {}}
    # 일부 처리기만 다시 만들면 같은 판의 기존 사전에 덮어 씁니다 (나머지 처리기 항목을 지우지 않게)
    old_path = os.path.join(out, "vocab.json")
    if os.path.exists(old_path):
        try:
            old = json.load(open(old_path, encoding="utf-8"))
        except ValueError:
            old = {}
        if old.get("format") == FORMAT and old.get("stanza") == stanza.__version__:
            for key in ("tokenize", "mwt", "pos", "pretrain", "lemma", "depparse"):
                if key in old:
                    vocab[key] = old[key]
            vocab["files"].update(old.get("files", {}))
    ok = True

    def step(name, fn):
        nonlocal ok
        try:
            print(f"  📦 {name}...")
            fn()
            print("    ✅ Done")
        except NotImplementedError as e:
            print(f"    ⚠️ 건너뜀: {e}")
        except Exception as e:
            ok = False
            print(f"    ❌ {type(e).__name__}: {e}")

    if "tokenize" in processors and "tokenize" in nlp.processors:
        def tok():
            tr = nlp.processors["tokenize"].trainer
            m = tr.model.cpu().eval()
            g = TokGraph(m).eval()
            v = tr.vocab
            text = " ".join(words)
            x = torch.tensor([[v.unit2id(c) for c in text]], dtype=torch.long)
            f = torch.zeros(1, x.size(1), int(tr.args["feat_dim"]))
            export(g, (x, f), os.path.join(out, "tokenizer.onnx"), g.input_names, g.output_names)
            n_cls = 5 if m.args["use_mwt"] else 3
            funcs = list(tr.args["feat_funcs"])
            # 특징 칸 순서 (tokenization/data.py para_to_sentences): 글자 특징(feat_funcs 순) → end_of_para → start_of_para
            # → 구조 특징(STRUCTURAL_FEATURES 순) → 사전 특징 앞쪽 num_dict_feat 칸 · 뒤쪽 num_dict_feat 칸
            layout = [x for x in funcs if x in PER_CHAR_FEAT_FUNCS]
            layout += [x for x in ("end_of_para", "start_of_para") if x in funcs]
            layout += [x for x in STRUCTURAL_FEATURES if x in funcs]
            n_dict = int(tr.args.get("num_dict_feat", 0) or 0) if tr.args.get("use_dictionary") else 0
            layout += [f"dict_forward_{i + 1}" for i in range(n_dict)] + [f"dict_backward_{i + 1}" for i in range(n_dict)]
            if len(layout) != int(tr.args["feat_dim"]):
                raise ValueError(f"특징 칸 수가 모델과 다릅니다: {layout} ≠ {tr.args['feat_dim']}")
            entry = {"main": list(v._id2unit), "lower": bool(v.lower), "feat_funcs": funcs, "feat_layout": layout,
                     "feat_dim": int(tr.args["feat_dim"]), "num_dict_feat": n_dict, "lexicon_file": None,
                     "skip_newline": bool(tr.args.get("skip_newline", False)), "replaces_spaces": bool(v.lang_replaces_spaces),
                     "use_regex_tokens": True, "max_seqlen": 1000,
                     "classes": ["none", "token", "sentence", "mwt", "mwt+sentence"][:n_cls],
                     "inputs": g.input_names, "outputs": g.output_names, "lang": tr.args.get("lang")}
            files = ["tokenizer.onnx"]
            if n_dict:
                # create_dictionary 는 두 글자 이상 낱말만 씁니다. 접두 · 접미 집합은 실행기가 이 목록에서 만듭니다.
                lex = sorted(w for w in (tr.lexicon or []) if len(w) > 1)
                with open(os.path.join(out, "tokenize_lexicon.json"), "w", encoding="utf-8") as fh:
                    json.dump(lex, fh, ensure_ascii=False)
                entry["lexicon_file"] = "tokenize_lexicon.json"
                entry["lexicon_size"] = len(lex)
                files.append("tokenize_lexicon.json")
            vocab["tokenize"] = entry
            vocab["files"]["tokenize"] = files
        step("Tokenizer", tok)

    if "mwt" in processors and "mwt" in nlp.processors:
        def mwt():
            proc = nlp.processors["mwt"]
            tr = proc.trainer
            # 학습 사전: 토크나이저가 MWT 로 표시한 토큰을 먼저 여기서 찾습니다 (그대로 · 소문자 + 대문자 복원 · 첫 글자 대문자).
            with open(os.path.join(out, "mwt_dict.json"), "w", encoding="utf-8") as fh:
                json.dump(tr.expansion_dict, fh, ensure_ascii=False)
            entry = {"dict_file": "mwt_dict.json", "dict_size": len(tr.expansion_dict),
                     "dict_only": bool(tr.args.get("dict_only", False)), "ensemble_dict": bool(proc.config.get("ensemble_dict", False)),
                     "type": "dict", "never_decode_unk": True}
            files = ["mwt_dict.json"]
            if not entry["dict_only"] and tr.model is not None:
                m = tr.model.cpu().eval()
                if int(tr.args.get("beam_size", 1)) != 1:
                    raise NotImplementedError("빔 탐색 MWT 는 지원하지 않습니다")
                src = torch.tensor([[2] + tr.vocab.map(list(words[1])) + [3]], dtype=torch.long)
                entry.update({"char": list(tr.vocab._id2unit), "vocab_size": int(m.vocab_size), "sos": 2, "eos": 3, "unk": 1, "pad": 0})
                if tr.args.get("force_exact_pieces", False):
                    g = MwtClassifierGraph(m).eval()
                    export(g, (src,), os.path.join(out, "mwt_classifier.onnx"), g.input_names, g.output_names)
                    entry.update({"type": "classifier", "classifier": {"inputs": g.input_names, "outputs": g.output_names}})
                    files.append("mwt_classifier.onnx")
                else:
                    enc, dec = MwtEncoder(m).eval(), LemmaDecoder(m).eval()
                    export(enc, (src,), os.path.join(out, "mwt_encoder.onnx"), enc.input_names, enc.output_names)
                    with torch.no_grad():
                        h_in, hn, cn = enc(src)
                    export(dec, (torch.tensor([[2]]), hn, cn, h_in, src), os.path.join(out, "mwt_decoder.onnx"), dec.input_names, dec.output_names)
                    entry.update({"type": "seq2seq", "oov_slots": LEMMA_OOV_SLOTS, "max_dec_len": int(m.max_dec_len),
                                  "encoder": {"inputs": enc.input_names, "outputs": enc.output_names},
                                  "decoder": {"inputs": dec.input_names, "outputs": dec.output_names}})
                    files += ["mwt_encoder.onnx", "mwt_decoder.onnx"]
            vocab["mwt"] = entry
            vocab["files"]["mwt"] = files
        step("MWT", mwt)

    pretrain = None
    if "pos" in processors and "pos" in nlp.processors:
        def pos():
            nonlocal pretrain
            proc = nlp.processors["pos"]
            m = proc.trainer.model.cpu().eval()
            pretrain = proc.pretrain
            g = PosGraph(m).eval()
            export(g, pos_example(m, words, pretrain), os.path.join(out, "pos.onnx"), g.input_names, g.output_names)
            entry = {"word": list(m.vocab["word"]._id2unit), "word_lower": bool(m.vocab["word"].lower),
                     "char": list(m.vocab["char"]._id2unit), "upos": list(m.vocab["upos"]._id2unit),
                     "tags": {name: vocab_units(m.vocab[name]) for name in m.tag_names[1:]},
                     "tag_order": list(m.tag_names), "inputs": g.input_names, "outputs": g.output_names,
                     "pretrain": bool(m.args["pretrain"]), "charlm": None, "ignore": vocab_ignores(m.vocab)}
            if g.charlm:
                entry["charlm"] = {"forward": list(m.charmodel_forward.char_vocab()._id2unit),
                                   "backward": list(m.charmodel_backward.char_vocab()._id2unit),
                                   "start": CHARLM_START, "end": CHARLM_END}
            vocab["pos"] = entry
            vocab["files"]["pos"] = ["pos.onnx"]
        step("POS", pos)
    if pretrain is not None:
        # 사전학습 단어 사전은 POS · 의존 구문이 같이 씁니다 (언제나 소문자로 찾음, 공백은 \xa0, 없으면 UNK).
        # 25만 낱말이라 vocab.json 과 따로 둡니다 (pretrain_vocab.json, 번호 순 목록).
        units = list(pretrain.vocab._id2unit)
        with open(os.path.join(out, "pretrain_vocab.json"), "w", encoding="utf-8") as fh:
            json.dump(units, fh, ensure_ascii=False)
        vocab["pretrain"] = {"file": "pretrain_vocab.json", "size": len(units), "lower_lookup": True, "space": "\xa0", "dim": int(pretrain.emb.shape[1])}
        vocab["files"]["pretrain"] = ["pretrain_vocab.json"]

    if "lemma" in processors and "lemma" in nlp.processors:
        def lemma():
            proc = nlp.processors["lemma"]
            tr = proc.trainer
            entry = {"char": list(tr.vocab["char"]._id2unit), "pos": list(tr.vocab["pos"]._id2unit),
                     "caseless": bool(tr.caseless), "dict_only": bool(tr.args.get("dict_only", False)),
                     "ensemble_dict": bool(proc.config.get("ensemble_dict", False)), "use_identity": bool(getattr(proc, "use_identity", False)),
                     "beam_size": int(proc.config.get("beam_size", 1)), "dict_file": "lemma_dict.json",
                     "contextual_lemmatizers": len(getattr(tr, "contextual_lemmatizers", []) or [])}
            with open(os.path.join(out, "lemma_dict.json"), "w", encoding="utf-8") as fh:
                json.dump(tr.pos_dict, fh, ensure_ascii=False)
            files = ["lemma_dict.json"]
            if tr.model is not None and not entry["dict_only"]:
                m = tr.model.cpu().eval()
                enc, dec = LemmaEncoder(m).eval(), LemmaDecoder(m).eval()
                src = torch.tensor([[2] + tr.vocab["char"].map(list(words[1])) + [3]], dtype=torch.long)
                pos_id = torch.tensor([tr.vocab["pos"].unit2id("NOUN")], dtype=torch.long)
                export(enc, (src, pos_id), os.path.join(out, "lemma_encoder.onnx"), enc.input_names, enc.output_names)
                with torch.no_grad():
                    h_in, hn, cn, _ = enc(src, pos_id)
                export(dec, (torch.tensor([[2]]), hn, cn, h_in, src), os.path.join(out, "lemma_decoder.onnx"), dec.input_names, dec.output_names)
                entry.update({"vocab_size": int(m.vocab_size), "oov_slots": LEMMA_OOV_SLOTS, "sos": 2, "eos": 3, "unk": 1, "pad": 0,
                              "max_dec_len": int(m.max_dec_len), "edit": ["none", "identity", "lower"] if m.edit else [],
                              "use_pos": bool(m.use_pos), "copy": bool(m.copy),
                              "encoder": {"inputs": enc.input_names, "outputs": enc.output_names},
                              "decoder": {"inputs": dec.input_names, "outputs": dec.output_names}})
                files += ["lemma_encoder.onnx", "lemma_decoder.onnx"]
            vocab["lemma"] = entry
            vocab["files"]["lemma"] = files
        step("Lemma", lemma)

    if "depparse" in processors and "depparse" in nlp.processors:
        def dep():
            proc = nlp.processors["depparse"]
            m = proc.trainer.model.cpu().eval()
            pt = getattr(proc, "pretrain", None) or pretrain
            g = DepGraph(m).eval()
            ex = dep_example(m, words, pt)
            export(g, tuple(ex[k] for k in g.input_names), os.path.join(out, "depparse.onnx"), g.input_names, g.output_names)
            entry = {"word": list(m.vocab["word"]._id2unit), "word_lower": bool(m.vocab["word"].lower),
                     "lemma": list(m.vocab["lemma"]._id2unit), "lemma_lower": bool(m.vocab["lemma"].lower),
                     "char": list(m.vocab["char"]._id2unit), "upos": list(m.vocab["upos"]._id2unit),
                     "xpos": vocab_units(m.vocab["xpos"]), "deprel": list(m.vocab["deprel"]._id2unit),
                     "inputs": g.input_names, "outputs": g.output_names, "root_word": "\n", "charlm": None,
                     "ignore": vocab_ignores(m.vocab)}
            if g.charlm:
                entry["charlm"] = {"forward": list(m.charmodel_forward.char_vocab()._id2unit),
                                   "backward": list(m.charmodel_backward.char_vocab()._id2unit),
                                   "start": CHARLM_START, "end": CHARLM_END}
            vocab["depparse"] = entry
            vocab["files"]["depparse"] = ["depparse.onnx"]
        step("Depparse", dep)

    path = os.path.join(out, "vocab.json")
    with open(path, "w", encoding="utf-8") as fh:
        # git 에 넣는 파일이라 줄 단위 비교가 되게 들여씁니다 (큰 사전은 따로 두고 git 에 넣지 않음)
        json.dump(vocab, fh, ensure_ascii=False, indent=1)
    print(f"  📦 vocab.json ({os.path.getsize(path) / 1e6:.1f} MB) · 처리기 {[k for k in ('tokenize', 'mwt', 'pos', 'lemma', 'depparse') if k in vocab]}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", default=",".join(LANGS), help="쉼표로 구분한 언어 코드 (기본: 12개 언어)")
    ap.add_argument("--out", default="./onnx_final", help="출력 폴더 (언어마다 하위 폴더)")
    ap.add_argument("--processors", default="tokenize,mwt,pos,lemma,depparse")
    args = ap.parse_args()
    processors = set(args.processors.split(","))
    failed = []
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        try:
            if not convert_lang(lang, args.out, processors):
                failed.append(lang)
        except BaseException as e:
            print(f"⚠️ {lang} failed during conversion: {e}")
            failed.append(lang)
    if failed:
        print(f"\n실패한 언어: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
