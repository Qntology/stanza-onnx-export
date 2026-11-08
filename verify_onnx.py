"""변환본(ONNX + vocab.json)을 PyTorch Stanza 와 대조합니다.

세 가지를 잽니다.
 1. 원문 끝에서 끝: vocab.json · 딸린 사전(pretrain_vocab · lemma_dict · mwt_dict · tokenize_lexicon)과 ONNX 만으로
    (RefRuntime, 실행기가 따를 규약) 원문 → 토큰 · 문장 → MWT 펼침 → 품사(upos · xpos · feats) → 원형 → 의존 구문(머리 · 관계)을
    이어 돌리고, 같은 원문을 넣은 stanza.Pipeline(tokenize[,mwt],pos,lemma,depparse) 결과와 문장 단위로 비교합니다.
 2. 처리기별: 파이프라인이 낸 낱말 · upos · xpos · 원형을 넣어 품사 · 원형 · 의존 구문을 따로 잽니다 (앞 단계 오류가 뒤로 번지지 않게).
 3. 그래프: 같은 입력을 넣은 PyTorch 래퍼(final_definitive_conversion 의 *Graph)와 ONNX 출력의 최대 절대 차이.

    python verify_onnx.py --langs en,ko [--onnx ./onnx_final] [--sentences verify_sentences.json] [--json report.json]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")

PAD, UNK, EMPTY, ROOT = 0, 1, 2, 3

# ---- 토큰화 규약 (stanza tokenization/data.py · utils.py 와 같은 식)
NEWLINE_WHITESPACE_RE = re.compile(r"\n[\s\u0080-\u009f]*\n")
NUMERIC_RE = re.compile(r"^[\d]+([,\.]+[\d]+)*[,\.]*$")
WHITESPACE_RE = re.compile(r"[\s\u0080-\u009f]")
SPACE_RE = re.compile(r"\s")
STRUCTURAL = {
    "labeled_field": re.compile(r"\b[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*){0,2}\s*:"),
    "phone_id": re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b|\bx\d{3,5}\b"),
    "date_pattern": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September"
                               r"|October|November|December)\s+\d{1,2},?\s+\d{4}\b"),
    "currency": re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
}
# 이메일 · URL 은 글자 사이를 자르지 않고 마지막 글자를 토큰 끝으로 (utils.py update_pred_regex)
EMAIL_RAW_RE = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""
URL_RAW_RE = r"""(?:https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s"]{2,}|www\.[a-zA-Z0-9]+\.[^\s"]{2,})|[a-zA-Z0-9]+\.(?:gov|org|edu|net|com|co)(?:\.[^\s"]{2,})"""
MASK_RE = re.compile(f"(?:{EMAIL_RAW_RE}|{URL_RAW_RE})")


def dict_expansion(d, word):
    """stanza mwt/trainer.py dict_expansion: 그대로 → 전부 대문자면 소문자로 찾아 대문자로 → 첫 글자만 대문자면 소문자로 찾아 첫 글자 대문자로"""
    hit = d.get(word)
    if hit is not None:
        return hit
    if word.isupper():
        hit = d.get(word.lower())
        if hit is not None:
            return hit.upper()
    if word[0].isupper() and word[1:].islower():
        hit = d.get(word.lower())
        if hit is not None:
            return hit[0].upper() + hit[1:]
    return None


def session(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def index(units):
    return {u: i for i, u in enumerate(units)}


def first_argmax(row):
    """처음 나온 최댓값 자리 (torch.max · numpy.argmax 와 같음)"""
    return int(np.argmax(row))


class Composite:
    """stanza CompositeVocab (xpos · feats) 의 unit2id · id2unit"""

    def __init__(self, spec):
        self.sep = spec["sep"]
        self.keyed = spec["keyed"]
        self.keys = [k for k, _ in spec["parts"]]
        self.units = [u for _, u in spec["parts"]]
        self.ids = [index(u) for u in self.units]

    def parts(self, unit):
        parts = list(unit) if not self.sep else unit.split(self.sep)
        if self.keyed:
            if len(parts) == 1 and parts[0] == "_":
                return {}
            return dict(p.split("=", 1) for p in parts)
        return [] if unit == "_" else parts

    def unit2id(self, unit):
        p = self.parts(unit)
        if self.keyed:
            return [self.ids[i].get(p[k], UNK) if k in p else EMPTY for i, k in enumerate(self.keys)]
        return [self.ids[i].get(p[i], UNK) if i < len(p) else EMPTY for i in range(len(self.keys))]

    def id2unit(self, ids):
        items = []
        for i, v in enumerate(ids):
            if v == EMPTY:
                continue
            items.append(f"{self.keys[i]}={self.units[i][v]}" if self.keyed else self.units[i][v])
        res = self.sep.join(items)
        return res if res else "_"


class RefRuntime:
    """vocab.json 과 ONNX 만 쓰는 참고 실행기. Rust 실행기는 이 규약을 그대로 따릅니다."""

    def __init__(self, folder):
        self.dir = folder
        self.v = json.load(open(os.path.join(folder, "vocab.json"), encoding="utf-8"))
        assert self.v.get("format") == 2, "format 2 vocab.json 이 아닙니다 (이전 변환본)"
        self.pretrain = None
        if "pretrain" in self.v:
            self.pretrain = index(json.load(open(os.path.join(folder, self.v["pretrain"]["file"]), encoding="utf-8")))
        self.s = {}

    def sess(self, name):
        if name not in self.s:
            self.s[name] = session(os.path.join(self.dir, name))
        return self.s[name]

    # ---- 공통 입력
    def pretrain_ids(self, words):
        return [self.pretrain.get(w.lower().replace(" ", "\xa0"), UNK) for w in words]

    @staticmethod
    def char_feed(words, units):
        ids = index(units)
        width = max(1, max(len(w) for w in words))
        out = np.zeros((len(words), width), dtype=np.int64)
        for i, w in enumerate(words):
            for j, c in enumerate(w):
                out[i, j] = ids.get(c, UNK)
        return out, np.array([len(w) for w in words], dtype=np.int64)

    @staticmethod
    def charlm_feed(words, units, start, end, forward):
        ids = index(units)
        seq = words if forward else [w[::-1] for w in reversed(words)]
        chars, offsets = [start], []
        for w in seq:
            chars.extend(w)
            chars.append(end)
            offsets.append(len(chars) - 1)
        if not forward:
            offsets.reverse()
        return np.array([[ids.get(c, UNK) for c in chars]], dtype=np.int64), np.array(offsets, dtype=np.int64)

    def char_inputs(self, words, entry, char_units):
        feed = {}
        if entry.get("charlm"):
            lm = entry["charlm"]
            feed["charlm_f_ids"], feed["charlm_f_offsets"] = self.charlm_feed(words, lm["forward"], lm["start"], lm["end"], True)
            feed["charlm_b_ids"], feed["charlm_b_offsets"] = self.charlm_feed(words, lm["backward"], lm["start"], lm["end"], False)
        else:
            feed["char_ids"], feed["char_lens"] = self.char_feed(words, char_units)
        return feed

    # ---- 품사
    def pos(self, words):
        e = self.v["pos"]
        word_idx = index(e["word"])
        feed = {"word_ids": np.array([[word_idx.get(w.lower() if e["word_lower"] else w, UNK) for w in words]], dtype=np.int64)}
        if "pretrain_ids" in e["inputs"]:
            feed["pretrain_ids"] = np.array([self.pretrain_ids(words)], dtype=np.int64)
        feed.update(self.char_inputs(words, e, e["char"]))
        feed = {k: feed[k] for k in e["inputs"]}
        outs = dict(zip(e["outputs"], self.sess("pos.onnx").run(None, feed)))
        n = len(words)
        upos = [e["upos"][first_argmax(outs["upos"][0, i])] for i in range(n)]
        cols = {"upos": upos}
        for name in e["tag_order"][1:]:
            spec = e["tags"][name]
            if isinstance(spec, dict):
                comp = Composite(spec)
                cols[name] = [comp.id2unit([first_argmax(outs[f"{name}_{k}"][0, i]) for k in range(len(comp.keys))]) for i in range(n)]
            else:
                # WordVocab.ignore 가 있으면 EMPTY 번호는 '_' (fr · pt 처럼 xpos 를 학습하지 않은 모델)
                ignore = e.get("ignore", {}).get(name)
                ids = [first_argmax(outs[name][0, i]) for i in range(n)]
                cols[name] = ["_" if ignore and k == EMPTY else spec[k] for k in ids]
        return cols, outs

    # ---- 원형
    def lemma_dict(self):
        if not hasattr(self, "_ldict"):
            path = os.path.join(self.dir, self.v["lemma"]["dict_file"])
            self._ldict = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        return self._ldict

    def lookup(self, w, pos):
        d = self.lemma_dict()
        hit = d.get(pos, {}).get(w)
        if hit is not None:
            return hit
        return d.get("*", {}).get(w)

    def seq2seq(self, word, upos):
        e = self.v["lemma"]
        char_idx = index(e["char"])
        V, K = e["vocab_size"], e["oov_slots"]
        delta = sorted({c for c in word if c not in char_idx})[:K]
        delta_idx = {c: V + i for i, c in enumerate(delta)}
        src = [e["sos"]] + [char_idx.get(c, delta_idx.get(c, UNK)) for c in word] + [e["eos"]]
        src = np.array([src], dtype=np.int64)
        pos_id = np.array([index(e["pos"]).get(upos, UNK)], dtype=np.int64)
        h_in, hn, cn, edit = self.sess("lemma_encoder.onnx").run(None, {"src_ids": src, "pos_id": pos_id})
        dec = self.sess("lemma_decoder.onnx")
        token = np.array([[e["sos"]]], dtype=np.int64)
        out = []
        for _ in range(e["max_dec_len"]):
            lp, hn, cn = dec.run(None, {"token": token, "hn": hn, "cn": cn, "h_in": h_in, "src_ids": src})
            t = first_argmax(lp[0])
            if t == e["eos"]:
                break
            out.append(t)
            token = np.array([[t]], dtype=np.int64)
        chars = [e["char"][t] if t < V else delta[t - V] if t - V < len(delta) else "<UNK>" for t in out]
        pred = "".join(chars)
        if e["edit"]:
            ed = first_argmax(edit[0])
            pred = word if ed == 1 else word.lower() if ed == 2 else pred
        if len(pred) == 0 or "<UNK>" in pred:
            pred = word
        return pred

    def lemma(self, words, upos):
        e = self.v["lemma"]
        out = []
        for w, p in zip(words, upos):
            if e["use_identity"]:
                out.append(w)
                continue
            key = w.lower() if e["caseless"] else w
            hit = self.lookup(key, p)
            if hit is None and not e["dict_only"]:
                hit = self.seq2seq(key, p)
            out.append(hit if hit else (w if hit is None else "_"))
        return [x if x else "_" for x in out]

    # ---- 의존 구문
    def depparse(self, words, upos, xpos, lemmas):
        e = self.v["depparse"]
        n1 = len(words) + 1
        feed = {}
        if "pretrain_ids" in e["inputs"]:
            feed["pretrain_ids"] = np.array([[ROOT] + self.pretrain_ids(words)], dtype=np.int64)
        if "word_ids" in e["inputs"]:
            wi, li = index(e["word"]), index(e["lemma"])
            feed["word_ids"] = np.array([[ROOT] + [wi.get(w.lower() if e["word_lower"] else w, UNK) for w in words]], dtype=np.int64)
            feed["lemma_ids"] = np.array([[ROOT] + [li.get(l.lower() if e["lemma_lower"] else l, UNK) for l in lemmas]], dtype=np.int64)
        if "upos_ids" in e["inputs"]:
            ui = index(e["upos"])
            feed["upos_ids"] = np.array([[ROOT] + [ui.get(u, UNK) for u in upos]], dtype=np.int64)
            if isinstance(e["xpos"], dict):
                comp = Composite(e["xpos"])
                rows = [[ROOT] * len(comp.keys)] + [comp.unit2id(x) for x in xpos]
                feed["xpos_ids"] = np.array([rows], dtype=np.int64)
            else:
                xi = index(e["xpos"])
                ignore = set(e.get("ignore", {}).get("xpos", []))
                feed["xpos_ids"] = np.array([[ROOT] + [EMPTY if x in ignore else xi.get(x, UNK) for x in xpos]], dtype=np.int64)
        if e.get("charlm"):
            feed.update(self.char_inputs(["\n"] + list(words), e, e["char"]))
        else:
            ids, lens = self.char_feed(list(words), e["char"])
            root = np.zeros((1, ids.shape[1]), dtype=np.int64)
            root[0, 0] = ROOT
            feed["char_ids"] = np.concatenate([root, ids], 0)
            feed["char_lens"] = np.concatenate([[1], lens]).astype(np.int64)
        feed = {k: feed[k] for k in e["inputs"]}
        unlabeled, deprel = self.sess("depparse.onnx").run(None, feed)
        return unlabeled[0], deprel[0]

    # ---- 토큰화 (원문 → 문단 → 글자 특징 → tokenizer.onnx → 토큰 · 문장)
    def paragraphs(self, text):
        """TokenizationDataset.__init__: 빈 줄로 문단을 나누고 오른쪽 공백을 지운 뒤, 공백류는 ' ' 하나로, 연속 공백은 하나만"""
        e = self.v["tokenize"]
        out = []
        for pt in NEWLINE_WHITESPACE_RE.split(text):
            pt = pt.rstrip()
            if not pt:
                continue
            chars = [WHITESPACE_RE.sub(" ", c) for c in pt if not (e["skip_newline"] and c == "\n")]
            out.append([c for i, c in enumerate(chars) if not (i > 0 and c == " " and chars[i - 1] == " ")])
        return out

    def lexicon(self):
        """create_dictionary: 낱말 · 접두(마지막 글자 뺀) · 접미(첫 글자 뺀) 집합"""
        if not hasattr(self, "_lex"):
            e = self.v["tokenize"]
            words = json.load(open(os.path.join(self.dir, e["lexicon_file"]), encoding="utf-8")) if e.get("lexicon_file") else []
            prefixes, suffixes = set(), set()
            for w in words:
                for i in range(1, len(w)):
                    prefixes.add(w[:i])
                    suffixes.add(w[len(w) - i:])
            self._lex = (set(words), prefixes, suffixes)
        return self._lex

    def dict_feats(self, para, idx, n):
        """TokenizationDataset.extract_dict_feat: 첫 글자는 대소문자 그대로, 붙이는 글자는 소문자"""
        words, prefixes, suffixes = self.lexicon()
        fwd, bwd = [0] * n, [0] * n
        fw = bw = para[idx]
        prefix = suffix = True
        for win in range(1, n + 1):
            if idx + win <= len(para) - 1 and prefix:
                fw += para[idx + win].lower()
                fwd[win - 1] = 1 if fw in words else 0
                if fw not in prefixes:
                    prefix = False
            if idx - win >= 0 and suffix:
                bw = para[idx - win].lower() + bw
                bwd[win - 1] = 1 if bw in words else 0
                if bw not in suffixes:
                    suffix = False
            if not prefix and not suffix:
                break
        return fwd + bwd

    def tok_features(self, para):
        e = self.v["tokenize"]
        layout = e["feat_layout"]
        n = len(para)
        raw = "".join(para)
        masks = {}
        for name in layout:
            if name in STRUCTURAL:
                mask = [0] * n
                for m in STRUCTURAL[name].finditer(raw):
                    for p in range(m.start(), m.end()):
                        mask[p] = 1
                masks[name] = mask
        nd = e["num_dict_feat"]
        f = np.zeros((1, n + 1, len(layout)), dtype=np.float32)
        for i, c in enumerate(para):
            dfe = self.dict_feats(para, i, nd) if nd else None
            for k, name in enumerate(layout):
                if name == "space_before":
                    v = c.startswith(" ")
                elif name == "capitalized":
                    v = c[0].isupper()
                elif name == "numeric":
                    v = NUMERIC_RE.match(c) is not None
                elif name == "end_of_para":
                    v = i == n - 1
                elif name == "start_of_para":
                    v = i == 0
                elif name in masks:
                    v = masks[name][i]
                elif name.startswith("dict_forward_"):
                    v = dfe[int(name.rsplit("_", 1)[1]) - 1]
                elif name.startswith("dict_backward_"):
                    v = dfe[nd + int(name.rsplit("_", 1)[1]) - 1]
                else:
                    raise ValueError(f"모르는 특징: {name}")
                f[0, i, k] = float(v)
        return f

    def tokenize(self, text):
        """[[ (토큰, 분류 1~4) ]] 문장 목록. 분류 3 · 4 는 MWT 표시 (utils.py predict · decode_predictions)"""
        e = self.v["tokenize"]
        ids = index(e["main"])
        sents = []
        for para in self.paragraphs(text):
            n = len(para)
            if n + 1 > e["max_seqlen"]:
                raise NotImplementedError("max_seqlen 을 넘는 문단은 나눠 넣어야 합니다")
            x = np.array([[ids.get(c.lower() if e["lower"] else c, UNK) for c in para] + [PAD]], dtype=np.int64)
            out = self.sess("tokenizer.onnx").run(None, {"x": x, "f": self.tok_features(para)})[0][0]
            pred = [first_argmax(out[i]) for i in range(n)]
            pred[-1] = 2 if pred[-1] < 2 else 4 if pred[-1] > 2 else 2
            if e["use_regex_tokens"]:
                text_p = "".join(para)
                for m in MASK_RE.finditer(text_p):
                    for p in range(m.start(), m.end() - 1):
                        pred[p] = 0
                    if pred[m.end() - 1] == 0:
                        pred[m.end() - 1] = 1
            cur, sent = "", []
            for c, p in zip(para, pred):
                cur += c
                if p >= 1:
                    tok = SPACE_RE.sub(" ", cur.lstrip())
                    if e["replaces_spaces"]:
                        tok = tok.replace(" ", "")
                    cur = ""
                    if not tok:
                        continue
                    sent.append((tok, p))
                    if p in (2, 4):
                        sents.append(sent)
                        sent = []
            if sent:
                sents.append(sent)
        return sents

    # ---- MWT (토크나이저가 3 · 4 로 표시한 토큰만)
    def mwt_dict(self):
        if not hasattr(self, "_mdict"):
            self._mdict = json.load(open(os.path.join(self.dir, self.v["mwt"]["dict_file"]), encoding="utf-8"))
        return self._mdict

    def mwt_src(self, tok):
        e = self.v["mwt"]
        char_idx = index(e["char"])
        V, K = e["vocab_size"], e.get("oov_slots", 0)
        delta = sorted({c for c in tok if c not in char_idx})
        delta_idx = {c: V + i for i, c in enumerate(delta) if i < K} if K else {}
        src = [e["sos"]] + [char_idx.get(c, delta_idx.get(c, UNK)) for c in tok] + [e["eos"]]
        return np.array([src], dtype=np.int64), delta

    def mwt_model(self, tok):
        e = self.v["mwt"]
        src, delta = self.mwt_src(tok)
        if e["type"] == "classifier":
            logits = self.sess("mwt_classifier.onnx").run(None, {"src_ids": src})[0][0]
            pieces = []
            for i, c in enumerate(tok):
                if logits[i + 1, 1] > logits[i + 1, 0]:
                    pieces.append(" ")
                pieces.append(c)
            return "".join(pieces).strip()
        V = e["vocab_size"]
        h_in, hn, cn = self.sess("mwt_encoder.onnx").run(None, {"src_ids": src})
        dec = self.sess("mwt_decoder.onnx")
        token = np.array([[e["sos"]]], dtype=np.int64)
        out = []
        for _ in range(e["max_dec_len"]):
            lp, hn, cn = dec.run(None, {"token": token, "hn": hn, "cn": cn, "h_in": h_in, "src_ids": src})
            row = lp[0].copy()
            if e["never_decode_unk"]:
                row[UNK] = -np.inf
            t = first_argmax(row)
            if t == e["eos"]:
                break
            out.append(t)
            token = np.array([[t]], dtype=np.int64)
        chars = [e["char"][t] if t < V else delta[t - V] for t in out if t >= 4 and (t < V or t - V < len(delta))]
        pred = "".join(chars)
        # 펼침이 비었거나 공백이 없으면 원문 토큰 (mwt/trainer.py predict)
        return pred if pred.strip() and " " in pred else tok

    def mwt_expand(self, tok):
        e = self.v["mwt"]
        if e["dict_only"] or e["ensemble_dict"]:
            hit = dict_expansion(self.mwt_dict(), tok)
            if hit is not None:
                return hit
            if e["dict_only"]:
                return tok
        return self.mwt_model(tok) if e["type"] != "dict" else tok

    def words_of(self, sent):
        """[(토큰, [낱말…])] — 펼침이 한 낱말이면 원문 토큰 그대로 (doc.py set_mwt_expansions)"""
        out = []
        for tok, p in sent:
            if p in (3, 4) and "mwt" in self.v:
                parts = [x for x in self.mwt_expand(tok).split(" ") if x]
                out.append((tok, parts if len(parts) > 1 else [tok]))
            else:
                out.append((tok, [tok]))
        return out

    def analyze(self, text):
        """원문 → 문장마다 {tokens, words, upos, xpos, feats, lemma, head, deprel}"""
        res = []
        for sent in self.tokenize(text):
            toks = self.words_of(sent)
            words = [w for _, ws in toks for w in ws]
            cols, _ = self.pos(words)
            xpos = cols.get("xpos", ["_"] * len(words))
            feats = cols.get("feats", ["_"] * len(words))
            lemmas = self.lemma(words, cols["upos"])
            un, rel = self.depparse(words, cols["upos"], xpos, lemmas)
            head, deprel = decode_tree(un, rel, self.v["depparse"]["deprel"])
            res.append({"tokens": [t for t, _ in toks], "words": words, "upos": cols["upos"], "xpos": xpos, "feats": feats,
                        "lemma": lemmas, "head": head, "deprel": deprel})
        return res


def log_softmax(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def decode_tree(unlabeled, deprel, deprel_units):
    """stanza GraphParser.decode_graph_predictions (resolve_head_constraints 포함) 를 ONNX 점수에 그대로 씁니다."""
    from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
    from stanza.models.depparse.model import GraphParser
    from stanza.models.pos.vocab import WordVocab
    vocab = WordVocab()
    vocab._id2unit = deprel_units
    vocab._unit2id = index(deprel_units)
    vocab.lower = False
    vocab.ignore = []
    scores = log_softmax(unlabeled, 1)
    preds = [scores[None], (deprel.argmax(-1) + 4)[None], log_softmax(deprel, 2)[None]]
    out = GraphParser.decode_graph_predictions(vocab, 1, preds, [unlabeled.shape[0]], resolve_head_constraints=True)[0]
    return [h for h, _ in out], [r for _, r in out]


def open_pipeline(lang, pretokenized=False):
    """MWT 가 있는 언어는 함께 엽니다 (변환기와 같은 순서)"""
    import stanza
    for procs in ("tokenize,mwt,pos,lemma,depparse", "tokenize,pos,lemma,depparse"):
        try:
            return stanza.Pipeline(lang, processors=procs, tokenize_pretokenized=pretokenized, download_method=None, verbose=False)
        except Exception as e:
            err = e
    raise err


def graph_diffs(lang, folder, sample, nlp, rt):
    """같은 입력에서 PyTorch 래퍼와 ONNX 출력의 최대 절대 차이"""
    import torch
    import final_definitive_conversion as conv
    out = {}
    with torch.no_grad():
        p = nlp.processors["pos"]
        g = conv.PosGraph(p.trainer.model.cpu().eval()).eval()
        ex = conv.pos_example(g.m, sample, p.pretrain)
        ref = g(*ex)
        got = session(os.path.join(folder, "pos.onnx")).run(None, {k: t.numpy() for k, t in zip(g.input_names, ex)})
        out["pos"] = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(ref, got))

        tr = nlp.processors["lemma"].trainer
        if tr.model is not None:
            enc, dec = conv.LemmaEncoder(tr.model.cpu().eval()).eval(), conv.LemmaDecoder(tr.model.cpu().eval()).eval()
            src = torch.tensor([[2] + tr.vocab["char"].map(list(sample[0])) + [3]])
            pos_id = torch.tensor([tr.vocab["pos"].unit2id("NOUN")])
            r_enc = enc(src, pos_id)
            g_enc = session(os.path.join(folder, "lemma_encoder.onnx")).run(None, {"src_ids": src.numpy(), "pos_id": pos_id.numpy()})
            diff = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(r_enc, g_enc))
            tok = torch.tensor([[2]])
            r_dec = dec(tok, r_enc[1], r_enc[2], r_enc[0], src)
            g_dec = session(os.path.join(folder, "lemma_decoder.onnx")).run(None, {"token": tok.numpy(), "hn": r_enc[1].numpy(), "cn": r_enc[2].numpy(), "h_in": r_enc[0].numpy(), "src_ids": src.numpy()})
            diff = max(diff, max(float(np.abs(r.numpy() - o).max()) for r, o in zip(r_dec, g_dec)))
            out["lemma"] = diff

        d = nlp.processors["depparse"]
        g = conv.DepGraph(d.trainer.model.cpu().eval()).eval()
        ex = conv.dep_example(g.m, sample, getattr(d, "pretrain", None) or p.pretrain)
        ref = g(*[ex[k] for k in g.input_names])
        got = session(os.path.join(folder, "depparse.onnx")).run(None, {k: ex[k].numpy() for k in g.input_names})
        finite = lambda a: np.where(np.isfinite(a), a, 0.0)
        out["depparse"] = max(float(np.abs(finite(r.numpy()) - finite(o)).max()) for r, o in zip(ref, got))

        # 토크나이저: 실제 특징(사전 · 구조 포함)과 끝의 PAD 한 칸을 넣고 stanza Tokenizer.forward 와 비교
        tr = nlp.processors["tokenize"].trainer
        para = rt.paragraphs(" ".join(sample))[0]
        ids = index(rt.v["tokenize"]["main"])
        x = np.array([[ids.get(c, UNK) for c in para] + [PAD]], dtype=np.int64)
        f = rt.tok_features(para)
        ref = tr.model(torch.from_numpy(x), torch.from_numpy(f), [x.shape[1]])
        got = session(os.path.join(folder, "tokenizer.onnx")).run(None, {"x": x, "f": f})[0]
        out["tokenize"] = float(np.abs(ref.numpy() - got).max())

        if "mwt" in nlp.processors and nlp.processors["mwt"].trainer.model is not None:
            tr = nlp.processors["mwt"].trainer
            m = tr.model.cpu().eval()
            src, _ = rt.mwt_src(sample[0])
            if rt.v["mwt"]["type"] == "classifier":
                ref = conv.MwtClassifierGraph(m).eval()(torch.from_numpy(src))
                got = session(os.path.join(folder, "mwt_classifier.onnx")).run(None, {"src_ids": src})[0]
                # stanza CharacterClassifier.forward 와도 대조
                orig = m(torch.from_numpy(src), torch.from_numpy(src) == 0)
                out["mwt"] = max(float(np.abs(ref.numpy() - got).max()), float(np.abs(orig.numpy() - got).max()))
            else:
                enc, dec = conv.MwtEncoder(m).eval(), conv.LemmaDecoder(m).eval()
                s = torch.from_numpy(src)
                r_enc = enc(s)
                g_enc = session(os.path.join(folder, "mwt_encoder.onnx")).run(None, {"src_ids": src})
                diff = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(r_enc, g_enc))
                tok = torch.tensor([[2]])
                r_dec = dec(tok, r_enc[1], r_enc[2], r_enc[0], s)
                g_dec = session(os.path.join(folder, "mwt_decoder.onnx")).run(None, {"token": tok.numpy(), "hn": r_enc[1].numpy(), "cn": r_enc[2].numpy(), "h_in": r_enc[0].numpy(), "src_ids": src})
                # 원본 decode 한 걸음과도 대조 (늘린 칸은 원본에 없으므로 V 칸까지만)
                V = m.vocab_size
                orig, _ = m.decode(m.embedding(tok), r_enc[1], r_enc[2], r_enc[0], s == 0, src=s)
                diff = max(diff, max(float(np.abs(r.numpy() - o).max()) for r, o in zip(r_dec, g_dec)),
                           float(np.abs(orig[0, 0, :V].numpy() - g_dec[0][0, :V]).max()))
                out["mwt"] = diff
    return out


FIELDS = ["upos", "xpos", "feats", "lemma", "head", "deprel"]


def ref_sentences(doc):
    out = []
    for s in doc.sentences:
        ws = s.words
        out.append({"tokens": [t.text for t in s.tokens], "words": [w.text for w in ws],
                    "upos": [w.upos for w in ws], "xpos": [w.xpos or "_" for w in ws], "feats": [w.feats or "_" for w in ws],
                    "lemma": [w.lemma for w in ws], "head": [w.head for w in ws], "deprel": [w.deprel for w in ws]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", default="en,ko")
    ap.add_argument("--onnx", default="./onnx_final")
    ap.add_argument("--sentences", default="verify_sentences.json")
    ap.add_argument("--json", default=None, help="결과를 JSON 으로 저장")
    ap.add_argument("--show", type=int, default=3, help="어긋난 예를 몇 개까지 보일지")
    ap.add_argument("--dump", default=None, help="PyTorch Stanza 전체 파이프라인 결과(원문별)를 JSON 으로 저장 — 다른 실행기(Rust 등) 대조용")
    args = ap.parse_args()

    sents = json.load(open(args.sentences, encoding="utf-8"))
    report, dump = {}, {}
    for lang in [x for x in args.langs.split(",") if x]:
        folder = os.path.join(args.onnx, lang)
        rt = RefRuntime(folder)
        nlp = open_pipeline(lang)
        counts = {k: [0, 0] for k in FIELDS}
        e2e = {"tokens": [0, 0], "words": [0, 0], "all": [0, 0]}
        misses, e2e_misses = [], []
        for group in [g for g, v in sents[lang].items() if isinstance(v, list)]:
            for text in sents[lang][group]:
                ref = ref_sentences(nlp(text))
                dump.setdefault(lang, []).append({"text": text, "sentences": ref})
                got = rt.analyze(text)
                # 1. 원문 끝에서 끝
                same_tok = [r["tokens"] for r in ref] == [g["tokens"] for g in got]
                same_words = [r["words"] for r in ref] == [g["words"] for g in got]
                same_all = same_words and all(r[k] == g[k] for r, g in zip(ref, got) for k in FIELDS)
                for key, ok in (("tokens", same_tok), ("words", same_words), ("all", same_all)):
                    e2e[key][0] += int(ok)
                    e2e[key][1] += 1
                if not same_all:
                    e2e_misses.append({"text": text, "pytorch": [r["words"] for r in ref], "onnx": [g["words"] for g in got],
                                       "fields": [k for k in FIELDS if same_words and any(r[k] != g[k] for r, g in zip(ref, got))]})
                # 2. 처리기별 (파이프라인 낱말 · 태그를 넣고)
                for r in ref:
                    words = r["words"]
                    cols, _ = rt.pos(words)
                    g = {"upos": cols["upos"], "xpos": cols.get("xpos", ["_"] * len(words)), "feats": cols.get("feats", ["_"] * len(words))}
                    g["lemma"] = rt.lemma(words, r["upos"])
                    un, rel = rt.depparse(words, r["upos"], r["xpos"], r["lemma"])
                    g["head"], g["deprel"] = decode_tree(un, rel, rt.v["depparse"]["deprel"])
                    for k in FIELDS:
                        same = [a == b for a, b in zip(g[k], r[k])]
                        counts[k][0] += sum(same)
                        counts[k][1] += len(same)
                        for w, a, b, ok in zip(words, g[k], r[k], same):
                            if not ok:
                                misses.append((k, text, w, b, a))
        sample = ref_sentences(nlp(sents[lang]["query"][1]))[0]["words"]
        diffs = graph_diffs(lang, folder, sample, nlp, rt)
        report[lang] = {"end_to_end": {k: {"same": a, "total": b, "rate": a / b} for k, (a, b) in e2e.items()},
                        "per_processor": {k: {"same": int(a), "total": int(b), "rate": float(a / b)} for k, (a, b) in counts.items()},
                        "graph_max_abs_diff": diffs,
                        "end_to_end_mismatches": e2e_misses,
                        "mismatches": [{"field": k, "sentence": s, "word": w, "pytorch": str(b), "onnx": str(a)} for k, s, w, b, a in misses]}
        print(f"== {lang}: 원문 끝에서 끝 {e2e['all'][0]}/{e2e['all'][1]} 문장 일치 (토큰 {e2e['tokens'][0]} · 낱말 {e2e['words'][0]})")
        for k, (a, b) in counts.items():
            print(f"   {k:<7} {a}/{b} = {a / b:.1%}")
        print("   그래프 최대 절대 차이:", {k: f"{v:.2e}" for k, v in diffs.items()})
        for m in e2e_misses[: args.show]:
            print(f"   ✗ 끝에서 끝 '{m['text']}': PyTorch {m['pytorch']} · ONNX {m['onnx']} {m['fields']}")
        for k, s, w, b, a in misses[: args.show]:
            print(f"   ✗ {k} '{w}' ({s}): PyTorch {b!r} · ONNX {a!r}")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, ensure_ascii=False, indent=1)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
