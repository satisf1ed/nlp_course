import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, List
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch.nn.functional as F
import torch


class NMTInspector:
    """
    Быстрые вызовы:
      - BLEU:     inspector.bleu(dm.valid)
      - Примеры:  inspector.show_examples(dm.valid, n=5)
      - Внимание: inspector.plot_attention(dm.valid, which="encoder", layer=0, sample=0)
                  which ∈ {"encoder","decoder_self","decoder_cross"}
    """
    def __init__(self, model, dm: NMTDataModule, device: Optional[str] = None):
        self.model = model
        self.dm = dm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ---------- генерация одной фразой ----------
    @torch.no_grad()
    def translate_ids(self, src_ids: List[int], max_len: int = 64) -> List[int]:
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        out = self.model.generate(src, bos_id=self.dm.BOS_IDX, eos_id=self.dm.EOS_IDX,
                                  max_new_tokens=max_len-1)
        return out[0].tolist()

    @torch.no_grad()
    def bleu(self, loader: DataLoader, max_len: Optional[int] = None, num_batches: Optional[int] = None) -> float:
        smoothie = SmoothingFunction().method4
        refs, hyps = [], []
        processed = 0
        for src, tgt in loader:
            B = src.size(0)
            for i in range(B):
                src_i = src[i:i+1].to(self.device)
                gen = self.model.generate(src_i, bos_id=self.dm.BOS_IDX, eos_id=self.dm.EOS_IDX,
                                          max_new_tokens=(max_len or src_i.size(1))-1)[0].tolist()
                tgt_ids = tgt[i].tolist()

                ref_words = [self.dm.itos_tgt.get(t, "<unk>") for t in tgt_ids[1:] if self.dm.itos_tgt.get(t,"") not in {self.dm.PAD,self.dm.BOS}]
                hyp_words = [self.dm.itos_tgt.get(t, "<unk>") for t in gen[1:]       if self.dm.itos_tgt.get(t,"") not in {self.dm.PAD,self.dm.BOS}]

                # обрезать по EOS
                if self.dm.EOS in ref_words:
                    ref_words = ref_words[:ref_words.index(self.dm.EOS)]
                if self.dm.EOS in hyp_words:
                    hyp_words = hyp_words[:hyp_words.index(self.dm.EOS)]

                refs.append([ref_words])
                hyps.append(hyp_words)

            processed += 1
            if num_batches is not None and processed >= num_batches:
                break

        return corpus_bleu(refs, hyps, smoothing_function=smoothie) * 100.0

    @torch.no_grad()
    def show_examples(self, loader: DataLoader, n: int = 5, max_len: Optional[int] = None):
        printed = 0
        for src, tgt in loader:
            B = src.size(0)
            for i in range(B):
                gen = self.model.generate(src[i:i+1].to(self.device),
                                          bos_id=self.dm.BOS_IDX, eos_id=self.dm.EOS_IDX,
                                          max_new_tokens=(max_len or src.size(1))-1)[0].tolist()
                src_text  = self.dm.decode_ids(src[i].tolist(), lang="src", stop_at_eos=False)
                pred_text = self.dm.decode_ids(gen, lang="tgt", stop_at_eos=True)
                true_text = self.dm.decode_ids(tgt[i].tolist(), lang="tgt", stop_at_eos=True)
                print("\n---")
                print("SRC :", src_text)
                print("PRED:", pred_text)
                print("TRUE:", true_text)
                printed += 1
                if printed >= n:
                    return

    # ---------- визуализация внимания ----------
    def _run_forward_for_attention(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Прогоняем один батч через модель, чтобы заполнить last_attention
        во всех SingleHeadAttention-ах. Возвращаем memory/промежуточные не нужны —
        главное, что внутри модулей уже есть self.last_attention.
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(src.to(self.device), tgt[:, :-1].to(self.device))

    @torch.no_grad()
    def plot_attention(self, loader: DataLoader,
                       which: str = "encoder",
                       layer: int = 0,
                       sample: int = 0):
        """
        which ∈ {"encoder", "decoder_self", "decoder_cross"}
        """
        # берём один батч и одну строку
        src, tgt = next(iter(loader))
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        self._run_forward_for_attention(src, tgt)

        # выберем нужный слой и модуль внимания
        if which == "encoder":
            attn_mod = self.model.enc_layers[layer].attn
            # подписи
            src_tokens = [self.dm.itos_src.get(int(t), "<unk>") for t in src[sample].tolist()]
            # рисуем как есть
            attn_mod.visualize_attention(query_tokens=src_tokens, key_tokens=src_tokens,
                                         layer_name=f"Encoder layer {layer}")
        elif which == "decoder_self":
            attn_mod = self.model.dec_layers[layer].self_attn
            tgt_tokens = [self.dm.itos_tgt.get(int(t), "<unk>") for t in tgt[sample].tolist()]
            attn_mod.visualize_attention(query_tokens=tgt_tokens[:-1], key_tokens=tgt_tokens[:-1],
                                         layer_name=f"Decoder SELF layer {layer}")
        elif which == "decoder_cross":
            attn_mod = self.model.dec_layers[layer].cross_attn
            tgt_tokens = [self.dm.itos_tgt.get(int(t), "<unk>") for t in tgt[sample].tolist()]
            src_tokens = [self.dm.itos_src.get(int(t), "<unk>") for t in src[sample].tolist()]
            attn_mod.visualize_attention(query_tokens=tgt_tokens[:-1], key_tokens=src_tokens,
                                         layer_name=f"Decoder CROSS layer {layer}")
        else:
            raise ValueError("which должен быть одним из: 'encoder', 'decoder_self', 'decoder_cross'")


@dataclass
class DataConfig:
    dataset_name: str = "bentrevett/multi30k"
    src_field: str = "de"
    tgt_field: str = "en"
    max_seq_len: int = 64
    batch_size: int = 64
    min_freq: int = 2
    num_workers: int = 0
    shuffle_train: bool = True

class NMTDataModule:
    """
    Упрощённый модуль данных:
      - .train/.valid/.test — DataLoader’ы
      - .stoi / .itos — словари
      - .PAD/BOS/EOS индексы
      - .decode_ids() и .detok() — удобные декодеры в текст
      - .from_hf(...) — статический конструктор одной строкой
    """

    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader,
                 src_vocab: List[str],
                 tgt_vocab: List[str],
                 src_stoi: Dict[str, int],
                 tgt_stoi: Dict[str, int],
                 tokenizer: Callable[[str], List[str]],
                 src_field: str,
                 tgt_field: str):
        self.train = train_loader
        self.valid = valid_loader
        self.test  = test_loader

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_stoi  = src_stoi
        self.tgt_stoi  = tgt_stoi
        self.itos_src  = {i:w for w,i in src_stoi.items()}
        self.itos_tgt  = {i:w for w,i in tgt_stoi.items()}

        self.PAD_IDX = src_stoi[self.PAD]  # 0
        self.TGT_PAD_IDX = tgt_stoi[self.PAD]
        self.BOS_IDX = tgt_stoi[self.BOS]
        self.EOS_IDX = tgt_stoi[self.EOS]

        self._tok = tokenizer
        self.src_field = src_field
        self.tgt_field = tgt_field

    # ---------- статический конструктор ----------
    @staticmethod
    def from_hf(cfg: DataConfig) -> "NMTDataModule":
        """
        ОДНА СТРОКА:
            dm = NMTDataModule.from_hf(DataConfig(...))
        """
        ds = load_dataset(cfg.dataset_name)

        _word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        def tokenize(text: str) -> list[str]:
            return [t.lower() for t in _word_re.findall(text.strip())]

        # соберём пары из train для построения словарей
        train_pairs = [
            (tokenize(ex[cfg.src_field]), tokenize(ex[cfg.tgt_field]))
            for ex in ds["train"]
        ]

        def build_vocab(sents, min_freq=2):
            c = Counter(tok for s in sents for tok in s)
            vocab = [NMTDataModule.PAD, NMTDataModule.BOS, NMTDataModule.EOS, NMTDataModule.UNK] \
                    + [w for w, f in c.items() if f >= min_freq]
            stoi = {w: i for i, w in enumerate(vocab)}
            return vocab, stoi

        src_vocab, src_stoi = build_vocab([s for s, _ in train_pairs], cfg.min_freq)
        tgt_vocab, tgt_stoi = build_vocab([t for _, t in train_pairs], cfg.min_freq)

        PAD_IDX = src_stoi[NMTDataModule.PAD]
        BOS_IDX = tgt_stoi[NMTDataModule.BOS]
        EOS_IDX = tgt_stoi[NMTDataModule.EOS]

        def encode(tokens: list[str], stoi: dict[str, int]) -> list[int]:
            unk = stoi.get(NMTDataModule.UNK, stoi.get("<unk>", 0))
            return [stoi.get(t, unk) for t in tokens]

        def pad_to_max_len(ids: list[int], max_len: int, pad_idx: int) -> list[int]:
            return ids[:max_len] if len(ids) >= max_len else ids + [pad_idx] * (max_len - len(ids))

        def make_collate_fn(max_len: int):
            def collate(batch: list[dict]):
                src_batch_ids, tgt_batch_ids = [], []
                for item in batch:
                    src_toks = tokenize(item[cfg.src_field])
                    tgt_toks = tokenize(item[cfg.tgt_field])

                    s_ids = encode(src_toks, src_stoi)
                    t_ids = [BOS_IDX] + encode(tgt_toks, tgt_stoi) + [EOS_IDX]

                    s_ids = pad_to_max_len(s_ids, max_len, PAD_IDX)
                    t_ids = pad_to_max_len(t_ids, max_len, PAD_IDX)

                    src_batch_ids.append(s_ids)
                    tgt_batch_ids.append(t_ids)

                return (
                    torch.tensor(src_batch_ids, dtype=torch.long),
                    torch.tensor(tgt_batch_ids, dtype=torch.long),
                )
            return collate

        collate = make_collate_fn(cfg.max_seq_len)
        train_loader = DataLoader(ds["train"], batch_size=cfg.batch_size, shuffle=cfg.shuffle_train,
                                  num_workers=cfg.num_workers, collate_fn=collate)
        valid_loader = DataLoader(ds["validation"], batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, collate_fn=collate)
        test_loader  = DataLoader(ds["test"], batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, collate_fn=collate)

        return NMTDataModule(
            train_loader, valid_loader, test_loader,
            src_vocab, tgt_vocab, src_stoi, tgt_stoi, tokenize,
            src_field=cfg.src_field, tgt_field=cfg.tgt_field
        )

    # ---------- утилиты расшифровки ----------
    def detok(self, tokens: List[str]) -> str:
        txt = " ".join(tokens)
        txt = txt.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?") \
                 .replace(" ;", ";").replace(" :", ":").replace(" '", "'")
        return txt.strip()

    def decode_ids(self, ids: List[int], *, lang: str = "tgt", stop_at_eos: bool = True) -> str:
        itos = self.itos_tgt if lang == "tgt" else self.itos_src
        special_skip = {self.PAD, self.BOS, self.EOS}
        words = []
        for t in ids:
            w = itos.get(int(t), "<unk>")
            if w in special_skip:
                if stop_at_eos and w == self.EOS: break
                continue
            words.append(w)
        return self.detok(words)
