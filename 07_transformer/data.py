import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, List, Any
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch.nn.functional as F
import torch


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
    

def _iter_attention_modules(model):
    """
    Пройти по всем SingleHeadAttention в модели (encoder/decoder).
    Ожидается структура из предыдущих сообщений: enc_layers[i].attn,
    dec_layers[i].self_attn, dec_layers[i].cross_attn
    """
    for layer in getattr(model, "enc_layers", []):
        if hasattr(layer, "attn"):
            yield layer.attn
    for layer in getattr(model, "dec_layers", []):
        if hasattr(layer, "self_attn"):
            yield layer.self_attn
        if hasattr(layer, "cross_attn"):
            yield layer.cross_attn

def set_model_rope(model, enabled: bool):
    """Глобально включить/выключить RoPE у всех attention-модулей."""
    for attn in _iter_attention_modules(model):
        if hasattr(attn, "use_rope"):
            attn.use_rope = bool(enabled)

def set_model_local_window(model, window: Optional[int]):
    """
    Поставить «локальное окно» для self-attention энкодера и декодера.
    Для cross-attention окно обычно не применяют — принудительно None.
    """
    for layer in getattr(model, "enc_layers", []):
        if hasattr(layer, "attn") and hasattr(layer.attn, "local_window"):
            layer.attn.local_window = window
    for layer in getattr(model, "dec_layers", []):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "local_window"):
            layer.self_attn.local_window = window
        if hasattr(layer, "cross_attn") and hasattr(layer.cross_attn, "local_window"):
            layer.cross_attn.local_window = None  # cross-attn оставляем глобальным

# ====== ИНСПЕКТОР ======
class NMTInspector:
    """
    Быстрые вызовы:
      - BLEU:     inspector.bleu(dm.valid)
      - Примеры:  inspector.show_examples(dm.valid, n=5)
      - Внимание: inspector.plot_attention(dm.valid, which="encoder", layer=0, sample=0)

    Управление фичами:
      inspector.set_rope(True/False)
      inspector.set_local_window(w or None)
      inspector.set_kv_cache(True/False)
      inspector.set_no_repeat_ngram(n or None)

    Или на время одного вызова:
      inspector.bleu(dm.valid, opts={"use_cache": True, "no_repeat_ngram": 3})
    """

    def __init__(self, model, dm, device: Optional[str] = None,
                 use_cache: bool = False,
                 no_repeat_ngram: Optional[int] = None):
        self.model = model
        self.dm = dm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # runtime-параметры генерации
        self.use_cache = bool(use_cache)
        self.no_repeat_ngram = no_repeat_ngram

    # ---------- управление фичами ----------
    def set_rope(self, enabled: bool):
        set_model_rope(self.model, enabled)

    def set_local_window(self, window: Optional[int]):
        set_model_local_window(self.model, window)

    def set_kv_cache(self, enabled: bool):
        self.use_cache = bool(enabled)

    def set_no_repeat_ngram(self, n: Optional[int]):
        self.no_repeat_ngram = n

    # ---------- служебные утилиты ----------
    def _first_pad_index(self, ids: List[int], pad_idx: int) -> int:
        for i, t in enumerate(ids):
            if int(t) == pad_idx:
                return i
        return len(ids)

    def _effective_len_src(self, src_ids: List[int]) -> int:
        return self._first_pad_index(src_ids, self.dm.PAD_IDX)

    def _effective_len_tgt_input(self, tgt_ids: List[int]) -> int:
        if len(tgt_ids) <= 1:
            return 0
        dec_in = tgt_ids[:-1]
        return self._first_pad_index(dec_in, self.dm.TGT_PAD_IDX)

    def _ids_to_tokens(self, ids: List[int], itos: dict, skip_pad=True) -> List[str]:
        out = []
        for t in ids:
            w = itos.get(int(t), "<unk>")
            if skip_pad and w == self.dm.PAD:
                break
            out.append(w)
        return out

    def _plot_heatmap(self, attn_map: np.ndarray,
                      y_labels: Optional[List[str]] = None,
                      x_labels: Optional[List[str]] = None,
                      title: str = "Карта внимания"):
        attn_map = np.nan_to_num(attn_map, nan=0.0, posinf=0.0, neginf=0.0)
        if attn_map.ndim != 2:
            raise ValueError(f"Attention map must be 2D [Lq,Lk], got shape {attn_map.shape}")
        Lq, Lk = attn_map.shape
        plt.figure(figsize=(max(6, Lk * 0.4), max(4, Lq * 0.4)))
        plt.imshow(attn_map, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0)
        plt.colorbar(label="Attention weight")
        plt.title(title)
        plt.xlabel(f"Key (Lk={Lk})")
        plt.ylabel(f"Query (Lq={Lq})")
        if x_labels is not None:
            x_labels = x_labels[:Lk]
            plt.xticks(range(len(x_labels)), x_labels, rotation=90)
        if y_labels is not None:
            y_labels = y_labels[:Lq]
            plt.yticks(range(len(y_labels)), y_labels)
        plt.tight_layout()
        plt.show()

    def _attn_2d(self, attn, sample: int) -> np.ndarray:
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().float().cpu().numpy()
        if attn.ndim == 3:
            return attn[sample]
        if attn.ndim == 2:
            return attn
        raise ValueError(f"Unexpected attention ndim={attn.ndim}, expected 2 or 3.")

    # ---------- генерация / BLEU / примеры ----------
    @torch.no_grad()
    def _generate(self, src: torch.Tensor, *, opts: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Внутренний генератор, который пробрасывает флаги:
          - use_cache (KV-cache)
          - no_repeat_ngram
        """
        o = {"use_cache": self.use_cache, "no_repeat_ngram": self.no_repeat_ngram}
        if opts:
            o.update(opts)
        return self.model.generate(
            src,
            bos_id=self.dm.BOS_IDX,
            eos_id=self.dm.EOS_IDX,
            max_new_tokens=src.size(1) - 1,  # по умолчанию – не длиннее источника
            no_repeat_ngram=o["no_repeat_ngram"],
            use_cache=o["use_cache"],
        )

    @torch.no_grad()
    def translate_ids(self, src_ids: List[int], max_len: int = 64,
                      *, opts: Optional[Dict[str, Any]] = None) -> List[int]:
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        o = {"use_cache": self.use_cache, "no_repeat_ngram": self.no_repeat_ngram}
        if opts:
            o.update(opts)
        out = self.model.generate(
            src,
            bos_id=self.dm.BOS_IDX, eos_id=self.dm.EOS_IDX,
            max_new_tokens=max_len-1,
            no_repeat_ngram=o["no_repeat_ngram"],
            use_cache=o["use_cache"],
        )
        return out[0].tolist()

    @torch.no_grad()
    def bleu(self, loader: DataLoader, max_len: Optional[int] = None,
             num_batches: Optional[int] = None, *,
             opts: Optional[Dict[str, Any]] = None) -> float:
        smoothie = SmoothingFunction().method4
        refs, hyps = [], []
        processed = 0
        self.model.eval()

        for src, tgt in loader:
            B = src.size(0)
            for i in range(B):
                src_i = src[i:i+1].to(self.device)
                o = {"use_cache": self.use_cache, "no_repeat_ngram": self.no_repeat_ngram}
                if opts:
                    o.update(opts)
                gen = self.model.generate(
                    src_i,
                    bos_id=self.dm.BOS_IDX, eos_id=self.dm.EOS_IDX,
                    max_new_tokens=(max_len or src_i.size(1))-1,
                    no_repeat_ngram=o["no_repeat_ngram"],
                    use_cache=o["use_cache"],
                )[0].tolist()

                tgt_ids = tgt[i].tolist()
                ref_words = [self.dm.itos_tgt.get(t, "<unk>") for t in tgt_ids[1:]
                             if self.dm.itos_tgt.get(t, "") not in {self.dm.PAD, self.dm.BOS}]
                hyp_words = [self.dm.itos_tgt.get(t, "<unk>") for t in gen[1:]
                             if self.dm.itos_tgt.get(t, "") not in {self.dm.PAD, self.dm.BOS}]
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
    def show_examples(self, loader: DataLoader, n: int = 5, max_len: Optional[int] = None,
                      *, opts: Optional[Dict[str, Any]] = None):
        printed = 0
        self.model.eval()
        for src, tgt in loader:
            B = src.size(0)
            for i in range(B):
                src_i = src[i:i+1].to(self.device)
                o = {"use_cache": self.use_cache, "no_repeat_ngram": self.no_repeat_ngram}
                if opts:
                    o.update(opts)
                gen = self.model.generate(
                    src_i,
                    bos_id=self.dm.BOS_IDX, eos_id=self.dm.EOS_IDX,
                    max_new_tokens=(max_len or src.size(1))-1,
                    no_repeat_ngram=o["no_repeat_ngram"],
                    use_cache=o["use_cache"],
                )[0].tolist()

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

    # ---------- визуализация внимания (без PAD) ----------
    def _run_forward_for_attention(self, src: torch.Tensor, tgt: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(src.to(self.device), tgt[:, :-1].to(self.device))

    @torch.no_grad()
    def plot_attention(self, loader: DataLoader,
                       which: str = "encoder",
                       layer: int = 0,
                       sample: int = 0):
        # 1) возьмём батч и прогоним (teacher forcing)
        src_batch, tgt_batch = next(iter(loader))
        src_batch = src_batch.to(self.device)
        tgt_batch = tgt_batch.to(self.device)
        self._run_forward_for_attention(src_batch, tgt_batch)

        # 2) выберем пример и фактические длины (без PAD)
        src_ids = src_batch[sample].tolist()
        tgt_ids = tgt_batch[sample].tolist()
        L_src    = self._effective_len_src(src_ids)
        L_dec_in = self._effective_len_tgt_input(tgt_ids)

        # 3) достанем карту внимания и подписи
        if which == "encoder":
            attn_mod = self.model.enc_layers[layer].attn
            if attn_mod.last_attention is None:
                raise RuntimeError("encoder.attn.last_attention пуст. Сначала сделайте forward().")
            attn_map = self._attn_2d(attn_mod.last_attention, sample)[:L_src, :L_src]
            tokens = self._ids_to_tokens(src_ids[:L_src], self.dm.itos_src, skip_pad=True)
            self._plot_heatmap(attn_map, y_labels=tokens, x_labels=tokens,
                               title=f"Карта внимания (Encoder layer {layer})")
        elif which == "decoder_self":
            attn_mod = self.model.dec_layers[layer].self_attn
            if attn_mod.last_attention is None:
                raise RuntimeError("decoder.self_attn.last_attention пуст. Сначала сделайте forward().")
            attn_map = self._attn_2d(attn_mod.last_attention, sample)[:L_dec_in, :L_dec_in]
            tokens = self._ids_to_tokens(tgt_ids[:-1][:L_dec_in], self.dm.itos_tgt, skip_pad=True)
            self._plot_heatmap(attn_map, y_labels=tokens, x_labels=tokens,
                               title=f"Карта внимания (Decoder SELF layer {layer})")
        elif which == "decoder_cross":
            attn_mod = self.model.dec_layers[layer].cross_attn
            if attn_mod.last_attention is None:
                raise RuntimeError("decoder.cross_attn.last_attention пуст. Сначала сделайте forward().")
            attn_map = self._attn_2d(attn_mod.last_attention, sample)[:L_dec_in, :L_src]
            q_tokens = self._ids_to_tokens(tgt_ids[:-1][:L_dec_in], self.dm.itos_tgt, skip_pad=True)
            k_tokens = self._ids_to_tokens(src_ids[:L_src], self.dm.itos_src, skip_pad=True)
            self._plot_heatmap(attn_map, y_labels=q_tokens, x_labels=k_tokens,
                               title=f"Карта внимания (Decoder CROSS layer {layer})")
        else:
            raise ValueError("which должен быть одним из: 'encoder', 'decoder_self', 'decoder_cross'")