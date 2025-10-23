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
