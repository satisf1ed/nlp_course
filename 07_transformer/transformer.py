from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class PositionalEncoding(nn.Module):
    """
        Класс для добавления информации о позиции токенов в последовательности.
        Эта реализация использует синусоидальные и косинусоидальные функции разных частот,
        как описано в статье "Attention Is All You Need".

        Кодировки не являются обучаемыми параметрами. Они вычисляются заранее и
        добавляются к эмбеддингам токенов.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]

        # Создаем тензор, представляющий абсолютные позиции токенов: [0, 1, 2, ..., max_len-1].
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]

        # Вычисляем знаменатель для формулы позиционной кодировки.
        # Это логарифмически масштабированный набор частот, который позволяет модели легко определять относительные позиции.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # заполняем чётные/нечётные индексы разными функциями
        pe[:, 0::2] = torch.sin(position * div_term)  # sin на чётных
        pe[:, 1::2] = torch.cos(position * div_term)  # cos на нечётных

        # register_buffer() регистрирует тензор 'pe' как часть модели.
        # Это важно, т.к. в отличие от обычного параметра, он не будет обучаться (не попадет в optimizer),
        # но будет перемещаться между устройствами (CPU/GPU) вместе с моделью вызовом model.to(device).
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Прямой проход: добавляет позиционные кодировки к входному тензору.

            Args:
                x (torch.Tensor): Входной тензор эмбеддингов токенов.
                                Форма: (batch_size, seq_len, d_model)

            Returns:
                torch.Tensor: Тензор с добавленной позиционной информацией.
                            Форма: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)

        # К каждому вектору эмбеддинга в 'x' прибавляем соответствующий
        # вектор позиционной кодировки. Мы берем срез [ :seq_len, : ] из нашей
        # предвычисленной матрицы 'pe'. Broadcasting позаботится о сложении
        # тензора (seq_len, d_model) с тензором (batch, seq_len, d_model).
        return x + self.pe[:seq_len, :]

class SingleHeadAttention(nn.Module):
    """
        Класс, реализующий механизм внимания с одной "головой" (Single-Head Attention).

        Она выполняет всю основную логику:
        1. Линейно проецирует входы в пространства Query, Key, Value.
        2. Вычисляет оценки внимания через скалярное произведение (dot-product).
        3. Масштабирует оценки, применяет маску и softmax для получения весов.
        4. Применяет веса к векторам Value для получения взвешенного результата.
        5. Пропускает результат через финальный линейный слой.
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Финальный линейный слой, который преобразует результат внимания
        # перед его передачей на следующий уровень сети.
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): Тензор запросов (query). Форма: [batch, seq_len_q, d_model]
            k (torch.Tensor): Тензор ключей (key). Форма: [batch, seq_len_k, d_model]
            v (torch.Tensor): Тензор значений (value). Форма: [batch, seq_len_k, d_model]
            mask (Optional[torch.Tensor]): Маска для скрытия определенных позиций.
                                           Форма: [batch, 1, seq_len_q, seq_len_k] или [batch, seq_len_q, seq_len_k]

        Returns:
            torch.Tensor: Выходной тензор после применения внимания. Форма: [batch, seq_len_q, d_model]
        """
        Q = self.W_q(q)  # Форма: [batch, seq_len_q, d_model]
        K = self.W_k(k)  # Форма: [batch, seq_len_k, d_model]
        V = self.W_v(v)  # Форма: [batch, seq_len_k, d_model]

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)  # [b, Lq, Lk]

        if mask is not None:
            # Этот блок нужен для унификации. Маска может прийти с лишней размерностью "heads".
            # Например, с формой [batch, 1, seq_len, seq_len]
            if mask.dim() == 4:  # [batch, 1, seq_len, seq_len]
                mask = mask.squeeze(1)

            # Заполняем элементы scores очень маленьким числом там, где маска равна 0.
            # Это приведет к тому, что после softmax на этих позициях будут нули.
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # Форма: [batch, seq_len_q, seq_len_k]
        self.last_attention = attn.detach().cpu()

        out = torch.matmul(attn, V)       # Форма: [batch, seq_len_q, d_model]
        out = self.W_o(out)               # Форма: [batch, seq_len_q, d_model]
        out = self.dropout(out)

        return out

    def visualize_attention(self, query_tokens=None, key_tokens=None, layer_name="SingleHeadAttention"):
        """
            Простая визуализация тепловой карты внимания для первого элемента батча.
        """
        attn_map = self.last_attention[0].numpy()  # [Lq, Lk]

        plt.figure(figsize=(8, 6))
        plt.imshow(attn_map, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Карта внимания ({layer_name})")
        plt.xlabel("Key")
        plt.ylabel("Query")

        if key_tokens is not None:
            plt.xticks(range(len(key_tokens)), key_tokens, rotation=90)
        if query_tokens is not None:
            plt.yticks(range(len(query_tokens)), query_tokens)

        plt.tight_layout()
        plt.show()

class PositionwiseFeedForward(nn.Module):
    """
        Реализует полносвязную нейронную сеть (Feed-Forward Network), которая
        применяется к к каждому вектору эмбеддинга

        Состоит из двух линейных слоев с функцией активации ReLU между ними.
        forward(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        # для визуализаций
        self.last_hidden = None  # активации после ReLU: [b,L,d_ff]

    def forward(self, x):
        h = self.relu(self.lin1(x))
        self.last_hidden = h.detach()
        y = self.lin2(self.drop(h))
        return y

class EncoderLayer(nn.Module):
    """
        Один слой кодировщика Transformer.

        Состоит из двух основных подслоев:
        1. Механизм self-attention
        2. Позиционно-ориентированная полносвязная сеть (PositionwiseFeedForward)

        Вокруг каждого из этих подслоев применяется остаточное соединение (residual connection)
        с последующей нормализацией слоя (Layer Normalization). Это т.н. "Add & Norm".
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = SingleHeadAttention(d_model, dropout=dropout)   # <- вот здесь
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Обратите внимание: `x` передается как `q`, `k` и `v`. Это и есть
        # "self-attention" — последовательность ищет связи внутри самой себя.
        attn_out = self.attn(x, x, x, src_mask)

        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.ffn(x)

        x = self.norm2(x + self.dropout2(ff_out))
        return x

class DecoderLayer(nn.Module):
    """
        Один слой декодера Transformer.

        Декодер немного сложнее, так как у него три основных подслоя:
        1. Маскированное self-attention (внимание к самому себе в целевой последовательности).
        Маска не дает "подсматривать" в будущие токены при генерации.
        2. Cross-attention (перекрестное внимание), где декодер "смотрит" на выход
        энкодера, чтобы сопоставить исходную и целевую последовательности.
        3. Позиционно-ориентированная полносвязная сеть (PositionwiseFeedForward).

        Как и в энкодере, вокруг каждого подслоя есть "Add & Norm" (остаточное
        соединение и нормализация слоя).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = SingleHeadAttention(d_model, dropout=dropout)
        self.cross_attn = SingleHeadAttention(d_model, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor],
                tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Декодер смотрит на уже сгенерированные токены.
        # `tgt_mask` не позволяет позиции i "видеть" позиции j > i.
        a1 = self.self_attn(x, x, x, tgt_mask)

        x = self.norm1(x + self.dropout1(a1))

        # Это самый важный шаг. Декодер использует информацию, полученную на предыдущем шаге,
        # чтобы "задать вопрос" (Query) к "памяти" энкодера (`memory`).
        # q=x:                "Что мне сгенерировать дальше, учитывая то, что я уже сгенерировал?"
        # k=memory, v=memory: "Вот полная информация об исходном предложении. Найди релевантные части."
        # `src_mask` используется, чтобы игнорировать <pad> токены в исходном предложении.
        a2 = self.cross_attn(x, memory, memory, src_mask)

        x = self.norm2(x + self.dropout2(a2))
        f = self.ffn(x)
        x = self.norm3(x + self.dropout3(f))
        return x

def generate_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
        Создает маску для скрытия <pad> токенов.

        Args:
            seq (torch.Tensor): Входная последовательность токенов. Форма: [Batch, SeqLen]
            pad_idx (int): Индекс токена-заполнителя (<pad>).

        Returns:
            torch.Tensor: Маска, где 1 означает "обращать внимание", а 0 - "игнорировать".
                        Финальная форма [Batch, 1, 1, SeqLen] подготовлена для
                        broadcasting'а с матрицей внимания [B, H, L, L].
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def generate_subsequent_mask(size: int) -> torch.Tensor:
    """
        Создает "look-ahead" маску, чтобы декодер не мог подсматривать в будущее.

        Args:
            size (int): Длина последовательности (L).

        Returns:
            torch.Tensor: Нижнетреугольная матрица. Форма: [1, 1, L, L].
                        Например, для size=3 это будет:
                        [[[1, 0, 0],
                          [1, 1, 0],
                          [1, 1, 1]]]
    """
    return torch.tril(torch.ones(1, 1, size, size, dtype=torch.uint8))

def combine_masks(pad_mask: torch.Tensor, look_ahead_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
        Объединяет маску для padding'а и look-ahead маску (если она есть).
    """
    if look_ahead_mask is None:
        return pad_mask
    return pad_mask & look_ahead_mask

class Transformer(nn.Module):
    """
        Полная архитектура Transformer, объединяющая Encoder и Decoder.
    """
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 1,             # параметр оставлен для совместимости. По факту код не поддерживает n_heads > 1.
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 pad_idx: int = 0):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, 1, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, 1, d_ff, dropout) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.pad_idx = pad_idx

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        device = src.device

        # Создание масок
        src_pad_mask = generate_padding_mask(src, self.pad_idx).to(device)  # Маска для <pad> в исходнике
        tgt_pad_mask = generate_padding_mask(tgt, self.pad_idx).to(device)  # Маска для <pad> в таргете
        look_ahead   = generate_subsequent_mask(tgt.size(1)).to(device)     # "Look-ahead" маска для таргета
        tgt_mask     = combine_masks(tgt_pad_mask, look_ahead)              # Финальная маска для декодера

        # Умножение на sqrt(d_model) - деталь из оригинальной статьи "Attention is All You Need".
        # Это делается для того, чтобы эмбеддинги и позиционные кодировки имели сопоставимую магнитуду.
        src_emb = self.dropout(self.pos_encoding(self.src_embed(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embed(tgt) * math.sqrt(self.d_model)))

        # `memory` - это выход энкодера, который содержит контекстуализированное
        # представление исходной последовательности. Он будет использоваться в каждом слое декодера.
        memory = src_emb
        for layer in self.enc_layers:
            memory = layer(memory, src_pad_mask)

        # Выход одного слоя декодера становится входом для следующего
        out = tgt_emb
        for layer in self.dec_layers:
            # Каждый слой декодера принимает эмбеддинги таргета, "память" кодировщика и обе маски
            out = layer(out, memory, src_pad_mask, tgt_mask)

        # Преобразуем выход последнего слоя декодера [B, Lt, d_model] в логиты [B, Lt, Vt].
        logits = self.out_proj(out)
        return logits