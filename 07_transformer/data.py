def get_loaders(
    max_seq_len: int = 64,
    batch_size: int = 32,
    min_freq: int = 2,
    dataset_name: str = "bentrevett/multi30k",
    src_field: str = "de",
    tgt_field: str = "en",
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    import re
    from collections import Counter
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import torch

    ds = load_dataset(dataset_name)

    _word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(text: str) -> list[str]:
        return [t.lower() for t in _word_re.findall(text.strip())]

    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    train_pairs = [
        (tokenize(ex[src_field]), tokenize(ex[tgt_field]))
        for ex in ds["train"]
    ]

    def build_vocab(sents, min_freq=2):
        c = Counter(tok for s in sents for tok in s)
        vocab = [PAD, BOS, EOS, UNK] + [w for w, f in c.items() if f >= min_freq]
        stoi = {w: i for i, w in enumerate(vocab)}
        return vocab, stoi

    src_vocab, src_stoi = build_vocab([s for s, _ in train_pairs], min_freq=min_freq)
    tgt_vocab, tgt_stoi = build_vocab([t for _, t in train_pairs], min_freq=min_freq)

    PAD_IDX = src_stoi[PAD]  # 0
    BOS_IDX = tgt_stoi[BOS]
    EOS_IDX = tgt_stoi[EOS]
    UNK_IDX = tgt_stoi[UNK]

    def encode(tokens: list[str], stoi: dict[str, int]) -> list[int]:
        unk = stoi.get(UNK, stoi.get("<unk>", 0))
        return [stoi.get(t, unk) for t in tokens]

    def pad_to_max_len(ids: list[int], max_len: int, pad_idx: int) -> list[int]:
        return ids[:max_len] if len(ids) >= max_len else ids + [pad_idx] * (max_len - len(ids))

    def make_collate_fn(max_len: int):
        def collate(batch: list[dict]):
            src_batch_ids, tgt_batch_ids = [], []
            for item in batch:
                src_toks = tokenize(item[src_field])
                tgt_toks = tokenize(item[tgt_field])

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

    collate = make_collate_fn(max_seq_len)

    train_loader = DataLoader(
        ds["train"], batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, collate_fn=collate
    )
    valid_loader = DataLoader(
        ds["validation"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate
    )
    test_loader = DataLoader(
        ds["test"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate
    )

    return train_loader, valid_loader, test_loader
