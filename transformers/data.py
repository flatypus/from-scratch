from utils import token_split
from utils import ByteSeq
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch import nn

PAD_TOKEN = "<PAD>"
START_EN_TOKEN = "<2EN>"
START_YUE_TOKEN = "<2YUE>"
END_TOKEN = "<END>"
PAD_ID = 0


def encode(text, vocab, merges):
    merge_priority = {merge: i for i, merge in enumerate(merges)}
    words = token_split(text)
    all_ids = []
    for word in words:
        tokens = [ByteSeq(b) for b in word.encode("utf-8")]
        while len(tokens) > 1:
            best = None
            best_idx = None
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i+1]
                if pair in merge_priority:
                    if best is None or merge_priority[pair] < merge_priority[best]:
                        best = pair
                        best_idx = i
            if best is None:
                break
            tokens[best_idx] = best
            tokens.pop(best_idx + 1)
        for t in tokens:
            all_ids.append(vocab[t])
    return all_ids


class TranslationDataset(Dataset):
    def __init__(self, en_file, yue_file, vocab, merges):
        with open(en_file, encoding="utf-8") as f:
            en_lines = f.read().strip().split("\n")
        with open(yue_file, encoding="utf-8") as f:
            yue_lines = f.read().strip().split("\n")

        self.data = []
        for en, yue in tqdm(zip(en_lines, yue_lines), total = len(en_lines)):
            en_src = torch.tensor(encode(en, vocab, merges))
            yue_tgt = torch.tensor(
                [vocab[START_YUE_TOKEN]] +
                encode(yue, vocab, merges) +
                [vocab[END_TOKEN]]
            )
            self.data.append((en_src, yue_tgt))

            yue_src = torch.tensor(encode(yue, vocab, merges))
            en_tgt = torch.tensor(
                [vocab[START_EN_TOKEN]] +
                encode(en, vocab, merges) +
                [vocab[END_TOKEN]]
            )

            self.data.append((yue_src, en_tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD_ID)
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=PAD_ID)
    return srcs, tgts
