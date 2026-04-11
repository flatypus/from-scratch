import os, pickle, torch
from utils import ByteSeq
from model import Transformer
from data import encode, PAD_TOKEN, END_TOKEN, PAD_ID

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("mps")

shared = pickle.load(open(os.path.join(HERE, "shared.pkl"), "rb"))
chars, merges = shared["chars"], shared["tokens"]

START_TOKEN = "<START>"
vocab_list = [PAD_TOKEN] + sorted(chars) + merges + [START_TOKEN, END_TOKEN]
vocab = {t: i for i, t in enumerate(vocab_list)}
inv = {i: t for t, i in vocab.items()}
START, END = vocab[START_TOKEN], vocab[END_TOKEN]

model = Transformer(
    len(vocab), 
    len(vocab), 
    d_model=256, 
    n_heads=8,
    source_l_layers=3, 
    target_l_layers=3, 
    max_seq_len=512, 
    dropout=0.0
)

model.load_state_dict(torch.load(os.path.join(HERE, "best_model.pt"), map_location=device, weights_only=True))
model = model.to(device).eval()


def detokenize(ids):
    skip = {PAD_ID, START, END}
    return b"".join(inv[i].data for i in ids if i not in skip).decode("utf-8", errors="replace")


@torch.no_grad()
def translate(en, max_new=80):
    enc = model.encoder(torch.tensor([encode(en, vocab, merges)], device=device))
    out = [START]
    for _ in range(max_new):
        nxt = int(model.decoder(torch.tensor([out], device=device), enc)[0, -1].argmax())
        out.append(nxt)
        if nxt == END:
            break
    return detokenize(out)

for s in ["McDonald's is the best fast food restaurant in the world, but 正斗 in Richmond, BC is my favorite restaurant in the world."]:
    print(f"EN:  {s}\nYUE: {translate(s)}\n")
