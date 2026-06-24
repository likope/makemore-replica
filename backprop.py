import torch
import torch.nn.functional as F
import random


names = open("names.txt", "r").read().splitlines()
chars = sorted(list(set(''.join(names))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)        
block_size = 3                

def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
Xtr, Ytr = build_dataset(names[:n1])


n_emb = 10                    
n_hidden = 200                
g = torch.Generator().manual_seed(2147483647)

C  = torch.randn((vocab_size, n_emb),      generator=g)
W1 = torch.randn((n_emb * block_size, n_hidden), generator=g) * (5/3)/((n_emb*block_size)**0.5)
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True


batch_size = 32
n = batch_size               
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]   


emb = C[Xb]                              # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)      # (32, 30)
hpreact = embcat @ W1 + b1               # (32, 200)
h = torch.tanh(hpreact)                  # (32, 200)
logits = h @ W2 + b2                     # (32, 27)

# cross_entropy vista da dentro:
logit_maxes = logits.max(1, keepdim=True).values   # (32, 1)
norm_logits = logits - logit_maxes                 # (32, 27)
counts = norm_logits.exp()                          # (32, 27)
counts_sum = counts.sum(1, keepdim=True)            # (32, 1)
counts_sum_inv = counts_sum**-1                     # (32, 1)
probs = counts * counts_sum_inv                     # (32, 27)
logprobs = probs.log()                              # (32, 27)
loss = -logprobs[range(n), Yb].mean()

for t in [logprobs, probs, counts_sum_inv, counts_sum, counts,
          norm_logits, logit_maxes, logits, h, hpreact, embcat, emb]:
    t.retain_grad()
for p in parameters:
    p.grad = None
loss.backward()

print(loss.item())
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
dprobs = dlogprobs * (1.0 / probs)
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
dcounts_sum = dcounts_sum_inv * (-counts_sum**-2)
dcounts = dprobs * counts_sum_inv + dcounts_sum * torch.ones_like(counts)
dnorm_logits = dcounts * counts
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits = dnorm_logits.clone()  
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
dh = dlogits @ W2.T          # (32,27) @ (27,200) = (32,200)
dW2 = h.T @ dlogits          # (200,32) @ (32,27) = (200,27)
db2 = dlogits.sum(0)         # (32,27) summed over rows = (27,)
dhpreact = (1.0 - h**2) * dh
dembcat = dhpreact @ W1.T       # dhpreact + the OTHER input (W1), transposed for shape
dW1     = embcat.T @ dhpreact   # dhpreact + the OTHER input (embcat), transposed for shape
db1     = dhpreact.sum(0)       # dhpreact summed over the batch (bias was broadcast)
demb = dembcat.view(emb.shape)       # (32,30) -> (32,3,10)
dC = torch.zeros_like(C)                 # (27,10), start at zero
for i in range(Xb.shape[0]):             # over 32 examples
    for j in range(Xb.shape[1]):         # over 3 positions
        ix = Xb[i, j]                    # which row of C was looked up
        dC[ix] += demb[i, j]             # ACCUMULATE, not assign
cmp('C', dC, C)