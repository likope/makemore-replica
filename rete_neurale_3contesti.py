import torch
import random
import torch.nn.functional as F

names       = open("names.txt", "r").read().splitlines()    #salvataggio dei nomi
chars       = sorted(list(set(''.join(names))))             #lista dei caraatteri presenti nei nomi
stoi        = {s: i+1 for i,s in enumerate(chars)}          #da carattere a numero
stoi['.']   = 0
itos = {i: s for s, i in stoi.items()}
window_context = 3  #numero di caratteri di contesto

def build_dataset(names):
    """
    f per la costruzione del dataset input e target.
    """
    X,Y = [], []    #input, target = liste
    for name in names: #per ogni nome nella lista dei nomi
        context = [0] * window_context  #il primo input è uguale a [0, 0, 0]
        for char in name + '.': #per ogni carattere presente nei nomi.
            ix = stoi[char] #è uguale al numero del carattere
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

# ── split 80/10/10 (shuffle PRIMA dei build) ──────────
random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

Xtr,  Ytr  = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte,  Yte  = build_dataset(names[n2:])

C = torch.randn((27,10)) #prendi i 27 input(caratteri) e li trasformi in 10 dimensioni
W1 = torch.randn((30, 200))
b1 = torch.randn(200)
W2 = torch.randn((200, 27)) * 0.01
b2 = torch.randn(27) * 0
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

for k in range(100000):
    ix = torch.randint(0, Xtr.shape[0], (32,)) #randint genera n interi casuali, 0=minimo incluso, massimo escluso, la forma dell output
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters: #backward
        p.grad = None
    loss.backward()

    lr = 0.1 if k < 20000 else 0.01 #decresce il learning rate dopo 20000 giri
    for p in parameters:
        p.data += -lr * p.grad

    if k % 1000 == 0: #ogni mille giri stampa la loss
        print(loss.item())

name = []
context = [0] * window_context
#sampling
while True:
    emb = C[torch.tensor(context)]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1)
    ix = ix.item()
    if ix == 0:
        break
    name.append(ix)
    context = context[1:] + [ix]
print(''.join(itos[i] for i in name))

