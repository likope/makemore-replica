import torch
import torch.nn.functional as f


words = open('names.txt', 'r').read().splitlines() #estrazione dei nomi dal file names.txt
chars = sorted(list(set('.'.join(words)))) #lista dei caratteri dell alfabeto
stoi = {s: i for i,s in enumerate(chars)} #da caratteri a numeri
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()} #da numeri a caratteri

xs, ys = [], [] #xs = carattere precedente, ys = carattere successivo

for w in words:
    """
    Per ogni nome aggiunge un punto all inizio e alla fine facendo imparare alla rete l inizio di un nome e la sua fine, poi mette i caratteri nelle liste della rete
    """
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs) #conversione in tensori per pytorch
ys = torch.tensor(ys)

xenc = f.one_hot(xs, num_classes=27).float() #vettori one-hot per la rete neurale
g = torch.Generator().manual_seed(2147483647) #generatore casuale, il seme rende la generazione fissa
W = torch.randn((27, 27), generator=g, requires_grad=True) #matrice 27x27 di numeri casuali

num = xs.nelement() #numero totale di esempi

for k in range(100):
    """
    La rete fa 100 passi e per ognuno calcola il gradiente con lo scopo di ridurre il piu possibile la loss.
    """

    # forward
    xenc = f.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() #calcolo della negative-log likehood

    # backward
    W.grad = None
    loss.backward() #calcolo dei gradienti

    # update
    W.data += -50 * W.grad #update dei pesi

    print(loss.item()) #print della loss

i = 0 #inizializzazione contatore 1
ix = 0 #inizializzazione indice
ch = [] #inizializzazione della lista dei nomi

#ciclo di generazione
while True:
    while True:
        xenc = f.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        if ix == 0:
            i = i+1
            print(ch)
            ch = []
            break
        ch.append(itos[ix])
    if i==10:
        break

#il ciclo cosi impostato genera 10 nomi