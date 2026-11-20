"""Training loop με προβολή καταγραφών ανά epoch και προαιρετικό LR scheduler."""
import time
import torch
import torch.nn as nn


def train_one_run(model, train_loader, test_loader, epochs=30, lr=1e-3,
                  weight_decay=5e-4, optimizer_name='adam',
                  scheduler_name='constant', device='cuda', verbose=True):
    # Optimizer
    # Adam: προσαρμοστικό LR ανά παράμετρο, ταχύτερη σύγκλιση χωρίς tuning.
    # Κατάλληλο για 30 epochs, ο SGD θα απαιτούσε περισσότερα epochs και LR tuning.
    if optimizer_name == 'adam': 
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    else:
        raise ValueError(optimizer_name)

    # Scheduler
    if scheduler_name == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif scheduler_name == 'step':
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    else:
        sched = None

    crit = nn.CrossEntropyLoss()
    hist = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        hist['lr'].append(opt.param_groups[0]['lr'])

        # --- Train ---
        model.train()
        tl, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        tr_loss, tr_acc = tl / total, 100.0 * correct / total

        # --- Eval ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                correct += (model(x).argmax(1) == y).sum().item()
                total += x.size(0)
        te_acc = 100.0 * correct / total

        hist['epoch'].append(ep)
        hist['train_loss'].append(tr_loss)
        hist['train_acc'].append(tr_acc)
        hist['test_acc'].append(te_acc)

        if verbose:
            print(f"    Ep {ep:02d}/{epochs} | lr={hist['lr'][-1]:.5f} | "
                  f"loss={tr_loss:.3f} | train={tr_acc:5.2f}% | test={te_acc:5.2f}%")
        if sched is not None:
            sched.step()

    hist['time'] = time.time() - t0
    return hist