import torch
import torchvision

# Φόρτωση του dataset σε τανυστές(tensors) στην GPU, με flatten.
def dataset_to_tensors(ds, device, batch_size=1000):
    xs, ys = [], []
    for x, y in torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False):
        xs.append(x.view(x.size(0), -1))        # (B, 3072)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)

# Υπολογίζει μια φορά τις αποστάσεις από τους k κοντινότερους γείτονες, για πολλές χρήσεις από τον αντίστοιχο ταξινομητή
def precompute_knn(X_test, X_train, k=3, chunk_size=500):
    all_idx, all_dist = [], []
    for s in range(0, X_test.size(0), chunk_size):
        xb = X_test[s : s + chunk_size]
        d = torch.cdist(xb, X_train)                              # (b, 50000)
        top_d, top_i = torch.topk(d, k, dim=1, largest=False)
        all_idx.append(top_i)
        all_dist.append(top_d)
    return torch.cat(all_idx), torch.cat(all_dist)

# Αξιολόγηση/μετρηση της ακρίβειας του k-nn
def knn_accuracy(top_idx, train_labels, test_labels, k=1):
    if k == 1:
        preds = train_labels[top_idx[:, 0]]
    else:
        nn_labels = train_labels[top_idx[:, :k]]                  # (N_test, k)
        preds = torch.mode(nn_labels, dim=1).values               # πλειοψηφία
    return (preds == test_labels).float().mean().item() * 100.0

# Έλεγχος του ncc με επίσης μεταβλητή είσοδο τα train_labels 
def ncc_accuracy(X_train, train_labels, X_test, test_labels, n_classes=100):
    device = X_train.device
    dim = X_train.size(1)
    centroids = torch.zeros(n_classes, dim, device=device)
    present = torch.zeros(n_classes, dtype=torch.bool, device=device)

    for c in range(n_classes):
        mask = (train_labels == c)
        if mask.any():
            centroids[c] = X_train[mask].mean(dim=0)
            present[c] = True

    dists = torch.cdist(X_test, centroids)                        # (N_test, 100)
    dists[:, ~present] = float('inf')                             # κλάσεις χωρίς δείγματα
    preds = dists.argmin(dim=1)
    return (preds == test_labels).float().mean().item() * 100.0