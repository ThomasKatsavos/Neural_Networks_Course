import time
import torch
import torchvision
import torchvision.transforms as transforms
from classifiers import dataset_to_tensors, precompute_knn, knn_accuracy, ncc_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([transforms.ToTensor()])
train_ds = torchvision.datasets.CIFAR100(root='./data', train=True,  download=True, transform=transform)
test_ds  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

X_train, y_train = dataset_to_tensors(train_ds, device)
X_test,  y_test  = dataset_to_tensors(test_ds,  device)

def sync():
    if device == 'cuda': torch.cuda.synchronize()

# --- NCC ---
sync(); t0 = time.time()
acc_ncc = ncc_accuracy(X_train, y_train, X_test, y_test)
sync(); t_ncc = time.time() - t0

# --- kNN (precompute μια φορά) ---
sync(); t0 = time.time()
top3_idx, _ = precompute_knn(X_test, X_train, k=3)
acc_1nn = knn_accuracy(top3_idx, y_train, y_test, k=1)
acc_3nn = knn_accuracy(top3_idx, y_train, y_test, k=3)
sync(); t_knn = time.time() - t0

print(f"NCC  : {acc_ncc:.2f}%  ({t_ncc:.2f}s)")
print(f"1-NN : {acc_1nn:.2f}%  ({t_knn:.2f}s)")
print(f"3-NN : {acc_3nn:.2f}%  ({t_knn:.2f}s)")