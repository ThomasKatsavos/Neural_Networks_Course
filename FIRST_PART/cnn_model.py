"""
Configurable CNN για CIFAR-100 — υποστηρίζει μεταβλητό βάθος.

Αρχιτεκτονική (VGG-style):
  Κάθε block = [Conv2d 3×3 → BatchNorm → ReLU] × 2 → MaxPool2d 2×2
  Τα κανάλια διπλασιάζονται ανά block (base_channels · 2^i),
  ενώ το spatial μέγεθος υποδιπλασιάζεται.

  Μετά από n_blocks blocks: Flatten -> Linear(·,512) -> ReLU -> Dropout(0.5) -> Linear(512,100)

Το βάθος (n_blocks) και το πλάτος (base_channels) είναι παράμετροι,
ώστε να υποστηρίζεται ablation study χωρίς περαιτέρω αλλαγή στον κώδικα.
"""

import torch.nn as nn


class CIFAR100CNN(nn.Module):
    def __init__(self, n_blocks=3, base_channels=32, num_classes=100, dropout=0.5):
        super().__init__()

        blocks = []
        in_ch = 3
        for i in range(n_blocks):
            out_ch = base_channels * (2 ** i)
            blocks += [
                nn.Conv2d(in_ch,  out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)

        # Υπολογισμός του  flatten dimension
        final_channels = base_channels * (2 ** (n_blocks - 1))
        final_size     = 32 // (2 ** n_blocks)
        flatten_dim    = final_channels * final_size * final_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))