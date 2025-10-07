from torch.utils.data import Dataset


class BNOAuxedDataset(Dataset):
    def __init__(self, u, v, truth, aux):

        self.u = u
        self.v = v
        self.truth = truth
        self.aux = aux

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        aux_items = tuple(a[idx] for a in self.aux)

        return {'u': self.u[idx],
                'v': self.v[idx],
                'truth': self.truth[idx],
                'aux': aux_items}


class PCNOAuxedDataset(Dataset):
    def __init__(self, x, y, aux):

        self.x = x
        self.y = y
        self.aux = aux

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        aux_items = tuple(a[idx] for a in self.aux)

        return {'x': self.x[idx],
                'y': self.y[idx],
                'aux': aux_items}
