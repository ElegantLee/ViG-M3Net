import glob
import random
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

################## grascale ######################
# class ImageDataset(Dataset):
#     def __init__(self, root,count = None,transforms_1=None,transforms_2=None, unaligned=False):
#         self.transform1 = transforms.Compose(transforms_1)
#         self.transform2 = transforms.Compose(transforms_2)
#         #######################OASIS###########################
#         # root_A = Path(os.path.join(os.getcwd(), "%s/A/train_A.npy" % root)).as_posix()
#         # root_B = Path(os.path.join(os.getcwd(), "%s/B/train_B.npy" % root)).as_posix()
#         # root_A = Path(os.path.join(os.getcwd(), "%s/A/A_debug.npy" % root)).as_posix()
#         # root_B = Path(os.path.join(os.getcwd(), "%s/B/B_debug.npy" % root)).as_posix()
#         # # (509, 256, 256, 3)--->(B, H, W, C)
#         # self.files_A = np.load(root_A).astype(np.float32)   # (H,W,C)
#         # self.files_B = np.load(root_B).astype(np.float32)
#         root_A = Path(os.path.join(os.getcwd(), "%s/A/train_A.npy" % root)).as_posix()
#         root_B = Path(os.path.join(os.getcwd(), "%s/B/train_B.npy" % root)).as_posix()
#         self.files_A = np.load(root_A).astype(np.float32)   # (H,W,C)
#         self.files_B = np.load(root_B).astype(np.float32)
#         self.unaligned = unaligned
#
#     def __getitem__(self, index):
#         seed = np.random.randint(2147483647) # make a seed with numpy generator
#         random.seed(seed) # apply this seed to img tranfsorms
#         # tensor(3, 256, 256)
#         item_A = self.transform1(self.files_A[index % self.files_A.shape[0]])
#         random.seed(seed)
#         if self.unaligned:
#             item_B = self.transform2(self.files_B[random.randint(0, self.files_B.shape[0] - 1)])
#
#         else:
#             item_B = self.transform2(self.files_B[index % self.files_B.shape[0]])
#         return {'A_img': item_A, 'B_img': item_B}
#     def __len__(self):
#         return max(self.files_A.shape[0], self.files_B.shape[0])
#
#
# class ValDataset(Dataset):
#     def __init__(self, root,count = None,transforms_=None, unaligned=False):
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned
#         root_A = Path(os.path.join(os.getcwd(), "%s/A/val_A.npy" % root)).as_posix()
#         root_B = Path(os.path.join(os.getcwd(), "%s/B/val_B.npy" % root)).as_posix()
#         self.files_A = np.load(root_A).astype(np.float32)
#         self.files_B = np.load(root_B).astype(np.float32)
#
#     def __getitem__(self, index):
#         item_A = self.transform(self.files_A[index % self.files_A.shape[0]])
#         if self.unaligned:
#             item_B = self.transform(self.files_B[random.randint(0, self.files_B.shape[0] - 1)])
#         else:
#             item_B = self.transform(self.files_B[index % self.files_B.shape[0]])
#         return {'A_img': item_A, 'B_img': item_B}
#     def __len__(self):
#         return max(self.files_A.shape[0], self.files_B.shape[0])


class ValGDataset(Dataset):
    def __init__(self, root, count=None, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.root_path = root
        self.unaligned = unaligned
        self.images = sorted(os.listdir(root))

    def __getitem__(self, index):
        img_path = os.path.join(self.root_path, self.images[index % len(self.images)])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.float32)
        img = img / 255.
        item = self.transform(img)
        return {'images': item}

    def __len__(self):
        return len(self.images)


# class TestDataset(Dataset):
#     def __init__(self, root, count=None, transforms_=None, unaligned=False):
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned
#         root_A = Path(os.path.join(os.getcwd(), "%s/A/test_A.npy" % root)).as_posix()
#         root_B = Path(os.path.join(os.getcwd(), "%s/B/test_B.npy" % root)).as_posix()
#         self.files_A = np.load(root_A).astype(np.float32)
#         self.files_B = np.load(root_B).astype(np.float32)
#
#     def __getitem__(self, index):
#         item_A = self.transform(self.files_A[index % self.files_A.shape[0]])
#         if self.unaligned:
#             item_B = self.transform(self.files_B[random.randint(0, self.files_B.shape[0] - 1)])
#         else:
#             item_B = self.transform(self.files_B[index % self.files_B.shape[0]])
#         return {'A_img': item_A, 'B_img': item_B}
#
#     def __len__(self):
#         return max(self.files_A.shape[0], self.files_B.shape[0])


################### RGB #######################
class ImageDataset(Dataset):
    def __init__(self, root,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned

    def __getitem__(self, index):
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img tranfsorms

        item_A = self.transform1(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        #random.seed(seed)
        if self.unaligned:
            item_B = self.transform2(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))

        else:
            item_B = self.transform2(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A_img': item_A, 'B_img': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))

    def __getitem__(self, index):
        img_test = np.load(self.files_A[index % len(self.files_A)])
        item_A = self.transform(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        if self.unaligned:
            item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A_img': item_A, 'B_img': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == '__main__':
    root = 'data/train_592_test_258/val2D/'
    files_A = sorted(glob.glob("%s/A_img/*" % root))
    print(files_A)
