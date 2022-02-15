import torch
import os
import PIL
from ncsnv2.datasets.vision import VisionDataset
from ncsnv2.datasets.utils import download_file_from_google_drive, check_integrity

class CelebA(VisionDataset):
    base_folder = "celeba"
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt")
    ]

    def __init__(self, root,
                 split="train",
                 target_type="attr",
                 transform=None, target_transform=None,
                 download=False):
        import pandas
        super().__init__(root)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.transform = transform
        self.target_transform = target_transform

        if split.lower() == "train":
            split = 0
        elif split.lower() == "valid":
            split = 1
        elif split.lower() == "test":
            split = 2
        else:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="valid" or split="test"')

        with open(os.path.join(self.root, self.base_folder, "list_eval_partition.txt"), "r") as f:
            splits = pandas.read_csv(f, delim_whitespace=True, header=None, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "identity_CelebA.txt"), "r") as f:
            self.identity = pandas.read_csv(f, delim_whitespace=True, header=None, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "list_bbox_celeba.txt"), "r") as f:
            self.bbox = pandas.read_csv(f, delim_whitespace=True, header=1, index_col=0)

        with open(os.path.join(self.root, self.base_folder, "list_landmarks_align_celeba.txt"), "r") as f:
            self.landmarks_align = pandas.read_csv(f, delim_whitespace=True, header=1)

        with open(os.path.join(self.root, self.base_folder, "list_attr_celeba.txt"), "r") as f:
            self.attr = pandas.read_csv(f, delim_whitespace=True, header=1)

        mask = (splits[1] == split)
        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(self.identity[mask].values)
        self.bbox = torch.as_tensor(self.bbox[mask].values)
        self.landmarks_align = torch.as_tensor(self.landmarks_align[mask].values)
        self.attr = torch.as_tensor(self.attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)

            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        if self.transform is not None:
            X = self.transform(X)
        
        target = index

        return X, target

    def __len__(self):
        return len(self.attr)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
