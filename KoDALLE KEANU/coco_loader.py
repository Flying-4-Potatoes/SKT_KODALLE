from random import randint, choice
from pathlib import Path
from typing import Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from transformers import AutoTokenizer
from preprocess import remove_style, remove_subj

import json
#import matplotlib.pyplot as plt

# with open("./data/coco/MSCOCO_train_val_Korean.json", "r", encoding="utf-8") as jsonco:
#     kcoco = json.load(jsonco)
#     print(kcoco[0]['file_path'].split('/')[1].split('.')[0])
#     print('\n'.join(kcoco[0]['caption_ko']))
#     image = Image.open("./data/coco/"+kcoco[0]['file_path'])
#     image.show()
    

class TextImageDataset(Dataset):
    def __init__(
        self,
        text_file: str,
        image_folder: str,
        text_len: int,
        image_size: int,
        truncate_captions: bool,
        resize_ratio: float,
        tokenizer: AutoTokenizer = None,
        shuffle: bool = False,
    ) -> None:
        super(TextImageDataset, self).__init__()
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        
        with open(text_file, "r", encoding='utf-8') as f:
            text_dict = json.load(f)
        text_dict = {text['file_path'].split('/')[1].split('.')[0]: text['caption_ko'] for text in text_dict}
        
        image_path = Path(image_folder)
        image_files = [
            *image_path.glob("**/*[0-9].png"),
            *image_path.glob("**/*[0-9].jpg"),
            *image_path.glob("**/*[0-9].jpeg"),
        ]
        image_files = {image_file.stem: image_file for image_file in image_files}
        
        keys = image_files.keys() & text_dict.keys()
        self.keys = list(keys)
        self.text_dict = {k: v for k, v in text_dict.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
            ]
        )
        
        # re.match
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        key = self.keys[index]
        descriptions = self.text_dict[key]     
        image_file = self.image_files[key]

        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {index}")
            return self.skip_sample(index)
        
        # ADD PREPROCESSING FUNCTION HERE
        encoded_dict = self.tokenizer(
            description,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_len,
            add_special_tokens=True,
            return_token_type_ids=False,  # for RoBERTa
        )
        
        flattened_dict = {i: v.squeeze() for i, v in encoded_dict.items()}
        input_ids = flattened_dict["input_ids"]
        attention_mask = flattened_dict["attention_mask"]
        
        try:
            image_tensor = self.image_transform(Image.open(image_file))
        except(UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occured trying to load file {image_file}.")
            print(f"Skipping index {index}")
            return self.skip_sample(index)
        
        return input_ids, image_tensor, attention_mask
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))
    
    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)
    
    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    

class ImgDatasetExample(Dataset):
    
    def __init__(
        self, image_folder: str, image_transform: transforms.Compose = None,
    ) -> None:
        self.image_trainsform = image_transform
        
        self.image_path = Path(image_folder)
        self.image_files = [
            *self.image_path.glob("**/*.png"),
            *self.image_path.glob("**/*.jpg"),
            *self.image_path.glob("**/*.jpeg"),
        ]
        
    def __getitem__(self, index: int) -> torch.tensor:
        image = Image.open(self.image_files[index])

        if self.image_transform:
            image = self.image_transform(image)
        return torch.tensor(image)

    def __len__(self) -> int:
        return len(self.image_files)