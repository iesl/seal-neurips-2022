from typing import Dict, Iterable, List
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.data.fields import ArrayField
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
import os, pickle, torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF

def thirty_six_crop(img, size=24):
    """ Crop the given PIL Image 32x32 into 36 crops of 24x24
    Inspired from five_crop implementation in pytorch
    https://pytorch.org/docs/master/_modules/torchvision/transforms/functional.html
    """
    crops = []
    for h in [0, 2, 3, 4, 6, 8]:
        for w in [0, 2, 3, 4, 6, 8]:
            crops.append(img.crop((w, h, w+size, h+size))) # notice that w comes first
    return crops

@DatasetReader.register("weizmann-horse-seg")
class WeizmannHorseSegReader(DatasetReader):
    def __init__(self, cropping=None):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(32, 32))])
        self.cropping = cropping # random or thirty_six or None

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        :param file_path: data directory, eg. data/weizmann_horse/weizmann_horse_train.npy
        """

        data = np.load(file_path, allow_pickle=True)

        for raw_image, mask in zip(data[0]['images'], data[0]['masks']):
            assert raw_image.shape == (32, 32, 3)
            assert mask.shape == (32, 32)
            raw_image = self.transform(raw_image) # This will scale from 0-255 to 0-1
            mask = self.transform(np.expand_dims(mask*255, axis=2))

            if self.cropping == "thirty_six":
                image = thirty_six_crop(raw_image)
                transform_test = transforms.Compose(
                    [transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))]
                )
                image = transform_test(image)
            elif self.cropping == "random":
                i, j, h, w = transforms.RandomCrop.get_params(raw_image, output_size=(24, 24))
                image = TF.to_tensor(TF.crop(raw_image, i, j, h, w)) # (3, 24, 24)
                mask = TF.crop(mask, i, j, h, w)
            else:
                image = TF.to_tensor(raw_image)

            raw_image = TF.to_tensor(raw_image)
            mask = TF.to_tensor(mask) >= 0.5 # make sure we binarize to integers 0 and 1

            yield Instance({
                'raw_image': ArrayField(raw_image), # (3, 32, 32)
                'image': ArrayField(image), # random (3, 24, 24); None (3, 32, 32), thirty_six (36, 3, 24, 24)
                'mask': ArrayField(mask.float()), # random (1, 24, 24); None (1, 32, 32), thirty_six (1, 32, 32)
            })

