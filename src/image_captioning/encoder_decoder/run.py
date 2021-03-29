import numpy as np
import torch
import torch.nn as nn
import argparse
from torchvision.datasets.coco import CocoCaptions
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List


def collate_fn(data: List[Tuple]) -> Tuple:
    """
    Create mini-batch tensors from list of tuples
    We override the default collate_fn, because  
    mergging caption (including padding) is not supported
    by the default collate_fn.
    
    
    Args: 
        data: List of tuple (image, caption)
        - image: torch tensor of shape (3, 256, 256)
        - caption: torch tensor of shape (?); variable length

   Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256)
        targets: torch tensor of shape (batch_size, padded_length)
        lengths: list; valid length for each padded caption
    """
    # Sort in descending order by caption length. 
    # Images with longest captions come first
    data.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Unpack the data 
    images, captions = zip(*data)
    print(f"Image shape: {images.shape}")
    # TODO: create a tuple of tensors of size 3
    # We have images, now we want to go from n 3-dimensional images to a
    # single four-dimensional tensor 
    # We get (N, 3, 256, 256) images as a result
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensors to 2D tensors)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    # Fill empty tensor with values
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return images, targets, lengths


def get_coco(data_path: str, batch_size: int) -> DataLoader:
    """
    Get coco dataset from local data path.
    Note: data must already be downloaded and available 
    on local disk.
    Args:
        data_path ():

    Returns:

    """
    train_path = data_path + '/train2017'
    annotation_file_path = data_path + '/annotations/instances_train2017.json'
    coco_training_data = CocoCaptions(root = data_path,
                            annFile=annotation_file_path,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))
    # Create DataLoader object
    print('Number of samples: ', len(coco_training_data))
    print(f"Image size: {coco_training_data}")

    train_dataloader = DataLoader(coco_training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_dataloader


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Size of mini-batch')
    parser.add_argument('-d', '--data_path', type=str, default= "../../../data/ms-coco")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    # Get dataset from local device
    train_dataloader = get_coco(args.data_path, args.batch_size)

    for i, (images, captions, lengths) in enumerate(tqdm(train_dataloader)):
        print(f"Shapes: {images.shape}. Captions: {captions.shape}. Lengths: {lengths}")
        if i == 100:
            break


if __name__ == "__main__":
    main()
