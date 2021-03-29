import numpy as np
import torch
import torch.nn as nn
import argparse
from torchvision.datasets.coco import CocoCaptions
from torchvision import transforms


def get_coco(data_path):
    """
    Get coco dataset from local data path.
    Note: data must already be downloaded and available 
    on local disk.
    Args:
        data_path ():

    Returns:

    """
    annotation_file_path = data_path + '/captions_val2014_fakecap_results.json'
    captions = CocoCaptions(root = data_path,
                            annFile=annotation_file_path,
                            transform=transforms.ToTensor())

    print('Number of samples: ', len(captions))
    img, target = captions[3]  # load 4th sample

    print("Image Size: ", img.size())
    print(target)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Size of mini-batch')
    parser.add_argument('-d', '--data_path', type=str, default= "../../../data")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    # TODO: set up dataloader, etc.
    yee = get_coco(args.data_path)


if __name__ == "__main__":
    main()
