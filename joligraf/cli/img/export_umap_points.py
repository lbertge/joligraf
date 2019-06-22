import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Callable

import click
import numpy as np
import umap
from PIL import Image
from pyfiglet import figlet_format
from scipy.spatial.distance import cdist
from termcolor import colored
from torchvision import models, transforms
from lapjv import lapjv

from joligraf.cli.crossfiles_utils_functions import scale

warnings.filterwarnings("ignore")


class Pretrained:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.model_conv = models.resnet18(pretrained=True).eval().to("cuda")

    def get_resnet_features(self, img: np.ndarray):
        img_tensor = self.transform(img)[None, :]
        feat = self.model_conv(img_tensor.to("cuda"))
        feat = feat[0].detach().cpu().numpy()
        return feat


def get_raw_pixel(img: np.ndarray) -> np.ndarray:
    return img.flatten()


def compute_features(
    images_folder: Path, list_img_files: list, limit: int, features_fn: Callable
):
    data = []
    min_vector_size = np.inf
    max_vector_size = -np.inf
    for filename in list_img_files[:limit]:
        img = Image.open(images_folder.joinpath(filename))
        img = img.resize((img.size[0] // 3, img.size[1] // 3))
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[:, :, :3]

        features = features_fn(img)
        if features.size < min_vector_size:
            min_vector_size = features.size

        if features.size > max_vector_size:
            max_vector_size = features.size

        data.append(features)

    return data, min_vector_size, max_vector_size


def get_list_img_filenames(
    images_folder: Path, recursive: bool = False, shuffle: bool = True
) -> list:
    image_regex = re.compile(r".*\.(jpg|jpeg|png|gif)$")

    list_img_files: list = []
    for root, dirnames, filenames in os.walk(images_folder):
        subset = list(
            filter(lambda filename: image_regex.match(filename.lower()), filenames)
        )
        subset = list(map(lambda filename: root + "/" + filename, subset))
        list_img_files += subset
        if not recursive:
            break

    list_img_files.sort()

    if shuffle:
        random.shuffle(list_img_files)

    return list_img_files


def features_fn_factory(type_: str) -> Callable:
    features_fn = {
        "pixel": get_raw_pixel,
        "pretrained": Pretrained().get_resnet_features,
    }
    return features_fn[type_]


def jonker_volgenant_projection(embeddings) -> Image:
    size = int(np.floor(np.sqrt(len(embeddings))))
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)))
    grid = grid.reshape(-1, 2)
    cost_matrix = cdist(grid, embeddings[: size ** 2], "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())

    _, col_ind, _ = lapjv(cost_matrix)

    grid_jv = grid[col_ind]
    return grid_jv


# TODO device choice
@click.command()
@click.argument("images_directory", type=click.Path(exists=True))
@click.option(
    "-R",
    "--recursive",
    type=bool,
    is_flag=True,
    help="Flag to search images recurvisely",
)
@click.option("--limit", type=int, default=1024, help="Max number of images to project")
@click.option("--out", type=str, default="umap_points.json", help="Output file in json")
@click.option(
    "--type",
    "representation",
    type=click.Choice(["pixel", "pretrained"]),
    default="pixel",
    help="Features representation",
)
@click.option(
    "--jv",
    "jonker_volgenant",
    is_flag=True,
    help="Flag to project on grid with jonker volgenant algorithm",
)
def main(
    images_directory: str,
    recursive: bool,
    limit: int,
    out: str,
    representation: str,
    jonker_volgenant: bool,
):
    """
    Return UMAP projection points from a folder containing images jpg, png gif.

    Parameters
    ----------
    images_directory : str
        Path to the directory with images
    recursive: bool
        Flag to search image recursively
    limit : int
        Max number of images to project
    out : str
        JSON output file
    representation : str
        Features representation among 'pixel' and 'resnet'
    jonker_volgenant: bool
        Flag to project position on grid
        
    """
    print(colored(figlet_format("UMAP Export", font="doom"), "cyan"))
    assert out[-5:] == ".json", "Please give a json output file"
    images_folder: Path = Path(images_directory)

    list_img_files = get_list_img_filenames(images_folder, recursive)

    data, min_vector_size, max_vector_size = compute_features(
        images_folder, list_img_files, limit, features_fn_factory(representation)
    )

    print(colored("Stats", "white", attrs=["underline"]))
    print(f"Min vector size: {min_vector_size}")
    print(f"Max vector size: {max_vector_size}")

    if min_vector_size == max_vector_size:
        print(colored("Dataset is uniform, proceed ..."))

        data = np.vstack(data)
        reducer = umap.UMAP()
        embeddings = reducer.fit_transform(data)

        embeddings[:, 0] = scale(embeddings[:, 0])
        embeddings[:, 1] = scale(embeddings[:, 1])

        embeddings = jonker_volgenant_projection(embeddings)

        to_export = []

        for filename, position in zip(list_img_files[:limit], embeddings):
            to_export.append(
                {"image": filename, "x": float(position[0]), "y": float(position[1])}
            )

        with open(out, "w") as umap_result_file:
            json.dump(to_export, umap_result_file)


if __name__ == "__main__":
    main()
