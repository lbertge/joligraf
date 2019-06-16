import json
import os
from pathlib import Path
import re
import warnings

import click
import numpy as np
import umap
from PIL import Image
from pyfiglet import figlet_format
from termcolor import colored

from joligraf.cli.crossfiles_utils_functions import scale

warnings.filterwarnings("ignore")


@click.command()
@click.argument("images_directory", type=click.Path(exists=True))
@click.option("--limit", type=int, default=1000, help="Max number of images to project")
@click.option("--out", type=str, default="umap_points.json", help="Output file in json")
def main(images_directory: str, limit: int, out: str):
    """Return UMAP projection points from a folder containing images jpg, png gif

    Parameters
    ----------
    images_directory : str
        Path to the directory with images
    limit : int
        Max number of images to project
    out : str
        JSON output file
    """
    print(colored(figlet_format("UMAP Export", font="doom"), "cyan"))
    assert out[-5:] == ".json", "Please give a json output file"
    images_folder: Path = Path(images_directory)
    image_regex = re.compile(r".*\.(jpg|jpeg|png|gif)$")

    list_img_files = [
        filename
        for filename in os.listdir(images_folder.resolve())
        if image_regex.match(filename.lower()) is not None
    ]

    list_img_files.sort()

    data = []
    min_vector_size = np.inf
    max_vector_size = -np.inf
    for filename in list_img_files[:limit]:
        img = Image.open(images_folder.joinpath(filename))
        img = img.resize((img.size[0] // 3, img.size[1] // 3))
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[:, :, :3]

        img = img.flatten()
        if img.size < min_vector_size:
            min_vector_size = img.size

        if img.size > max_vector_size:
            max_vector_size = img.size

        data.append(img)

    print(colored("Stats", "white", attrs=["underline"]))
    print(f"Min vector size: {min_vector_size}")
    print(f"Max vector size: {max_vector_size}")

    if min_vector_size == max_vector_size:
        print(colored("Dataset is uniform, proceed ..."))

        data = np.vstack(data)
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="correlation")
        embeddings = reducer.fit_transform(data)

        embeddings[:, 0] = scale(embeddings[:, 0])
        embeddings[:, 1] = scale(embeddings[:, 1])

        to_export = []
        for filename, position in zip(list_img_files[:limit], embeddings):
            to_export.append(
                {"image": filename, "x": float(position[0]), "y": float(position[1])}
            )

        with open(out, "w") as umap_result_file:
            json.dump(to_export, umap_result_file)


if __name__ == "__main__":
    main()
