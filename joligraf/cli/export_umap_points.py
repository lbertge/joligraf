import click
from pyfiglet import figlet_format
from termcolor import colored
import pathlib
from PIL import Image
import umap
import re
import os
import numpy as np
import json

import warnings

warnings.filterwarnings("ignore")


def scale(x, out_range=(-1, 1)):
    domain_min, domain_max = np.min(x), np.max(x)
    a, b = out_range
    return a + ((x - domain_min) * (b - a)) / (domain_max - domain_min)


@click.command()
@click.argument("images_folder", type=click.Path(exists=True))
@click.option("--limit", type=int, default=1000)
@click.option("--out", type=str, default="umap_points.json")
def main(images_folder, limit, out):
    print(colored(figlet_format("UMAP Export", font="doom"), "cyan"))
    images_folder = pathlib.Path(images_folder)
    image_regex = re.compile(r".*\.(jpg|png|gif)$")

    list_img_files = [
        filename
        for filename in os.listdir(images_folder.resolve())
        if image_regex.match(filename) is not None
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

        print(colored("Ranges", "white", attrs=["underline"]))
        print(f"X: [{embeddings[:,0].min()},{embeddings[:,0].max()}]")
        print(f"Y: [{embeddings[:,1].min()},{embeddings[:,1].max()}]")

        embeddings[:, 0] = scale(embeddings[:, 0])
        embeddings[:, 1] = scale(embeddings[:, 1])

        print(colored("New Ranges", "white", attrs=["underline"]))
        print(f"X: [{embeddings[:,0].min()},{embeddings[:,0].max()}]")
        print(f"Y: [{embeddings[:,1].min()},{embeddings[:,1].max()}]")

        to_export = []
        for filename, position in zip(list_img_files[:limit], embeddings):
            to_export.append(
                {"image": filename, "x": float(position[0]), "y": float(position[1])}
            )

        with open(out, "w") as umap_result_file:
            json.dump(to_export, umap_result_file)


if __name__ == "__main__":
    main()
