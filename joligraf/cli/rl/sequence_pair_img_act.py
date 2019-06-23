from pathlib import Path

import click
import holoviews as hv
import numpy as np
from bokeh.layouts import column
from bokeh.io import show, output_file
from PIL import Image
from tqdm import tqdm
import pandas as pd
from pprint import pformat

hv.extension("bokeh")


@click.command()
@click.option("--imgs", "image_folder", type=click.Path(exists=True))
@click.option("--npz", "npz_filepath", type=click.Path(exists=True))
@click.option("--limit", type=int, default=42)
@click.option("--start", type=int, default=0)
def main(image_folder: str, npz_filepath: str, limit: int, start: int):
    data = np.load(npz_filepath)
    subfiles_name = list(filter(lambda x: "action" in x, data.files))
    datasource = {key: data[key] for key in subfiles_name}
    img_folder_path = Path(image_folder)
    images_filepaths = list(img_folder_path.glob("*.png"))
    images_filepaths.sort()

    limit = min(start + limit, len(images_filepaths) - 1)

    widgets = []
    with_slides = {}
    for idx, filepath in enumerate(tqdm(images_filepaths[start:limit])):
        dataframe = dict([(key, str(value[idx])) for key, value in datasource.items()])

        image = hv.RGB.load_image(str(filepath))
        image.opts(title=f"Frame #{idx+1}")
        table = hv.Div(pformat(dataframe).replace("\n", "</br>"))
        fig = image + table
        with_slides[idx] = fig
        widgets.append(hv.render(fig))

    data.close()
    hmap = hv.HoloMap(with_slides, "frame")
    hmap = hmap.collate()

    hv.save(hmap, "holomap.html")
    show(column(*widgets))


if __name__ == "__main__":
    main()
