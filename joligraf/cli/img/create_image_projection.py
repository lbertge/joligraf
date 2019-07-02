import json

import click
from PIL import Image
from pyfiglet import figlet_format
from termcolor import colored
import numpy as np

from joligraf.cli.crossfiles_utils_functions import scale

CANVAS_RESOLUTION = (2560, 1440)
THUMBNAIL_SIZE = (100, 100)
PADDING = 200


def original_projection(points_projection: list) -> Image:
    canvas = Image.new("RGBA", CANVAS_RESOLUTION)

    for point_properties in points_projection:
        pos_x = scale(
            float(point_properties["x"]),
            in_range=(0, 1),
            out_range=(PADDING, CANVAS_RESOLUTION[0] - PADDING),
        )
        pos_x = int(pos_x)
        pos_y = scale(
            float(point_properties["y"]),
            in_range=(0, 1),
            out_range=(PADDING, CANVAS_RESOLUTION[1] - PADDING),
        )
        pos_y = int(pos_y)
        img_source = str(point_properties["image"])
        img: Image = Image.open(img_source)
        thumbnail = img.resize(THUMBNAIL_SIZE, Image.ANTIALIAS)

        canvas.paste(thumbnail, (pos_x, pos_y))

    return canvas


def grid_projection(points_projection: list) -> Image:
    size = int(np.sqrt(len(points_projection)))
    canvas = Image.new("RGBA", (size * THUMBNAIL_SIZE[0], size * THUMBNAIL_SIZE[0]))

    for point_properties in points_projection:
        pos_x = float(point_properties["x"])
        pos_y = float(point_properties["y"])

        pos_x *= (size - 1) * THUMBNAIL_SIZE[0]
        pos_y *= (size - 1) * THUMBNAIL_SIZE[0]

        pos_x = int(pos_x)
        pos_y = int(pos_y)

        img_source = str(point_properties["image"])
        img: Image = Image.open(img_source)
        thumbnail = img.resize(THUMBNAIL_SIZE, Image.ANTIALIAS)

        canvas.paste(thumbnail, (pos_x, pos_y))

    return canvas


@click.command()
@click.argument("data_json_file", type=click.Path(exists=True))
@click.option("--grid", "grid", is_flag=True, help="Flag to project on grid")
def main(data_json_file: str, grid: bool):
    """Project the images on a big image using the position in the json file 
    
    Parameters
    ----------
    data_json_file : str
        Path of the json file containing coordinate and image paths
    """
    print(colored(figlet_format("IMG projection", font="standard"), "cyan"))
    with open(data_json_file, "r") as json_file:
        points_projection = json.load(json_file)

    if grid:
        canvas = grid_projection(points_projection)
    else:
        canvas = original_projection(points_projection)

    canvas.save("projection.png")


if __name__ == "__main__":
    main()
