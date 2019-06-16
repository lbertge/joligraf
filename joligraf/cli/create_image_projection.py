import json
from pathlib import Path

import click
from PIL import Image
from pyfiglet import figlet_format
from resizeimage.resizeimage import resize_thumbnail
from termcolor import colored

from joligraf.cli.crossfiles_utils_functions import scale

CANVAS_RESOLUTION = (2560, 1440)
THUMBNAIL_SIZE = (100, 100)
PADDING = 200


@click.command()
@click.argument("images_folder", type=click.Path(exists=True))
@click.argument("data_json_file", type=click.Path(exists=True))
def main(images_folder: str, data_json_file: str):
    """Project the images on a big image using the position in the json file 
    
    Arguments:
        images_folder {str} -- Path of the images directory
        data_json_file {str} -- Path of the corresponding json file
    """
    print(colored(figlet_format("IMG projection", font="standard"), "cyan"))
    with open(data_json_file, "r") as json_file:
        points_projection = json.load(json_file)

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
        img_source = Path(images_folder).joinpath(point_properties["image"])
        img: Image = Image.open(str(img_source))
        thumbnail = resize_thumbnail(img, THUMBNAIL_SIZE, Image.ANTIALIAS)

        canvas.paste(thumbnail, (pos_x, pos_y))

    canvas.save("projection.png")


if __name__ == "__main__":
    main()
