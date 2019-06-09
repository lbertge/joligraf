from PIL import Image
import numpy as np
import click
import json
from pathlib import Path
from resizeimage.resizeimage import resize_thumbnail

CANVAS_RESOLUTION = (2560, 1440)
THUMBNAIL_SIZE = (100, 100)
PADDING = 200


def scale(x, in_range=(-1, 1), out_range=(-1, 1)):
    if in_range is None:
        domain_min, domain_max = np.min(x), np.max(x)
    else:
        domain_min, domain_max = in_range
    a, b = out_range
    return a + ((x - domain_min) * (b - a)) / (domain_max - domain_min)


@click.command()
@click.argument("images_folder", type=click.Path(exists=True))
@click.argument("data_json_file", type=click.Path(exists=True))
def main(images_folder, data_json_file):
    with open(data_json_file, "r") as json_file:
        points_projection = json.load(json_file)

    canvas = Image.new("RGBA", CANVAS_RESOLUTION)

    for point_properties in points_projection:
        pos_x = scale(
            float(point_properties["x"]),
            out_range=(PADDING, CANVAS_RESOLUTION[0] - PADDING),
        )
        pos_x = int(pos_x)
        pos_y = scale(
            float(point_properties["y"]),
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
