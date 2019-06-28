import json
from pathlib import Path
from typing import Optional

import click
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div
from bokeh.plotting import figure, output_file, show
from pyfiglet import figlet_format
from termcolor import colored

TOOLS = "crosshair,pan,wheel_zoom,zoom_in,zoom_out,reset,tap,save,box_zoom,box_select"


def display_event(div, root_image, source, block_height):
    code = """
    console.log(cb_obj.indices)
    var indexes = cb_obj.indices
    var htmlcode = ""
    htmlcode += "<div style='display: block; overflow: scroll; height: {}px'>"
    indexes.forEach(function (index, i){{
        var image_path = data[index]['image']
        var descriptif = "Index:"+index+"| Filename "+ data[index]['image']
        var htmlimg = "<img src='"+image_path+"' style='max-width: 200px' />"
        htmlcode += descriptif
        htmlcode += "<br />"
        htmlcode += htmlimg
        htmlcode += "<br />"

    }})
    htmlcode += "</div>"
    div.text = htmlcode
    """.format(
        block_height
    )
    return CustomJS(
        args=dict(div=div, root_image=root_image + "/", data=source), code=code
    )


@click.command()
@click.argument("images_folder", type=click.Path(exists=True))
@click.argument("data_json_file", type=click.Path(exists=True))
@click.option(
    "--img_ref",
    type=click.Path(exists=True),
    default=None,
    help="Path of the image version of the projection",
)
@click.option(
    "--out", type=str, default="scatter_projection.html", help="Output html file"
)
def main(images_folder: str, data_json_file: str, img_ref: Optional[str], out: str):
    """Project data point into interactive scatter plot

    Arguments:
        images_folder {str} -- Path of the images directory
        data_json_file {str} -- Path of the corresponding json file
        img_ref {Optional[str]} -- Path of the image version of the projection
                                   (see create_image_projection.py script)
        out {str} -- Output html file
    """

    print(colored(figlet_format("Scatter plot projection", font="standard"), "cyan"))
    assert out[-5:] == ".html", "Please give a html output file"
    images_folder = str(Path(images_folder).resolve())

    with open(data_json_file, "r") as json_file:
        points_projection = json.load(json_file)

    points = np.zeros((len(points_projection), 2))
    for i, point_properties in enumerate(points_projection):
        points[i] = [float(point_properties["x"]), float(point_properties["y"])]

    domain_min, domain_max = np.min(points[:, 1]), np.max(points[:, 1])
    domain = domain_max - domain_min

    s1 = ColumnDataSource(data=dict(x=points[:, 0], y=abs(points[:, 1] - domain)))

    p1 = figure(tools=TOOLS, plot_width=1280)
    p1.circle("x", "y", source=s1, size=10)

    div = Div(width=400, height=p1.plot_height)
    layout = row(p1, div)

    ###
    callback = display_event(
        div,
        root_image=images_folder,
        source=points_projection,
        block_height=p1.plot_height,
    )

    s1.selected.js_on_change("indices", callback)

    ###
    if img_ref:
        div_img_ref = Div(width=1280)
        div_img_ref.text = f"<img style='width:1280px' src='{img_ref}' / >"
        layout = column(layout, div_img_ref)

    output_file(out, title="scatter projection")

    show(layout)


if __name__ == "__main__":
    main()
