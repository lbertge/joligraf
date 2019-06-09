import numpy as np
import click
import json
from bokeh.plotting import figure, output_file, show
from bokeh.models import (
    CustomJS,
    Div,
    TapTool,
    HoverTool,
    BoxSelectTool,
    ColumnDataSource,
)
from bokeh.layouts import row, column
from bokeh import events
from pathlib import Path
from PIL import Image

TOOLS = (
    "crosshair,hover,pan,wheel_zoom,zoom_in,zoom_out,reset,tap,save,box_zoom,box_select"
)


def display_event(div, root_image, source, block_height):
    code = """
    console.log(cb_obj.indices)
    var indexes = cb_obj.indices
    var htmlcode = ""
    htmlcode += "<div style='display: block; overflow: scroll; height: {}px'>"
    indexes.forEach(function (index, i){{
        var image_path = root_image+data[index]['image']
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
@click.option("--img_ref", type=click.Path(exists=True), default=None)
def main(images_folder, data_json_file, img_ref):
    images_folder = Path(images_folder).resolve()
    images_folder = str(images_folder)

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

    output_file("scatter.html", title="scatter projection")

    show(layout)


if __name__ == "__main__":
    main()
