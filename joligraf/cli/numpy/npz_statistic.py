import numpy as np
import click
from scipy.stats import describe
from joligraf.cli.bokeh_helper import create_vhist_figure
from bokeh.models.widgets import Div
from bokeh.layouts import column, row
from bokeh.io import show, output_file
from pprint import pformat
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")


@click.command()
@click.argument("npz_filepath", type=click.Path(exists=True))
@click.option("--out", type=str)
def main(npz_filepath, out):
    assert out[-5:] == ".html", "Please enter an html output file"
    data = np.load(npz_filepath)

    widgets = []
    widgets += [
        Div(
            text="<h1>NPZ statistic</h1><b>Subfiles found</b></br>{}".format(data.files)
        )
    ]
    output_file(out, title="NPZ statistic")
    for subfile in data.files:
        subdata = data[subfile].squeeze()
        array_repr = subdata.__str__()
        array_describe = pformat(dict(describe(subdata)._asdict()))
        if subdata.ndim == 1:

            hist, edges = np.histogram(subdata)

            hv_hist = hv.Histogram((edges, hist))
            hv_hist.opts(tools=["hover"], title=subfile + " histogram", width=600)
            fig = hv.render(hv_hist)

        elif subdata.ndim == 2:
            heatmap = hv.Image(subdata)
            heatmap.opts(
                colorbar=True, width=600, height=600, tools=["hover"], title=subfile
            )
            fig = hv.render(heatmap)

        else:
            fig = Div(text="To many dimension to visualize")

        annotation = (
            "<h2>{}</h2>"
            "<b>Overview</b></br>{}</br>{}</br>"
            "<b>Describe</b></br>{}</br>"
        ).format(subfile, subdata.shape, array_repr, array_describe)
        annotation = annotation.replace("\n", "</br>")

        html_annotation = Div(text=annotation)

        widgets += [row([fig, html_annotation])]

    data.close()
    show(column(*widgets))


if __name__ == "__main__":
    main()
