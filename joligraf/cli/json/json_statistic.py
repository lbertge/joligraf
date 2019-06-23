import json
from pathlib import Path
from pprint import pformat

import click
import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from scipy.stats import describe

hv.extension("bokeh")
np.set_printoptions(threshold=5)


@click.command()
@click.argument("json_path", type=click.Path(exists=True))
@click.option("--type", "type_", type=click.Choice(["f", "d"]), default="f")
@click.option("--out", type=str, default=f"json_statistic.html")
def main(json_path: str, type_: str, out: str):
    assert out[-5:] == ".html"

    if type_ == "f":
        assert json_path[-5:] == ".json"
        list_json_paths = [Path(json_path)]
    else:
        list_json_paths = list(Path(json_path).glob("**/*.json"))

    all_data: list = []
    for pathfile in list_json_paths:
        with open(pathfile) as json_file:
            data = json.load(json_file)
            dictionary = {}
            for k, v in data.items():
                if type(v) == int or type(v) == float:
                    dictionary[k] = v

                elif type(v) == bool:
                    dictionary[k] = int(v)
            all_data += [dictionary]

    df = pd.DataFrame(all_data)
    widgets: list = []

    for column_name in df.columns:
        data = df[column_name].values
        frequencies, edges = np.histogram(data)
        hist = hv.Histogram((edges, frequencies))
        hist.opts(tools=["hover"], title=column_name, width=600)

        annotation = (
            "<h2>{}</h2>"
            "<b>Overview</b></br>{}</br>{}</br>"
            "<b>Describe</b></br>{}</br>"
        ).format(
            column_name,
            data.shape,
            np.array_repr(data),
            pformat(dict(describe(data)._asdict())),
        )
        annotation = annotation.replace("\n", "</br>")
        html_annotation = Div(text=annotation)

        widgets.append(row(hv.render(hist), html_annotation))

    layout = column(*widgets)

    output_file(out, title=out)
    save(layout)


if __name__ == "__main__":
    main()
