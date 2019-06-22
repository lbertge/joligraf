from pathlib import Path
import click
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from datetime import datetime
from collections import Counter

from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models.widgets import Div
from typing import Callable


def convert_string_metadata_type(metadata: dict) -> dict:
    metadata = metadata.copy()
    for key in ["width", "height"]:
        metadata[key] = int(metadata[key])

    time_format = "%H:%M:%S"
    if "." in metadata["duration"]:
        time_format += ".%f"
    metadata["duration"] = datetime.strptime(metadata["duration"], time_format)

    return metadata


def compute_stats(metadatas: list) -> dict:
    resolution_counter = Counter()
    duration_counter = Counter()

    for metadata in metadatas:
        resolution_counter[f"{metadata['width']}x{metadata['height']}"] += 1
        duration_rounded = metadata["duration"].minute
        duration_rounded += 1
        duration_counter[f"{duration_rounded-1} <= x < {duration_rounded}"] += 1

    # Sort entries
    resolution_counter = sorting(dict(resolution_counter), sorting_resolution_key)
    duration_counter = sorting(dict(duration_counter), sorting_interval_key)

    return {"resolution": resolution_counter, "duration": duration_counter}


def sorting(mydict: dict, sort_key: Callable) -> dict:
    tuples_ = tuple(mydict.items())
    tuples_sorted = sorted(tuples_, key=sort_key)
    return dict(tuples_sorted)


def sorting_resolution_key(key: tuple) -> int:
    return int(key[0].split("x")[0])


def sorting_interval_key(key: tuple) -> int:
    return int(key[0].split("<=")[0])


def create_hbar_figure(datasource: dict, title):
    y = list(datasource.keys())
    counts = list(datasource.values())
    source = ColumnDataSource(data=dict(y=y, x=counts))

    tooltips = [("count", "@x")]

    fig = figure(
        y_range=y,
        title=title,
        plot_height=1000,
        tooltips=tooltips,
        sizing_mode="stretch_both",
    )

    fig.hbar(y="y", right="x", height=0.5, source=source)
    fig.xaxis.axis_label = "counts"

    return fig


@click.command()
@click.argument("dataset_directory", type=click.Path(exists=True))
@click.option("-r", "--recursive", is_flag=True, default=False)
def main(dataset_directory: str, recursive: bool):
    """Look for mp4 videos

    Parameters
    ----------
    dataset_directory : str
        Directory to search for videos
    recursive : bool
        Operate on files and directories recursively
    """
    dataset_path: Path = Path(dataset_directory)
    listfiles = list(dataset_path.glob("**/*.mp4" if recursive else "*.mp4"))

    all_metadatas: list = []

    for filename_path in listfiles:
        filename = str(filename_path)
        parser = createParser(filename)
        if not parser:
            print(f"Unable to parse file {filename}")
            exit(1)

        with parser:
            try:
                metadata = extractMetadata(parser).exportDictionary(human=False)
            except Exception as err:
                print(f"Metadata extraction error: {err}")
                metadata = None

        metadata = metadata["Metadata"]
        metadata = convert_string_metadata_type(metadata)
        all_metadatas += [metadata]

    data = compute_stats(all_metadatas)

    html_title = Div(
        text=(
            "<h1>Metadata statistic</h1></br><b>Considered directory:</b> {}</br>"
            "<b>Total of videos found:</b> {}"
        ).format(dataset_path.resolve(), len(listfiles))
    )
    # Create figures
    figures = [
        create_hbar_figure(data["resolution"], "Resolution counts"),
        create_hbar_figure(data["duration"], "Duration counts"),
    ]

    # Adapt legend
    figures[0].yaxis.axis_label = "width x height"
    figures[1].yaxis.axis_label = "duration interval (min)"

    allwidgets = [html_title] + figures

    show(column(*allwidgets))


if __name__ == "__main__":
    main()
