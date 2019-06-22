import click
from pathlib import Path
import os


@click.command()
@click.argument("dataset_directory", type=click.Path(exists=True))
@click.option("-r", "--recursive", is_flag=True, default=False)
def main(dataset_directory: str, recursive: bool):
    dataset_path: Path = Path(dataset_directory)
    listfiles = list(dataset_path.glob("**/*.mp4" if recursive else "*.mp4"))

    for filename_path in listfiles:
        print(filename_path)
        parent = filename_path.parent
        filename = str(filename_path)
        output_dir = parent.joinpath("imgs")
        os.makedirs(output_dir, exist_ok=True)
        pattern = str(output_dir.resolve()) + "/img_%04d.png"
        os.popen(f"ffmpeg -i {filename} {pattern} -hide_banner")


if __name__ == "__main__":
    main()
