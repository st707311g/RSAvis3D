from __future__ import annotations

import logging
import tomllib
import warnings
from pathlib import Path
from typing import Annotated, Literal

import coloredlogs
import typer

from st_modules.rsa_dataset import RSA_Dataset, walk_to_find_rsa_dataset
from st_modules.RSAvis3D import RSAvis3D

warnings.simplefilter("ignore")
logger = logging.getLogger(Path(__file__).name)


def get_version_from_toml(path: str | Path) -> str:
    toml_path = Path(path)

    if not toml_path.exists():
        return f"Error: {toml_path} does not exist."

    with open(toml_path, "rb") as f:
        try:
            data = tomllib.load(f)

            if "project" in data and "version" in data["project"]:
                return data["project"]["version"]

            if "tool" in data and "poetry" in data["tool"]:
                return data["tool"]["poetry"]["version"]

            return "Version not found in pyproject.toml"

        except tomllib.TOMLDecodeError:
            return "Error: Invalid TOML format."


__version__ = get_version_from_toml(Path(__file__).with_name("pyproject.toml"))


def version_callback(value: bool):
    if value:
        typer.echo(f"Version: {__version__}")
        raise typer.Exit()


def main(
    search_dir: Annotated[
        Path,
        typer.Argument(help="Indicate search directory."),
    ],
    block_size: int = typer.Option(
        64,
        "--block-size",
        "-b",
        min=64,
        help="Indicate block size.",
    ),
    median_kernel_size: int = typer.Option(
        7,
        "--median-kernel-size",
        "-m",
        min=1,
        help="Indicate kernel size for 3D median filter.",
    ),
    cylinder_radius: int = typer.Option(
        0,
        "--cylinder-radius",
        "-c",
        min=0,
        help="Indicate radius of cylindrical cropping mask."
        " If 0, masking process will be skipped.",
    ),
    edge_size: int = typer.Option(
        21,
        "--edge-size",
        "-e",
        min=1,
        help="Indicate for blur kernel size for edge detection.",
    ),
    format: Literal["png", "tif", "jpg"] = typer.Option(
        "jpg",
        "--format",
        "-f",
        help="Indicate format type of saving image file.",
    ),
    intensity_factor: float = typer.Option(
        10.0,
        "--intensity-factor",
        "-i",
        min=0.0000001,
        help="Indicate intensity factor (> 0)."
        " Image intensity will be multiplied by this factor.",
    ),
    series_src: str = typer.Option(
        "ct",
        "--series-src",
        help="Indicate series name in the RSA dataset, which will be processed.",
    ),
    series_dst: str = typer.Option(
        "rsavis3d",
        "--series-dst",
        help="Indicate series name in the RSA dataset, which will be created.",
    ),
    debug_mode: bool = typer.Option(
        False,
        "--debug",
        "-D",
        help="Launch in debug mode.",
    ),
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    if debug_mode:
        coloredlogs.install(level=logging.DEBUG, logger=logger)
    else:
        coloredlogs.install(level=logging.INFO, logger=logger)

    if not search_dir.is_dir():
        logger.error("Indicate valid path.")
        exit()

    for dataset, skipped in walk_to_find_rsa_dataset(
        search_dir, [series_src], [series_dst]
    ):
        relative_path = dataset._config.path.relative_to(search_dir)
        logger.info(f"{relative_path=}")

        if skipped:
            logger.info(f"skipped: {relative_path}")
            continue

        proceed(
            dataset=dataset,
            relative_path=relative_path,
            series_src=series_src,
            series_dst=series_dst,
            format=format,
            block_size=block_size,
            median_kernel_size=median_kernel_size,
            edge_size=edge_size,
            cylinder_radius=cylinder_radius,
            intensity_factor=intensity_factor,
        )


def proceed(
    dataset: RSA_Dataset,
    relative_path: Path,
    series_src: str,
    series_dst: str,
    format: str,
    block_size: int,
    median_kernel_size: int,
    edge_size: int,
    cylinder_radius: int,
    intensity_factor: float,
):
    dst_path = dataset.create_new_series(series_dst, ".zip")

    overlap = (median_kernel_size + 1) // 2

    rsavis3d = RSAvis3D(
        src_path=dataset.get_series_path(series_src),
        dst_path=dst_path,
        dst_file_type=format,
        block_size=block_size,
        overlap=overlap,
        label=str(relative_path),
    )

    rsavis3d.run(
        median_kernel_size=median_kernel_size,
        edge_size=edge_size,
        cylinder_radius=cylinder_radius,
        intensity_factor=intensity_factor,
    )

    dataset.update_log(
        {
            dst_path.stem: {
                "median_kernel_size": median_kernel_size,
                "edge_size": edge_size,
                "cylinder_radius": cylinder_radius,
                "intensity_factor": intensity_factor,
            }
        }
    )


if __name__ == "__main__":
    typer.run(main)
