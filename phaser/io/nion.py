
from pathlib import Path
import re
import typing as t

import numpy
import pane
import pane.io
from numpy.typing import NDArray
from pane.annotations import shape
from pane.convert import IntoConverterHandlers, from_data
from typing_extensions import Self

from phaser.types import IsVersion
from phaser.utils.image import apply_flips
from phaser.utils.num import to_numpy

import zipfile as zf
import io



def _get_dir(f: pane.io.FileOrPath) -> t.Optional[Path]:
    if isinstance(f, (str, Path)):
        return Path(f).parent

    name = getattr(f, 'name', None)
    if name in (None, '<stdout>', '<stderr>'):
        return None
    path = Path(name)
    return path.parent if path.exists() else None


class SpatialCalibrations(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    offset: float
    scale: float
    units: str

class DetectorConfiguration(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    pass

class CameraProcessingParameters(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    readout_area: t.Tuple[float, float, float, float]
    processing: t.List[str]


class Properties(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    detector_configuration:DetectorConfiguration
    camera_processing_parameters: CameraProcessingParameters


class InstrumentMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    high_tension: float
    defocus:float

class ScanMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    fov_nm: float
    scan_context_size: t.Tuple[float, float]
    scan_size: t.Tuple[float, float] 
    rotation_deg: t.Optional[float] = 0

class Metadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    scan: ScanMetadata
    instrument: InstrumentMetadata
    


class NionMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    file_type: t.Literal['nion_metadata'] = 'nion_metadata'

    @classmethod
    def from_json(cls, json_data, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        
        self = from_data(json_data, cls, custom=custom)
    
        return self

    def __post_init__(self):
        object.__setattr__(self, 'path', None)

    title: str
    """Experiment name"""

    # version: t.Annotated[int, IsVersion(exactly=1)] = 1

    version:t.Optional[float]
    
    """Metadata version"""
    spatial_calibrations: t.List[SpatialCalibrations]

    intensity_calibration: t.Dict

    metadata: Metadata
    properties: Properties


    def is_simulated(self) -> bool:
        return self.file_type == "pyMultislicer_metadata"


def load_4d(path: t.Union[str, Path], scan_shape: t.Optional[t.Union[t.Tuple[int, int], t.Tuple[float, float]]] = None, *,
            memmap: bool = False, flips: t.Optional[t.Tuple[bool, bool, bool]] = None) -> NDArray[numpy.float32]:
    """
    Load a raw nion dataset into memory.

    The file is loaded so the dimensions are: (scan_y, scan_x, k_y, k_x), with y decreasing downwards.

    Patterns are not fftshifted or normalized upon loading.

    # Parameters

     - `path`: Path to file to load
     - `scan_shape`: Scan shape of dataset. Will be inferred from the filename if not specified.
     - `memmap`: If specified, memmap the file as opposed to loading it eagerly.
     - `flips`: Flips to apply to the diffraction patterns, `(flip_y, flip_x, transpose)`.
       Defaults to `(True, False, False)` (appears to be the most common orientation).

    Returns a numpy array (or `numpy.memmap`)
    """
    path = Path(path)

    scan_shape = tuple(map(int, scan_shape))

    n_y, n_x = scan_shape

    with zf.ZipFile(path, 'r') as data_file:

        with io.BufferedReader(data_file.open('data.npy', mode='r')) as f:
            a = numpy.load(f)
            print(f"Loaded 'data.npy'")
    
    if a.shape[0] !=  n_y:
        raise ValueError(f"Got {a.shape[0]} y probes, expected {n_y}.")
    
    if a.shape[1] !=  n_x:
        raise ValueError(f"Got {a.shape[1]} x probes, expected {n_x}.")

    return apply_flips(a, flips or (False, False, False)) 


@t.overload
def save_4d(arr: NDArray[numpy.float32], *, path: t.Union[str, Path], folder: None = None, name: None = None,
            flips: t.Optional[t.Tuple[bool, bool, bool]] = None):
    ...

@t.overload
def save_4d(arr: NDArray[numpy.float32], *, path: None = None, folder: t.Union[str, Path], name: t.Optional[str] = None,
            flips: t.Optional[t.Tuple[bool, bool, bool]] = None):
    ...

def save_4d(arr: NDArray[numpy.float32], *, path: t.Union[str, Path, None] = None,
            folder: t.Union[str, Path, None] = None, name: t.Optional[str] = None,
            flips: t.Optional[t.Tuple[bool, bool, bool]] = None):
    """
    Save a raw EMPAD dataset.

    Either `path` or `folder` can be specified. If `folder` is specified,
    `name` will be used as a format string to determine the filename.
    `path` and `folder` cannot be specified simultaneously.

    Patterns are not fftshifted or normalized upon saving.

    Parameters:
     - `arr`: Array to save
     - `path`: Path to save dataset to.
     - `folder`: Folder to save dataset inside.
     - `name`: When `folder` is specified, format to use to determine filename. Defaults to `"scan_x{x}_y{y}.raw"`.
       Will be formatted using the scan shape `{'x': n_x, 'y': n_y}`.
     - `flips`: Flips to apply to the diffraction patterns, `(flip_y, flip_x, transpose)`.
       Defaults to `(True, False, False)` (appears to be the most common orientation).
    """

    try:
        assert len(arr.shape) == 4
        assert arr.shape[2:] == (128, 128)
    except AssertionError as e:
        raise ValueError("Invalid data format") from e

    if folder is not None:
        if path is not None:
            raise ValueError("Cannot specify both 'path' and 'folder'")

        n_y, n_x = arr.shape[:2]
        path = Path(folder) / (name or "scan_x{x}_y{y}.raw").format(x=n_x, y=n_y)
    elif path is not None:
        path = Path(path)
    else:
        raise ValueError("Must specify either 'path' or 'folder'")

    out_shape = list(arr.shape)
    out_shape[2] = 130  # dead rows

    out = numpy.zeros(out_shape, dtype=numpy.float32)
    out[..., :128, :] = apply_flips(to_numpy(arr.astype(numpy.float32)), flips or (True, False, False))

    with open(path, 'wb') as f:
        out.tofile(f)
