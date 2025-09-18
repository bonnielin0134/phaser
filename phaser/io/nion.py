
from pathlib import Path
import typing as t

import numpy
import pane
import pane.io
from numpy.typing import NDArray
from pane.convert import IntoConverterHandlers, from_data
from typing_extensions import Self

from phaser.utils.image import apply_flips

import zipfile as zf
import io


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
        return from_data(json_data, cls, custom=custom)

    def __post_init__(self):
        object.__setattr__(self, 'path', None)

    title: str
    """Experiment name"""

    # version: t.Annotated[int, IsVersion(exactly=1)] = 1

    version:t.Optional[float]
    
    """Metadata version"""
    spatial_calibrations: t.List[SpatialCalibrations]

    intensity_calibration: t.Dict

    # intensity_calibration['offset']: t.Float
    # intensity_calibration['scale']: t.Float

    metadata: Metadata
    properties: Properties

    # scan: ScanMetadata
    

    # path: t.Optional[Path] = pane.field(init=False, exclude=True)
    # empad_version: t.Optional[int] = None
    # """Empad version used. Defaults to v1 if not specified."""

    # raw_filename: str
    # """Raw 4DSTEM data filename, relative to metadata location."""

    # det_flips: t.Optional[t.Tuple[bool, bool, bool]] = None
    # """
    # Flips to apply to the raw diffraction patterns, (flip_y, flip_x, transpose).
    # Defaults to `(True, False, False)` (appears to be the most common orientation).
    # """
    # det_rotation: float = 0.0
    # """Detector rotation (degrees)."""

    # orig_path: t.Optional[Path] = None
    # """Original path to experimental folder."""

    # 
    # """Current path to experimental folder (based on metadata loading)"""

    # author: t.Optional[str] = None
    # """Author of dataset"""
    # time: t.Optional[str] = None
    # """Image acquisition time (RFC 2822 format)"""
    # time_unix: t.Optional[float] = None
    # """Image acquisition time (seconds since Unix epoch)"""
    # bg_unix: t.Optional[float] = None
    # """Background image acquisition time (seconds since Unix epoch)"""
    # has_bg: t.Optional[bool] = None
    # """Whether background image is valid"""

    # voltage: float
    # """Accelerating voltage (V)."""
    # conv_angle: t.Optional[float] = None
    # """Convergence angle (mrad)."""
    # defocus: t.Optional[float] = None
    # """Defocus (m). Positive is overfocus."""
    # camera_length: t.Optional[float] = None
    # """Camera length (m)."""
    # diff_step: t.Optional[float] = None
    # """Diffraction pixel size (mrad/px)."""

    # scan_rotation: float = 0.0
    # """Scan rotation (degrees)."""
    # scan_shape: t.Tuple[int, int]
    # """Scan shape (x, y)."""
    # scan_fov: t.Tuple[float, float]
    # """Scan field of view (m)."""
    # scan_step: t.Tuple[float, float]
    # """Scan step (m/px)."""

    # exposure_time: t.Optional[float] = None
    # """Pixel exposure time (s)."""
    # post_exposure_time: t.Optional[float] = None
    # """Pixel post-exposure time (s)."""
    # beam_current: t.Optional[float] = None
    # """Approx. beam current (A)."""
    # adu: t.Optional[float] = None
    # """Single-electron intensity (data units)."""

    # scan_correction: t.Optional[t.Annotated[NDArray[numpy.floating], shape((2, 2))]] = None
    # """Scan correction matrix, [x', y'] = scan_correction @ [x, y]"""

    # scan_positions: t.Optional[t.List[t.Tuple[float, float]]] = None
    # """
    # Scan position override (m).
    # Should be specified as a 1d list of (x, y) positions, in scan order. `scan_correction` is applied to these positions (if present).
    # """

    # notes: t.Optional[str] = None

    # crop: t.Optional[t.Tuple[int, int, int, int]] = None
    # """Region scan is valid within, (min_y, max_y, min_x, max_x). Python-style slicing."""

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
