"""The CWRU bearing dataset: drive-end accelerometer, 12 kHz, 0.007-inch faults."""

from dataclasses import dataclass
from pathlib import Path

from scipy.io import loadmat

from src.contracts import Signal
from src.paths import RAW

SAMPLE_RATE = 12_000
URL = "https://engineering.case.edu/sites/default/files/{id}.mat"

# Each condition's four loads (0–3 hp) have consecutive file IDs.
FIRST_ID = {"normal": 97, "inner-race": 105, "ball": 118, "outer-race": 130}
RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# Two accelerometers: at the drive end (next to the faulty bearing) and at the fan end.
SENSORS = ("DE", "FE")

# Fault frequencies of the drive-end bearing (SKF 6205) as multiples of shaft speed.
FAULT_ORDERS = {"inner-race": 5.4152, "outer-race": 3.5848, "ball": 4.7135}


@dataclass(frozen=True)
class Recording:
    id: int
    condition: str
    load: int

    @property
    def rpm(self) -> int:
        return RPM[self.load]

    @property
    def url(self) -> str:
        return URL.format(id=self.id)

    @property
    def filename(self) -> str:
        return f"{self.condition}-{self.load}hp.mat"


RECORDINGS = tuple(
    Recording(id=first + load, condition=condition, load=load)
    for condition, first in FIRST_ID.items()
    for load in RPM
)


def load(recording: Recording, sensor: str = "DE", raw: Path = RAW) -> Signal:
    """The vibration of one recording at one sensor, selected by the recording's own ID.

    Some files carry other recordings too (normal-2hp.mat also holds a copy of
    the 1 hp run), so the channel is chosen by ID, never by position.
    """
    contents = loadmat(raw / recording.filename)
    key = f"X{recording.id:03d}"
    measured_rpm = contents.get(f"{key}RPM")
    return Signal(
        samples=contents[f"{key}_{sensor}_time"].ravel(),
        sample_rate=SAMPLE_RATE,
        rpm=int(measured_rpm.item()) if measured_rpm is not None else recording.rpm,
        condition=recording.condition,
        load=recording.load,
        source=recording.filename,
    )


def load_all(sensor: str = "DE", raw: Path = RAW) -> list[Signal]:
    return [load(recording, sensor, raw) for recording in RECORDINGS]
