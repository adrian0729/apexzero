"""Which OpenF1 circuit is which TUMFTM centreline, and whether the layouts still match.

OpenF1 names circuits by ``circuit_short_name`` ("Hungaroring", "Interlagos"); the
TUMFTM racetrack database names its files after the town ("Budapest", "SaoPaulo"). The
mapping is written out by hand because a fuzzy match would silently pair the wrong
layouts.

Several TUMFTM centrelines predate a layout change, so the current F1 track is not the
TUMFTM track even though the name matches. Registering such a session is still useful
(it is a negative control for the registration gate), but nothing downstream may treat
it as the same circuit. The flags below say why, and ``circuit_for`` refuses them unless
asked explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from apexzero.tracks.corners import EXCLUDED, R_DEV, R_TEST


@dataclass(frozen=True)
class Circuit:
    openf1: str             # OpenF1 circuit_short_name
    tumftm: str | None      # TUMFTM file stem, or None when the database has no such track
    layout_matches: bool    # TUMFTM geometry is the layout F1 races on in 2023-25
    note: str = ""

    @property
    def lockbox(self) -> bool:
        """An R-test circuit: registration of telemetry is fine, optimum lap times are not."""
        return self.tumftm in R_TEST

    @property
    def usable(self) -> bool:
        return self.tumftm is not None and self.layout_matches


# Layout notes are from the circuits' published layout changes; the registration gate
# (RMS residual and length) is the data-driven check that backs them up.
_OLD = "TUMFTM centreline predates the current F1 layout"
CIRCUITS: dict[str, Circuit] = {c.openf1: c for c in [
    Circuit("Austin", "Austin", True),
    Circuit("Hungaroring", "Budapest", True),
    Circuit("Catalunya", "Catalunya", False, f"{_OLD} (final chicane removed for 2023)"),
    Circuit("Melbourne", "Melbourne", False, f"{_OLD} (2022 rework, T9-10 chicane removed)"),
    Circuit("Mexico City", "MexicoCity", True),
    Circuit("Montreal", "Montreal", True),
    Circuit("Monza", "Monza", True),
    Circuit("Sakhir", "Sakhir", True),
    Circuit("Interlagos", "SaoPaulo", True),
    Circuit("Shanghai", "Shanghai", True, "no 2023 Chinese GP"),
    Circuit("Silverstone", "Silverstone", True),
    Circuit("Spa-Francorchamps", "Spa", True),
    Circuit("Spielberg", "Spielberg", True),
    Circuit("Suzuka", "Suzuka", True),
    Circuit("Yas Marina Circuit", "YasMarina", False, f"{_OLD} (2021 rework of several sections)"),
    Circuit("Zandvoort", "Zandvoort", False,
            f"{_OLD} (2020 rework with banked T3 and final corner)"),
    # Current F1 circuits without a TUMFTM centreline.
    *[Circuit(name, None, False, "not in the TUMFTM database") for name in (
        "Baku", "Imola", "Jeddah", "Kuala Lumpur", "Las Vegas", "Lusail", "Madring", "Miami",
        "Monte Carlo", "Singapore")],
]}

# TUMFTM tracks with no current F1 counterpart; IMS is the banked oval excluded everywhere.
NOT_ON_CALENDAR = sorted(set(R_DEV + R_TEST + EXCLUDED)
                         - {c.tumftm for c in CIRCUITS.values() if c.tumftm})


def circuit_for(openf1_name: str, allow_old_layout: bool = False) -> Circuit:
    """Look up an OpenF1 circuit; raise if it has no usable TUMFTM centreline."""
    try:
        c = CIRCUITS[openf1_name]
    except KeyError:
        raise KeyError(f"unknown OpenF1 circuit {openf1_name!r}; add it to CIRCUITS") from None
    if c.tumftm is None:
        raise ValueError(f"{openf1_name}: {c.note}")
    if not c.layout_matches and not allow_old_layout:
        raise ValueError(f"{openf1_name}: {c.note}")
    return c
