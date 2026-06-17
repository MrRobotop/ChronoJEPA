"""Test the ETT CSV loader."""

import numpy as np

from chronojepa.data import load_ett


def test_load_ett_drops_date_and_returns_channels(tmp_path) -> None:
    csv = tmp_path / "ett.csv"
    csv.write_text(
        "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n"
        "2016-07-01 00:00:00,1.0,2.0,3.0,4.0,5.0,6.0,7.0\n"
        "2016-07-01 01:00:00,1.5,2.5,3.5,4.5,5.5,6.5,7.5\n"
        "2016-07-01 02:00:00,2.0,3.0,4.0,5.0,6.0,7.0,8.0\n"
    )
    series = load_ett(csv)
    assert series.shape == (3, 7)
    assert series.dtype == np.float32
    assert np.allclose(series[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
