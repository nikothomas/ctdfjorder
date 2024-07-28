import os
from pathlib import Path
import pytest
import polars as pl
import pandas as pd
from ctdfjorder.exceptions.ctd_exceptions import CTDError
from ctdfjorder.constants.constants import *
from ctdfjorder.CTD import CTD  # Assuming CTD is in ctdfjorder.CTD


# Helper function to load example data from files
def load_example_data(file_path):
    return pl.read_csv(file_path)


@pytest.fixture
def mock_master_sheet(monkeypatch):
    class MockMasterSheet:
        def find_match(self, profile):
            return MockMatch()

    class MockMatch:
        unique_id = 'dummy_id'
        secchi_depth = 10.0
        latitude = -64.668455
        longitude = -62.641775

    monkeypatch.setattr("ctdfjorder.metadata.master_sheet.MasterSheet", MockMasterSheet)

@pytest.fixture
def ctd_instance(mock_master_sheet):
    test_file_path = os.path.join(Path(__file__).parent,'CC1531002_20181225_114931.csv')
    return CTD(str(test_file_path))


# Tests

def test_ctd_initialization_castaway(mock_master_sheet):
    ctd_instance = CTD('CC1531002_20181225_114931.csv')
    assert isinstance(ctd_instance, CTD)
    assert not ctd_instance.get_df().is_empty()


def test_ctd_initialization_invalid_file():
    with pytest.raises(CTDError):
        CTD('invalid_file.invalid')


def test_get_df_polars(ctd_instance):
    df = ctd_instance.get_df(pandas=False)
    assert isinstance(df, pl.DataFrame)


def test_get_df_pandas(ctd_instance):
    df = ctd_instance.get_df(pandas=True)
    assert isinstance(df, pd.DataFrame)


def test_remove_upcasts(ctd_instance):
    ctd_instance.remove_upcasts()
    assert not ctd_instance._data.is_empty()


def test_filter_columns_by_range_not_empty(ctd_instance):
    columns = [TEMPERATURE_LABEL]
    upper_bounds = [0.35]
    lower_bounds = [0.30]
    ctd_instance.filter_columns_by_range(columns=columns, upper_bounds=upper_bounds, lower_bounds=lower_bounds)
    assert not ctd_instance._data.is_empty()


def test_filter_columns_by_range_empty(ctd_instance):
    with pytest.raises(CTDError):
        columns = [TEMPERATURE_LABEL]
        upper_bounds = [10]
        lower_bounds = [9]
        ctd_instance.filter_columns_by_range(columns=columns, upper_bounds=upper_bounds, lower_bounds=lower_bounds)


def test_remove_non_positive_samples(ctd_instance):
    ctd_instance.remove_non_positive_samples()
    assert not ctd_instance._data.is_empty()
    assert ctd_instance._data.filter(pl.col(DEPTH_LABEL) < 0).is_empty()


def test_remove_invalid_salinity_values(ctd_instance):
    ctd_instance.remove_invalid_salinity_values()
    assert not ctd_instance._data.is_empty()
    assert ctd_instance._data.filter(pl.col(SALINITY_LABEL) < 10).is_empty()


def test_clean_invalid_method(ctd_instance):
    with pytest.raises(CTDError):
        ctd_instance.clean('invalid_method')


def test_add_absolute_salinity(ctd_instance):
    ctd_instance.add_absolute_salinity()
    assert SALINITY_ABS_LABEL in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(SALINITY_ABS_LABEL).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(SALINITY_ABS_LABEL).is_nan().any()).item()


def test_add_density(ctd_instance):
    ctd_instance.add_density()
    assert DENSITY_LABEL in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(DENSITY_LABEL).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(DENSITY_LABEL).is_nan().any()).item()


def test_add_potential_density(ctd_instance):
    ctd_instance.add_potential_density()
    assert POTENTIAL_DENSITY_LABEL in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(POTENTIAL_DENSITY_LABEL).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(POTENTIAL_DENSITY_LABEL).is_nan().any()).item()

def test_add_surface_salinity_temp_meltwater(ctd_instance):
    ctd_instance.add_surface_salinity_temp_meltwater()
    assert SURFACE_SALINITY_LABEL in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(SURFACE_SALINITY_LABEL).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(SURFACE_SALINITY_LABEL).is_nan().any()).item()
    assert SURFACE_TEMPERATURE_LABEL in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(SURFACE_TEMPERATURE_LABEL).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(SURFACE_TEMPERATURE_LABEL).is_nan().any()).item()
    assert MELTWATER_FRACTION_LABEL in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(MELTWATER_FRACTION_LABEL).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(MELTWATER_FRACTION_LABEL).is_nan().any()).item()
def test_add_mean_surface_density(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_density()
    ctd_instance.add_mean_surface_density()
    assert SURFACE_DENSITY_LABEL in ctd_instance._data.columns


def test_add_mld(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_potential_density()
    ctd_instance.add_mld(reference=10, method="potential_density_avg", delta=0.05)
    assert any(label.startswith("MLD") for label in ctd_instance._data.columns)


def test_add_bf_squared(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_bf_squared()
    assert BV_LABEL in ctd_instance._data.columns


def test_save_to_csv(ctd_instance, tmp_path):
    output_file = tmp_path / "output.csv"
    ctd_instance.save_to_csv(str(output_file), null_value="")
    assert output_file.exists()
