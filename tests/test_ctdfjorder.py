import os
from pathlib import Path
import pytest
import polars as pl
import pandas as pd
from ctdfjorder.exceptions.exceptions import CTDError
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


@pytest.fixture(params=['CC1531002_20181225_114931.csv',
                        "208039_20211215_1003.rsk",])
def ctd_instance(request, mock_master_sheet):
    test_file_path = os.path.join(Path(__file__).parent, request.param)
    return CTD(str(test_file_path))


# Tests

def test_ctd_initialization(ctd_instance):
    assert isinstance(ctd_instance, CTD)
    assert not ctd_instance.get_df().is_empty()


def test_ctd_initialization_multiprofile_rbr(mock_master_sheet):
    test_file_path = os.path.join(Path(__file__).parent, '208041_20230120_1643.rsk')
    ctd_instance_rbr = CTD(test_file_path)
    max_profile_id = ctd_instance_rbr.get_df().select(PROFILE_ID.label).max().item()
    assert isinstance(ctd_instance_rbr, CTD)
    assert(max_profile_id == 2)
    assert not ctd_instance_rbr.get_df().is_empty()


def test_ctd_initialization_invalid_file():
    with pytest.raises(CTDError):
        CTD('invalid_file.invalid')


@pytest.mark.parametrize("pandas", [False, True])
def test_get_df(ctd_instance, pandas):
    df = ctd_instance.get_df(pandas=pandas)
    if pandas:
        assert isinstance(df, pd.DataFrame)
    else:
        assert isinstance(df, pl.DataFrame)


def test_remove_upcasts(ctd_instance):
    ctd_instance.remove_upcasts()
    assert not ctd_instance._data.is_empty()


@pytest.mark.parametrize("upper_bound,lower_bound", [
    (0.35, -0.5),
    (0.0, -0.5)
])
def test_filter_columns_by_range_not_empty(ctd_instance, upper_bound, lower_bound):
    column = TEMPERATURE.label
    ctd_instance.filter_columns_by_range(column=column, upper_bound=upper_bound, lower_bound=lower_bound)
    assert not ctd_instance._data.is_empty()


def test_filter_columns_by_range_empty(ctd_instance):
    column = TEMPERATURE.label
    upper_bound = 10
    lower_bound = 9
    ctd_instance.filter_columns_by_range(column=column, upper_bound=upper_bound, lower_bound=lower_bound)
    assert ctd_instance._data.is_empty()


def test_remove_non_positive_samples(ctd_instance):
    ctd_instance.remove_non_positive_samples()
    assert not ctd_instance._data.is_empty()
    assert ctd_instance._data.filter(pl.col(DEPTH.label) < 0).is_empty()


def test_clean_invalid_method(ctd_instance):
    with pytest.raises(ValueError):
        ctd_instance.clean('invalid_method')


def test_add_absolute_salinity(ctd_instance):
    ctd_instance.add_absolute_salinity()
    assert ABSOLUTE_SALINITY.label in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(ABSOLUTE_SALINITY.label).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(ABSOLUTE_SALINITY.label).is_nan().any()).item()


def test_add_density(ctd_instance):
    ctd_instance.add_density()
    assert DENSITY.label in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(DENSITY.label).has_nulls()).item()


def test_add_potential_density(ctd_instance):
    ctd_instance.add_potential_density()
    assert POTENTIAL_DENSITY.label in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(POTENTIAL_DENSITY.label).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(POTENTIAL_DENSITY.label).is_nan().any()).item()


def test_add_surface_salinity(ctd_instance):
    ctd_instance.add_surface_salinity()
    assert SURFACE_SALINITY.label in ctd_instance._data.columns, "Surface salinity column should be added."
    assert not ctd_instance._data.select(pl.col(SURFACE_SALINITY.label).has_nulls()).item(), "Surface salinity column should not have nulls."
    assert not ctd_instance._data.select(pl.col(SURFACE_SALINITY.label).is_nan().any()).item(), "Surface salinity column should not have NaNs."


def test_add_surface_temperature(ctd_instance):
    ctd_instance.add_surface_temperature()
    assert SURFACE_TEMPERATURE.label in ctd_instance._data.columns, "Surface temperature column should be added."
    assert not ctd_instance._data.select(pl.col(SURFACE_TEMPERATURE.label).has_nulls()).item(), "Surface temperature column should not have nulls."
    assert not ctd_instance._data.select(pl.col(SURFACE_TEMPERATURE.label).is_nan().any()).item(), "Surface temperature column should not have NaNs."


def test_add_meltwater_fraction(ctd_instance):
    ctd_instance.add_surface_salinity()  # Ensure that surface salinity is calculated first
    ctd_instance.add_meltwater_fraction()
    assert MELTWATER_FRACTION_EQ_10.label in ctd_instance._data.columns, "Meltwater fraction EQ 10 column should be added."
    assert not ctd_instance._data.select(pl.col(MELTWATER_FRACTION_EQ_10.label).has_nulls()).item(), "Meltwater fraction EQ 10 column should not have nulls."
    assert not ctd_instance._data.select(pl.col(MELTWATER_FRACTION_EQ_10.label).is_nan().any()).item(), "Meltwater fraction EQ 10 column should not have NaNs."

    assert MELTWATER_FRACTION_EQ_11.label in ctd_instance._data.columns, "Meltwater fraction EQ 11 column should be added."
    assert not ctd_instance._data.select(pl.col(MELTWATER_FRACTION_EQ_11.label).has_nulls()).item(), "Meltwater fraction EQ 11 column should not have nulls."
    assert not ctd_instance._data.select(pl.col(MELTWATER_FRACTION_EQ_11.label).is_nan().any()).item(), "Meltwater fraction EQ 11 column should not have NaNs."


def test_add_mean_surface_density(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_density()
    ctd_instance.add_mean_surface_density()
    assert SURFACE_DENSITY.label in ctd_instance._data.columns


def test_add_mld(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_density()
    ctd_instance.add_mld(reference=1, method="abs_density_avg", delta=0.05)
    assert any(label.startswith("MLD") for label in ctd_instance._data.columns)


def test_add_n_squared(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_density()
    ctd_instance.add_n_squared()
    assert N2.label in ctd_instance._data.columns
    assert P_MID.label in ctd_instance._data.columns


def test_save_to_csv(ctd_instance, tmp_path):
    output_file = tmp_path / "output.csv"
    ctd_instance.save_to_csv(str(output_file), null_value="")
    assert output_file.exists()


def test_add_speed_of_sound(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_speed_of_sound()
    assert SPEED_OF_SOUND.label in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col(SPEED_OF_SOUND.label).has_nulls()).item()
    assert not ctd_instance._data.select(pl.col(SPEED_OF_SOUND.label).is_nan().any()).item()


def test_add_potential_temperature(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_potential_temperature()
    assert "potential_temperature" in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col("potential_temperature").has_nulls()).item()
    assert not ctd_instance._data.select(pl.col("potential_temperature").is_nan().any()).item()


def test_add_conservative_temperature(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_conservative_temperature()
    assert "conservative_temperature" in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col("conservative_temperature").has_nulls()).item()
    assert not ctd_instance._data.select(pl.col("conservative_temperature").is_nan().any()).item()


def test_add_dynamic_height(ctd_instance):
    ctd_instance.remove_non_positive_samples()
    ctd_instance.remove_upcasts()
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_conservative_temperature()
    ctd_instance.add_dynamic_height()
    assert "dynamic_height" in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col("dynamic_height").has_nulls()).item()
    assert not ctd_instance._data.select(pl.col("dynamic_height").is_nan().any()).item()


def test_add_thermal_expansion_coefficient(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_conservative_temperature()
    ctd_instance.add_thermal_expansion_coefficient()
    assert "thermal_expansion_coefficient" in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col("thermal_expansion_coefficient").has_nulls()).item()
    assert not ctd_instance._data.select(pl.col("thermal_expansion_coefficient").is_nan().any()).item()


def test_add_haline_contraction_coefficient(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_conservative_temperature()
    ctd_instance.add_haline_contraction_coefficient()
    assert "haline_contraction_coefficient" in ctd_instance._data.columns
    assert not ctd_instance._data.select(pl.col("haline_contraction_coefficient").has_nulls()).item()
    assert not ctd_instance._data.select(pl.col("haline_contraction_coefficient").is_nan().any()).item()

def test_add_profile_classification(ctd_instance):
    ctd_instance.add_absolute_salinity()
    ctd_instance.add_density()
    ctd_instance.add_mld(reference=10, method="abs_density_avg", delta=0.05)
    ctd_instance.add_profile_classification()
    assert CLASSIFICATION.label in ctd_instance._data.columns


