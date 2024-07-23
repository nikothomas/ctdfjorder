from ctdfjorder.constants import *
from ctdfjorder.CTDExceptions.CTDExceptions import CTDError
from ctdfjorder.ctdplot import plot_original_data, plot_predicted_data

from torch import where, min, mean, abs, tensor, float32, no_grad
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import polars as pl
import numpy as np

from os import path
from os import getcwd


class MinMaxScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.min_scaled = None

    def fit(self, df: pl.DataFrame, column_label: str) -> None:
        self.min_val = df.select(pl.col(column_label)).min().item()
        self.max_val = df.select(pl.col(column_label)).max().item()
        range_min, range_max = self.feature_range
        self.scale = (range_max - range_min) / (self.max_val - self.min_val)
        self.min_scaled = range_min - self.min_val * self.scale

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise ValueError(
                "This MinMaxScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        scaled_series = data * self.scale + self.min_scaled
        return scaled_series

    def inverse_transform(self, scaled_series: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise ValueError(
                "This MinMaxScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        inverse_scaled_series = (scaled_series - self.min_scaled) / self.scale + self.min_val
        return inverse_scaled_series


def clean_salinity_ai(profile: pl.DataFrame, profile_id: int) -> pl.DataFrame:
    """
    Cleans salinity using a GRU (Gated Recurrent Unit) machine learning model.

    Parameters
    ----------
    profile : pl.DataFrame
        Single profile of CTD (Conductivity, Temperature, Depth) data.
    profile_id : int
        Profile number, used for identification and logging.

    Returns
    -------
    pl.DataFrame
        CTD data with cleaned salinity values.

    Notes
    -----
    - This process bins the data every 0.5 dbar of pressure.
    - The GRU model uses a 16-width layer with a loss function designed to penalize decreasing salinity values with respect to pressure.
    - The penalization term is weighted to emphasize the importance of non-decreasing salinity with increasing pressure.

    GRU Model
    ---------
    The GRU model consists of:
    - An input layer that takes the binned CTD data.
    - A GRU layer with 16 units.
    - An output layer that predicts the salinity values.

    Loss Function
    -------------
    The custom loss function used in this model is the Mean Absolute Error (MAE) with an additional penalty term for salinity predictions that decrease with pressure. This is given by:

    .. math::
        L = \text{MAE}(y_{true}, y_{pred}) + \lambda \cdot \text{mean}(\text{penalties})

    where penalties are calculated as:

    .. math::
        \text{penalties} = \text{where}(\Delta s_{pred} < 0, \min(\Delta s_{pred}, 0), 0)

    Here, :math:`\Delta s_{pred}` represents the change in predicted salinity values, and :math:`\lambda` is the weighting factor for the penalty term.

    Methods
    -------
    - `GRUModel`: Defines the structure of the GRU model.
    - `loss_function`: Computes the custom loss including the penalty for decreasing salinity with pressure.
    - `build_gru`: Initializes the GRU model and the optimizer.
    - `run_gru`: Executes the GRU model on the provided CTD data, including preprocessing, model training, and postprocessing.

    Examples
    --------
    >>> cleaned_data = clean_salinity_ai(profile, profile_id=1)
    >>> cleaned_data.head()

    """

    class GRUModel(nn.Module):
        def __init__(self, input_shape):
            super(GRUModel, self).__init__()
            self.gru = nn.GRU(input_shape[1], 16, batch_first=True)
            self.output_layer = nn.Linear(16, input_shape[1])

        def forward(self, x):
            gru_out, _ = self.gru(x)
            output = self.output_layer(gru_out)
            return output

    def loss_function(y_true, y_pred):
        """
        MAE loss with additional term to penalize salinity predictions that increase with pressure.

        Parameters
        ----------
        y_true
            True salinity tensor.
        y_pred
            Predicted salinity tensor.
        Returns
        -------
        torch.Tensor
            Loss value as a tensor.
        """
        # Assuming salinity is at index 0
        salinity_true = y_true[:, :, 0]
        salinity_pred = y_pred[:, :, 0]

        # Calculate differences between consecutive values
        delta_sal_pred = salinity_pred[:, 1:] - salinity_pred[:, :-1]

        # Penalize predictions where salinity decreases while pressure increases
        penalties = where(
            delta_sal_pred < 0,
            min(delta_sal_pred, tensor(0.0, device=delta_sal_pred.device)),
            tensor(0.0, device=delta_sal_pred.device),
        )

        # Calculate mean absolute error
        mae = mean(abs(y_true - y_pred))

        # Add penalties
        total_loss = mae + 12.0 * mean(
            penalties
        )  # Adjust weighting of penalty as needed
        return total_loss

    def build_gru(input_shape):
        model = GRUModel(input_shape)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        return model, optimizer

    def run_gru(data: pl.DataFrame, show_plots=False):
        """
        Runs the GRU.

        Parameters
        ----------
        data : DataFrame
            CTD dataframe
        show_plots : bool, default False
            True saves plots in working directory within the plots folder.
        Returns
        -------
        DataFrame
            CTD data with clean salinity values.

        Raises
        ------
        CTDError
            When there are not enough values to train on.

        """
        filename = data.select(pl.col(FILENAME_LABEL))
        filtered_data = data.filter(pl.col(DEPTH_LABEL) > 1)
        filtered_data = filtered_data.with_columns(
            (pl.col(PRESSURE_LABEL) // 0.5 * 0.5).alias("pressure_bin")
        )
        # Define the desired columns and their aggregation functions
        column_agg_dict = {
            TEMPERATURE_LABEL: pl.mean(TEMPERATURE_LABEL),
            CHLOROPHYLL_LABEL: pl.mean(CHLOROPHYLL_LABEL),
            SEA_PRESSURE_LABEL: pl.mean(SEA_PRESSURE_LABEL),
            DEPTH_LABEL: pl.mean(DEPTH_LABEL),
            SALINITY_LABEL: pl.median(SALINITY_LABEL),
            SPEED_OF_SOUND_LABEL: pl.mean(SPEED_OF_SOUND_LABEL),
            SPECIFIC_CONDUCTIVITY_LABEL: pl.mean(SPECIFIC_CONDUCTIVITY_LABEL),
            CONDUCTIVITY_LABEL: pl.mean(CONDUCTIVITY_LABEL),
            DENSITY_LABEL: pl.mean(DENSITY_LABEL),
            POTENTIAL_DENSITY_LABEL: pl.mean(POTENTIAL_DENSITY_LABEL),
            SALINITY_ABS_LABEL: pl.mean(SALINITY_ABS_LABEL),
            TIMESTAMP_LABEL: pl.first(TIMESTAMP_LABEL),
            LONGITUDE_LABEL: pl.first(LONGITUDE_LABEL),
            LATITUDE_LABEL: pl.first(LATITUDE_LABEL),
            UNIQUE_ID_LABEL: pl.first(UNIQUE_ID_LABEL),
            FILENAME_LABEL: pl.first(FILENAME_LABEL),
            PROFILE_ID_LABEL: pl.first(PROFILE_ID_LABEL),
            SECCHI_DEPTH_LABEL: pl.first(SECCHI_DEPTH_LABEL),
        }
        available_columns = {
            col: agg_func
            for col, agg_func in column_agg_dict.items()
            if col in data.columns
        }
        data_binned = filtered_data.group_by("pressure_bin", maintain_order=True).agg(
            list(available_columns.values())
        )
        data_binned = data_binned.rename({"pressure_bin": PRESSURE_LABEL})
        if data_binned.limit(4).height < 2:
            raise CTDError(
                message=ERROR_GRU_INSUFFICIENT_DATA,
                filename=filename,
            )
        salinity = np.array(data_binned.select(pl.col(SALINITY_LABEL)).to_numpy())
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df=data_binned, column_label=SALINITY_LABEL)
        scaled_sequence = scaler.transform(salinity)
        scaled_seq = np.expand_dims(scaled_sequence, axis=0)
        min_pres = data_binned.select(pl.min(DEPTH_LABEL)).item()
        max_pres = data_binned.select(pl.max(DEPTH_LABEL)).item()
        pres_range = max_pres - min_pres
        epochs = int(pres_range * 12)
        input_shape = scaled_seq.shape[1:]
        model, optimizer = build_gru(input_shape)
        criterion = loss_function
        tensor_data = tensor(scaled_seq, dtype=float32)
        dataset = TensorDataset(tensor_data, tensor_data)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()  # Zero the gradients
                y_pred = model(x_batch)  # Forward pass
                loss = loss_function(y_batch, y_pred)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights
        model.eval()
        with no_grad():
            X_pred = model(tensor_data).numpy()
        predicted_seq = np.array(scaler.inverse_transform(X_pred[0])).flatten()
        if show_plots:
            xlim, ylim = plot_original_data(
                data.select(SALINITY_LABEL).to_numpy(),
                data.select(DEPTH_LABEL).to_numpy(),
                filename + str(profile_id),
                plot_path=path.join(getcwd(), "ctdplots", f"{filename}_original.png"),
            )
            plot_predicted_data(
                salinity=predicted_seq,
                depths=data_binned.select(DEPTH_LABEL).to_numpy(),
                filename=filename + str(profile_id),
                xlim=xlim,
                ylim=ylim,
                plot_path=path.join(getcwd(), "ctdplots", f"{filename}_predicted.png"),
            )
        data_binned = data_binned.with_columns(
            pl.Series(predicted_seq, dtype=pl.Float64).alias(SALINITY_LABEL)
        )
        return data_binned

    return run_gru(profile)