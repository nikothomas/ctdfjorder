from torch import optim

from ctdfjorder.constants.constants import *
from ctdfjorder.visualize.ctd_plot import plot_original_data, plot_predicted_data
import torch
from torch.utils.data import TensorDataset, DataLoader

import polars as pl
import numpy as np

from os import path
from os import getcwd

from sklearn.preprocessing import MinMaxScaler


class AI:
    """
    This class should not be imported, see :func:`ctdfjorder.CTD.CTD.clean` to use the methods here.

    """

    @staticmethod
    def clean_salinity_ai(profile: pl.DataFrame, profile_id: int) -> pl.DataFrame:
        r"""
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

        * An input layer that takes the binned CTD data.
        * A GRU layer with 16 units.
        * An output layer that predicts the salinity values.

        Loss Function
        -------------
        The custom loss function used in this model is the Mean Absolute Error (MAE) with an additional penalty term for salinity predictions that decrease with pressure. This is given by:

        .. math::
            L = \text{MAE}(y_{true}, y_{pred}) + \lambda \cdot \text{mean}(\text{penalties})

        where penalties are calculated as:

        .. math::
            \text{penalties} = \text{where}(\Delta s_{pred} < 0, \min(\Delta s_{pred}, 0), 0)

        Here, :math:`\Delta s_{pred}` represents the change in predicted salinity values, and :math:`\lambda` is the weighting factor for the penalty term.

        Examples
        --------
        >>> cleaned_data = AI.clean_salinity_ai(profile, profile_id=1)
        >>> cleaned_data.head()

        """

        class GRUModel(torch.nn.Module):
            def __init__(self, input_shape):
                super(GRUModel, self).__init__()
                self.gru = torch.nn.GRU(input_shape[1], 16, batch_first=True)
                self.output_layer = torch.nn.Linear(16, input_shape[1])

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
            penalties = torch.where(
                delta_sal_pred < 0,
                -torch.min(
                    delta_sal_pred, torch.tensor(0.0, device=delta_sal_pred.device)
                ),
                torch.tensor(0.0, device=delta_sal_pred.device),
            )

            # Calculate mean absolute error
            mae = torch.mean(torch.abs(y_true - y_pred))

            # Add penalties
            total_loss = mae + 12.0 * torch.mean(
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
            filename = data.select(pl.col(FILENAME.label).first()).item()
            filtered_data = data.filter(pl.col(DEPTH.label) > 1)
            filtered_data = filtered_data.with_columns(
                (pl.col(PRESSURE.label) // 0.5 * 0.5).alias("pressure_bin")
            )
            # Define the desired columns and their aggregation functions
            column_agg_dict = {
                TEMPERATURE.label: pl.mean(TEMPERATURE.label),
                YEAR.label: pl.first(YEAR.label),
                MONTH.label: pl.first(MONTH.label),
                CHLOROPHYLL.label: pl.mean(CHLOROPHYLL.label),
                SEA_PRESSURE.label: pl.mean(SEA_PRESSURE.label),
                DEPTH.label: pl.mean(DEPTH.label),
                SALINITY.label: pl.median(SALINITY.label),
                SPEED_OF_SOUND.label: pl.mean(SPEED_OF_SOUND.label),
                SPECIFIC_CONDUCTIVITY.label: pl.mean(SPECIFIC_CONDUCTIVITY.label),
                CONDUCTIVITY.label: pl.mean(CONDUCTIVITY.label),
                DENSITY.label: pl.mean(DENSITY.label),
                POTENTIAL_DENSITY.label: pl.mean(POTENTIAL_DENSITY.label),
                ABSOLUTE_SALINITY.label: pl.mean(ABSOLUTE_SALINITY.label),
                TIMESTAMP.label: pl.first(TIMESTAMP.label),
                LONGITUDE.label: pl.first(LONGITUDE.label),
                LATITUDE.label: pl.first(LATITUDE.label),
                UNIQUE_ID.label: pl.first(UNIQUE_ID.label),
                FILENAME.label: pl.first(FILENAME.label),
                PROFILE_ID.label: pl.first(PROFILE_ID.label),
                SECCHI_DEPTH.label: pl.first(SECCHI_DEPTH.label),
                SITE_NAME.label: pl.first(SITE_NAME.label),
                SITE_ID.label: pl.first(SITE_ID.label),
            }
            available_columns = {
                col: agg_func
                for col, agg_func in column_agg_dict.items()
                if col in data.columns
            }
            data_binned = filtered_data.group_by(
                "pressure_bin", maintain_order=True
            ).agg(list(available_columns.values()))
            data_binned = data_binned.rename({"pressure_bin": PRESSURE.label})
            if data_binned.limit(4).height < 2:
                raise ValueError(f"{filename} - Insufficient data for the GRU, profile must contain at least 5m of samples.")
            salinity = np.array(data_binned.select(pl.col(SALINITY.label)).to_numpy())
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(salinity)
            scaled_sequence = scaler.transform(salinity)
            scaled_seq = np.expand_dims(scaled_sequence, axis=0)
            min_pres = data_binned.select(pl.min(DEPTH.label)).item()
            max_pres = data_binned.select(pl.max(DEPTH.label)).item()
            pres_range = max_pres - min_pres
            epochs = int(pres_range * 12)
            input_shape = scaled_seq.shape[1:]
            model, optimizer = build_gru(input_shape)
            criterion = loss_function
            tensor_data = torch.tensor(scaled_seq, dtype=torch.float32)
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
            with torch.no_grad():
                X_pred = model(tensor_data).numpy()
            predicted_seq = np.array(scaler.inverse_transform(X_pred[0])).flatten()
            if show_plots:
                xlim, ylim = plot_original_data(
                    data.select(SALINITY.label).to_numpy(),
                    data.select(DEPTH.label).to_numpy(),
                    filename + str(profile_id),
                    plot_path=path.join(
                        getcwd(), DEFAULT_PLOTS_FOLDER, f"{filename}_original.png"
                    ),
                )
                plot_predicted_data(
                    salinity=predicted_seq,
                    depths=data_binned.select(DEPTH.label).to_numpy(),
                    filename=filename + str(profile_id),
                    xlim=xlim,
                    ylim=ylim,
                    plot_path=path.join(
                        getcwd(), DEFAULT_PLOTS_FOLDER, f"{filename}_predicted.png"
                    ),
                )
            data_binned = data_binned.with_columns(
                pl.Series(predicted_seq, dtype=pl.Float64).alias(SALINITY.label)
            )
            return data_binned

        return run_gru(profile)
