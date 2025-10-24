import corner
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import pandas as pd
from .consts import BPP_SHAPE, KICK_SHAPE, BPP_COLUMNS, KICK_COLUMNS

__all__ = ['BackPopsteriors']

class BackPopsteriors():
    def __init__(self, file=None, points=None, log_w=None, log_l=None, var_names=None,
                 blobs=None, var_labels=None):
        """Utility class to handle and analyse posterior samples from BackPop.

        Parameters
        ----------
        file : str, optional
            Path to an HDF5 file containing posterior samples. The file should contain datasets
            named 'points', 'log_w', 'log_l', 'var_names', and 'blobs'.
        points : ~numpy.ndarray, optional
            Array of shape (n_samples, n_vars) containing posterior samples.
        log_w : ~numpy.ndarray, optional
            Array of shape (n_samples,) containing log weights for each sample.
        log_l : ~numpy.ndarray, optional
            Array of shape (n_samples,) containing log likelihoods for each sample.
        var_names : list of str, optional
            List of variable names corresponding to the columns in `points`.
        blobs : ~numpy.ndarray, optional
            Array of shape (n_samples, ...) containing additional data associated with each sample.
            The exact shape and contents depend on the simulation outputs.
        var_labels : list of str, optional
            List of labels for the variables, used in plotting. If not provided, `var_names` will be used.

        Raises
        ------
        ValueError
            If neither `file` nor all of `points`, `log_w`, `log_l`, and `var_names` are provided.
        """
        self.file = file
        self.blobs = None
        self.bpp = None
        self.kick_info = None

        # load data from file if provided
        if file is not None:
            with h5.File(file, 'r') as f:
                self.points = f['points'][:]
                self.log_w = f['log_w'][:]
                self.log_l = f['log_l'][:]
                self.var_names = np.array(f['var_names'][:].astype(str).tolist())

                if 'blobs' in f:
                    self.blobs = f['blobs'][:]
                if 'bpp' in f:
                    self.bpp = pd.read_hdf(file, 'bpp')
                if 'kick_info' in f:
                    self.kick_info = pd.read_hdf(file, 'kick_info')
        # otherwise use provided data
        elif points is not None and log_w is not None and log_l is not None and var_names is not None:
            self.points = points
            self.log_w = log_w
            self.log_l = log_l
            self.var_names = var_names
            self.blobs = blobs 
        # or shout at the user
        else:
            raise ValueError("Must provide either a file or points, log_w, log_l, and labels directly.")

        self.labels = var_labels if var_labels is not None else self.var_names

        # if blobs are provided, parse them into dataframes
        if self.blobs is not None and self.bpp is None or self.kick_info is None:
            self.bpp = pd.DataFrame(self.blobs["bpp"].reshape(-1, BPP_SHAPE[-1]), columns=BPP_COLUMNS)
            self.kick_info = pd.DataFrame(self.blobs["kick_info"].reshape(-1, KICK_SHAPE[-1]),
                                          columns=KICK_COLUMNS)

            # set index so we can easily filter based on binaries
            self.bpp.index = np.repeat(np.arange(self.bpp.shape[0] / BPP_SHAPE[0]), BPP_SHAPE[0]).astype(int)
            self.kick_info.index = np.repeat(np.arange(self.kick_info.shape[0] / KICK_SHAPE[0]),
                                             KICK_SHAPE[0]).astype(int)

            # filter out empty data (evol_type would never be 0 in a real binary)
            self.bpp = self.bpp[self.bpp["evol_type"] > 0.0]

            # free up some memory
            self.blobs = None

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self):
        return (f"<BackPopPosteriors: {self.points.shape[0]} samples, "
                f"{self.points.shape[1]} variables")
    
    @property
    def n_vars(self):
        return self.points.shape[1]

    def cornerplot(self, which_vars=None, show=True, **kwargs):
        """Create a corner plot of the posterior samples.

        Parameters
        ----------
        which_vars : ``list`` of ``str``, optional
            List of variable names to include in the corner plot. If None, all variables will be used.
            Default is None.
        show : ``bool``, optional
            Whether to display the plot immediately. Default is True.
        **kwargs : additional keyword arguments
            Additional keyword arguments to pass to `corner.corner()`. See the `corner` documentation
            for available options.
        """
        mask = np.ones(self.n_vars, dtype=bool)
        if which_vars is not None:
            mask = np.isin(self.var_names, which_vars)
            if not np.any(mask):
                raise ValueError("No matching variable names found.")
        fig = corner.corner(
            self.points[:, mask], weights=np.exp(self.log_w), bins=kwargs.pop("bins", 20),
            labels=self.labels[mask], color=kwargs.pop("color", '#074662'),
            plot_datapoints=kwargs.pop("plot_datapoints", False),
            range=kwargs.pop("range", np.repeat(0.999, mask.sum())), **kwargs)

        if show:
            plt.show()
        return fig
    
    def save(self, file=None):
        """Save the posterior samples to an HDF5 file.

        Parameters
        ----------
        file : ``str``, optional
            Path to the output HDF5 file. If not provided, the file path used during initialization
            will be used.
        """
        if file is None and self.file is None:
            raise ValueError("Must provide a file path to save to.")
        elif file is None:
            file = self.file

        with h5.File(file, 'w') as f:
            f.create_dataset('points', data=self.points)
            f.create_dataset('log_w', data=self.log_w)
            f.create_dataset('log_l', data=self.log_l)
            f.create_dataset('var_names', data=[n for n in self.var_names])
            if self.blobs is not None:
                f.create_dataset('blobs', data=self.blobs)

        # save bpp and kick info if they exist
        if self.bpp is not None:
            self.bpp.to_hdf(file, 'bpp')
        if self.kick_info is not None:
            self.kick_info.to_hdf(file, 'kick_info')
