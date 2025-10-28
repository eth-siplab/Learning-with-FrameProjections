import torch
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import pywt

class WaveletTransform:
    def __init__(self, wavelet='cmor', scales=None, fs=1.0):
        """
        Initialize the WaveletTransform class with a specified wavelet and scales.

        Parameters:
        - wavelet (str): The name of the wavelet (default is 'cmor' for continuous Morlet wavelet).
        - scales (numpy array): Array of scales to use (if None, it is computed automatically).
        - fs (float): Sampling frequency (default is 1.0).
        """
        self.wavelet = wavelet
        self.fs = fs  # Sampling frequency
        
        # Define scales (if not provided, generate logarithmically spaced scales)
        if scales is None:
            self.scales = np.geomspace(1, 128, num=48)  # Log-spaced scales for better resolution, was 48
        else:
            self.scales = scales

        # Compute equivalent frequencies for visualization
        self.frequencies = pywt.scale2frequency(self.wavelet, self.scales) * self.fs

    def compute_cwt(self, signal):
        """
        Compute the Continuous Wavelet Transform (CWT) for an input signal.
        Supports signals with shape (B, T) or (B, T, C).
        """
        # Ensure the signal is a torch tensor and move to CPU for PyWavelets compatibility
        signal = signal.detach().cpu().numpy() if isinstance(signal, torch.Tensor) else np.array(signal)

        if signal.ndim == 2:
            # Single-channel case: shape (B, T)
            B, T = signal.shape
            cwt_result = np.zeros((B, len(self.scales), T))  # Shape: (B, scales, T)

            # start = time.perf_counter()
            for i in range(B):
                coefficients, _ = pywt.cwt(signal[i], self.scales, self.wavelet, sampling_period=1/self.fs)
                cwt_result[i] = np.abs(coefficients)
            # end = time.perf_counter()
            # print(f"CWT loop took {end - start:.6f} seconds")
            # import pdb;pdb.set_trace();
            cwt_result = cwt_result[:, :, :, None]  # Add dummy channel dimension

            freq_dim = cwt_result.shape[1]
            time_dim = cwt_result.shape[2]

        elif signal.ndim == 3:
            # Multi-channel case: shape (B, T, C)
            B, T, C = signal.shape
            cwt_result = np.zeros((B, C, len(self.scales), T))  # Shape: (B, C, scales, T)
            
            for i in range(B):
                for j in range(C):
                    coefficients, _ = pywt.cwt(signal[i, :, j], self.scales, self.wavelet, sampling_period=1/self.fs)
                    cwt_result[i, j] = np.abs(coefficients)

            freq_dim = cwt_result.shape[2]
            time_dim = cwt_result.shape[3]
        else:
            raise ValueError("Signal must have shape (B, T) or (B, T, C)")

        # plot one case
        # self.plot_signal_and_cwt(signal, cwt_result)
        # Convert back to PyTorch tensor
        return torch.tensor(cwt_result, dtype=torch.float32), freq_dim, time_dim

    def plot_signal_and_cwt(self, signal, cwt_result):
        """
        Plots the time-domain signal and the CWT scalogram using pcolormesh.
        - Uses **logarithmic frequency scale**.
        """
        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()
        if isinstance(cwt_result, torch.Tensor):
            cwt_result = cwt_result.detach().cpu().numpy()

        # Select first sample and first channel for plotting
        signal = signal[10] if signal.ndim == 2 else signal[0, :, 0]
        cwt_result = cwt_result[10] if cwt_result.ndim == 3 else cwt_result[0, :, :, 0]

        time = np.linspace(0, len(signal) / self.fs, len(signal))
        freq = self.frequencies  

        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot time-domain signal
        axs[0].plot(time, signal)
        axs[0].set_title("Time-Domain Signal")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")

        # Plot scalogram using pcolormesh (correct frequency axis)
        pcm = axs[1].pcolormesh(time, freq, cwt_result, shading='auto', cmap='inferno')
        axs[1].set_yscale("log")  # Logarithmic frequency scale
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Frequency (Hz)")
        axs[1].set_title("Continuous Wavelet Transform (Scaleogram)")
        fig.colorbar(pcm, ax=axs[1])

        plt.tight_layout()
        plt.savefig('signal_and_wavelet.png')
        plt.show()

class FourierTransform:
    def __init__(self, fs=1.0):
        self.fs = fs  # Sampling frequency

    def compute_FT(self, signal):
        if not isinstance(signal, torch.Tensor):
            signal = torch.as_tensor(signal, dtype=torch.float32) 
        fft_result = torch.fft.rfft(signal, dim=-1, norm='ortho')
        return fft_result
        
class TSNEPlotter:
    def __init__(self, perplexity=30, n_iter=1000, uniform_color=False, random_state=None):
        """
        Args:
            perplexity: TSNE perplexity.
            n_iter: Number of TSNE iterations.
            uniform_color: If True, use a single colormap (Blues) with intensity variation.
            random_state: Seed for reproducibility.
        """
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.uniform_color = uniform_color
        self.random_state = random_state

    def plot(self, embeddings, labels, save_path=None, show=True, title="TSNE Plot"):
        """
        Args:
            embeddings: numpy array of shape (N, D)
            labels: numpy array of shape (N,) (numeric labels)
            save_path: If provided, save the plot to this path.
            show: If True, call plt.show().
            title: Title of the plot.
        """
        tsne = TSNE(n_components=2, perplexity=self.perplexity, n_iter=self.n_iter, random_state=self.random_state)
        X_tsne = tsne.fit_transform(embeddings)

        plt.figure(figsize=(8, 8))
        if self.uniform_color:
            # Normalize label values between 0 and 1
            norm_labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=norm_labels, cmap="Blues", alpha=0.7)
            plt.colorbar(scatter, label="Intensity")
        else:
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="viridis", alpha=0.7)
            plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("TSNE 1")
        plt.ylabel("TSNE 2")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def plot_misalignment(self, emb1, emb2, save_path=None, show=True, title="Misalignment Plot"):
        """
        The function computes the cosine similarity for each corresponding pair of embeddings,
        converts that to an angle (in degrees), and then plots a scatter of sample index vs. angle.
        If two embeddings are perfectly aligned (cosine = 1), the angle is 0 degrees.
        """
        # Convert to numpy arrays if necessary.
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.detach().cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.detach().cpu().numpy()

        # Normalize each embedding along the feature dimension.
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity per sample.
        cosine_sim = np.sum(emb1_norm * emb2_norm, axis=1) 
        # Compute angle
        angles = np.arccos(cosine_sim)

        # import pdb; pdb.set_trace()

        r = np.ones_like(angles)
        # Create polar plot.
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,8))
        scatter = ax.scatter(angles, r, alpha=0.75)
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, label="Normalized Misalignment")
        # Optionally, draw a unit circle.
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(theta, np.ones_like(theta), color='gray', linestyle='--')
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return angles

    def plot_all_misalignment(self, emb1, emb2, save_path=None, show=False, title="Misalignment Distribution"):
        """
        Compute and visualize the misalignment angle between corresponding embeddings in emb1 and emb2.
        
        Args:
            emb1: Tensor or numpy array of shape (N, D) (e.g. time_embedded).
            emb2: Tensor or numpy array of shape (N, D) (e.g. cwt_embedded).
            save_path: if provided, save the plot to this path.
            show: if True, display the plot.
            title: Title of the plot.
            
        Returns:
            angles_deg: A numpy array of misalignment angles in degrees (shape (N,)).
        """
        # Convert tensors to numpy arrays if needed.
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.detach().cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.detach().cpu().numpy()
        
        # Normalize each embedding along the feature dimension.
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity matrices.
        sim1 = np.dot(emb1_norm, emb1_norm.T)  # shape: (N, N)
        sim2 = np.dot(emb2_norm, emb2_norm.T)  # shape: (N, N)   
        # import pdb; pdb.set_trace()
        # Clip to ensure valid range for arccos.
        sim1 = np.clip(sim1, -1.0, 1.0)
        sim2 = np.clip(sim2, -1.0, 1.0)     

        # Convert cosine similarity to angles (in radians).
        angles1 = np.arccos(sim1)  # shape: (N, N)
        angles2 = np.arccos(sim2)  # shape: (N, N)
        
        # Compute the difference between the angle matrices.
        diff_matrix = angles1 - angles2

        # Get upper-triangle indices (exclude the diagonal).
        triu_idx = np.triu_indices(diff_matrix.shape[0], k=1)
        diff_values = diff_matrix[triu_idx]  # vector of differences

        # Plot on a unit circle (polar plot). All points will have radius = 1.
        r = np.ones_like(diff_values)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        scatter = ax.scatter(diff_values, r, c=np.abs(diff_values), cmap="Blues", alpha=0.7)
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label="Angle Difference (radians)")
        
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, np.ones_like(theta), color='gray', linestyle='--')
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        
        return diff_matrix


    def plot_misalignment_density(self, emb1, emb2, bins=10, palette='rocket', save_path=None, show=True, title="Misalignment Plot"):
        """
        The function computes the cosine similarity for each corresponding pair of embeddings,
        converts that to an angle (in degrees), and then plots a scatter of sample index vs. angle.
        If two embeddings are perfectly aligned (cosine = 1), the angle is 0 degrees.
        """
        # Convert to numpy arrays if necessary.
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.detach().cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.detach().cpu().numpy()

        # Normalize each embedding along the feature dimension.
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity per sample.
        cosine_sim = np.sum(emb1_norm * emb2_norm, axis=1) 
        # Compute angle
        angles = np.arccos(cosine_sim)

        # → plot rose diagram
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(3.22, 2.18))
        ax.grid(True, color='gray', linewidth=0.3)
        counts, bin_edges, patches = ax.hist(angles, bins=bins, density=True, edgecolor='k', alpha=0.7)
        # compute bin centers and their dist from π/2
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dists = np.abs(bin_centers - (np.pi / 2))
        dists_norm = dists / dists.max()  # scale to [0,1]

        # build a Seaborn palette of length 'bins'
        # palette can be name (e.g. "viridis", "rocket", "RdBu") or list of colors
        pal = sns.color_palette(palette, bins)

        # map each normalized distance to an index in the palette
        for patch, dn in zip(patches, dists_norm):
            idx = int(dn * (bins - 1))  # 0 → bin 0, 1 → last bin
            patch.set_facecolor(pal[idx])

        # orientation & zoom
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        min_a, max_a = angles.min(), angles.max()
        pad = (max_a - min_a) * 0.05
        ax.set_thetamin(np.degrees(min_a - pad))
        ax.set_thetamax(np.degrees(max_a + pad))

        ax.set_title(title, va='bottom')

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            save_path = save_path.replace('.png', '.svg')
            fig.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)            
        if show:
            plt.show()
        else:
            plt.close(fig)

        return angles


    def plot_all_misalignment_density(self, emb1, emb2, bins=10, palette='rocket', save_path=None, show=True, title="Misalignment Plot"):
        """
        Compute and visualize the misalignment angle between corresponding embeddings in emb1 and emb2.
        
        Args:
            emb1: Tensor or numpy array of shape (N, D) (e.g. time_embedded).
            emb2: Tensor or numpy array of shape (N, D) (e.g. cwt_embedded).
            save_path: if provided, save the plot to this path.
            show: if True, display the plot.
            title: Title of the plot.
            
        Returns:
            angles_deg: A numpy array of misalignment angles in degrees (shape (N,)).
        """
        # Convert tensors to numpy arrays if needed.
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.detach().cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.detach().cpu().numpy()
        
        # Normalize each embedding along the feature dimension.
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity matrices.
        sim1 = np.dot(emb1_norm, emb1_norm.T)  # shape: (N, N)
        sim2 = np.dot(emb2_norm, emb2_norm.T)  # shape: (N, N)   
        # import pdb; pdb.set_trace()
        # Clip to ensure valid range for arccos.
        sim1 = np.clip(sim1, -1.0, 1.0)
        sim2 = np.clip(sim2, -1.0, 1.0)     

        # Convert cosine similarity to angles (in radians).
        angles1 = np.arccos(sim1)  # shape: (N, N)
        angles2 = np.arccos(sim2)  # shape: (N, N)
        
        # Compute the difference between the angle matrices.
        diff_matrix = angles1 - angles2

        # Get upper-triangle indices (exclude the diagonal).
        triu_idx = np.triu_indices(diff_matrix.shape[0], k=1)
        diff_values = diff_matrix[triu_idx]  # vector of differences  

        angles = diff_values    

        # → plot rose diagram
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(3.22, 2.18))
        ax.grid(True, color='gray', linewidth=0.3)
        counts, bin_edges, patches = ax.hist(angles, bins=bins, density=True, edgecolor='k', alpha=0.7)
        # compute bin centers and their dist from 0 --> ideal case
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dists = np.abs(bin_centers)
        dists_norm = dists / dists.max()  # scale to [0,1]

        # build a Seaborn palette of length 'bins'
        # palette can be name (e.g. "viridis", "rocket", "RdBu") or list of colors
        pal = sns.color_palette(palette, bins)

        # map each normalized distance to an index in the palette
        for patch, dn in zip(patches, dists_norm):
            idx = int(dn * (bins - 1))  # 0 → bin 0, 1 → last bin
            patch.set_facecolor(pal[idx])

        # orientation & zoom
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        min_a, max_a = angles.min(), angles.max()
        pad = (max_a - min_a) * 0.05
        ax.set_thetamin(np.degrees(min_a - pad))
        ax.set_thetamax(np.degrees(max_a + pad))

        ax.set_title(title, va='bottom')

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            save_path = save_path.replace('.png', '.svg')
            fig.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)                  
        if show:
            plt.show()
        else:
            plt.close(fig)

        return angles      

    def plot_distance_scatter(self, emb1, emb2, save_path=None, cmap='crest', show=True, title="Pairwise L₂ Distances"):
        """
        Compute all pairwise L₂ distances in emb1 and emb2,
        then scatter-plot d1 vs. d2. Points on the y=x line
        mean the two distances agree.
        """
        # → to numpy
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.detach().cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.detach().cpu().numpy()

        # (Optional) normalize embeddings
        emb1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

        # Compute pairwise L2 distances
        d1 = np.linalg.norm(emb1[:, None, :] - emb1[None, :, :], axis=-1)
        d2 = np.linalg.norm(emb2[:, None, :] - emb2[None, :, :], axis=-1)

        # Extract upper‐triangle values (i<j)
        iu = np.triu_indices(d1.shape[0], k=1)
        d1_vals = d1[iu]
        d2_vals = d2[iu]

        # Build DataFrame
        df = pd.DataFrame({'d1': d1_vals, 'd2': d2_vals})

        # Get seaborn default blue
        default_blue = sns.color_palette("deep")[0]

        # Create jointplot
        g = sns.jointplot(
            data=df,
            x='d1',
            y='d2',
            kind='hex',
            marginal_kws=dict(
                bins=20,
                fill=True,
                color=default_blue,
                edgecolor='white',       # white edge for histogram bars
                linewidth=0.5,
                alpha=0.8
            ),
            joint_kws=dict(
                gridsize=40,
                cmap=cmap,
                mincnt=1,
                edgecolor=None,
                linewidth=0,
                alpha=0.9
            )
        )

        # Add colorbar
        hexpoly = g.ax_joint.collections[0]
        cbar = g.fig.colorbar(hexpoly, ax=g.ax_joint, pad=0.01)
        cbar.set_label('Count')

        # Identity line
        g.ax_joint.plot([0, 2], [0, 2], '--', color='black', linewidth=0.9)

        # Axes limits
        g.ax_joint.set_xlim(0, 2)
        g.ax_joint.set_ylim(0, 2)

        # Labels and styling
        g.set_axis_labels('L₂ distance (emb1)', 'L₂ distance (emb2)')
        g.ax_joint.grid(False)

        plt.tight_layout()

        if save_path:
            g.savefig(save_path, bbox_inches='tight')
            svg = save_path.rsplit('.', 1)[0] + '.svg'
            g.savefig(svg, format='svg', bbox_inches='tight', transparent=True)

        if show:
            plt.show()
        else:
            plt.close(g)

        import pdb;pdb.set_trace();

        d1_vals = (d1_vals - np.mean(d1_vals)) / np.std(d1_vals)
        d2_vals = (d2_vals - np.mean(d2_vals)) / np.std(d2_vals)    
        return np.dot(d1_vals, d2_vals) / len(d1_vals) , d1_vals, d2_vals