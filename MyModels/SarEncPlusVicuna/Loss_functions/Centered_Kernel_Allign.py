import torch
import torch.nn as nn


class RBFCKALoss(nn.Module):
    """
    Differentiable RBF-kernel CKA loss.

    Given:
        X: [B, Dx]
        Y: [B, Dy]

    Computes:
        CKA(X, Y)

    and returns:
        1 - CKA(X, Y)

    X and Y do NOT need to have the same feature dimension.
    """

    def __init__(self, sigma=None, eps=1e-8):
        super().__init__()

        self.sigma = sigma
        self.eps = eps

    def _rbf_kernel(self, X):
        """
        X: [B, D]

        Returns:
            K: [B, B]
        """

        # Pairwise squared Euclidean distances
        sq_dist = torch.cdist(X, X, p=2).pow(2)

        if self.sigma is None:
            # Median heuristic.
            #
            # Detach sigma so the bandwidth itself does not
            # introduce another gradient pathway.
            nonzero_dist = sq_dist[sq_dist > 0]

            if nonzero_dist.numel() > 0:
                sigma = torch.sqrt(
                    0.5 * torch.median(nonzero_dist)
                ).detach()
            else:
                sigma = torch.tensor(
                    1.0,
                    device=X.device,
                    dtype=X.dtype
                )

        else:
            sigma = torch.as_tensor(
                self.sigma,
                device=X.device,
                dtype=X.dtype
            )

        K = torch.exp(
            -sq_dist / (
                2.0 * sigma.pow(2) + self.eps
            )
        )

        return K

    def _center_kernel(self, K):
        """
        Double-center kernel matrix:

            Kc = K
                 - row_mean
                 - column_mean
                 + global_mean
        """

        row_mean = K.mean(dim=1, keepdim=True)
        col_mean = K.mean(dim=0, keepdim=True)
        global_mean = K.mean()

        return (
            K
            - row_mean
            - col_mean
            + global_mean
        )

    def forward(self, X, Y):
        """
        X: [B, Dx]
        Y: [B, Dy]

        Returns:
            cka_loss
        """

        B = X.shape[0]

        if B < 2:
            raise ValueError(
                "CKA requires at least 2 samples."
            )

        # RBF kernels
        Kv = self._rbf_kernel(X)
        Kl = self._rbf_kernel(Y)

        # Center kernels
        Kv_c = self._center_kernel(Kv)
        Kl_c = self._center_kernel(Kl)

        # HSIC
        normalization = (B - 1) ** 2

        hsic_vl = torch.trace(
            Kv_c @ Kl_c
        ) / normalization

        hsic_vv = torch.trace(
            Kv_c @ Kv_c
        ) / normalization

        hsic_ll = torch.trace(
            Kl_c @ Kl_c
        ) / normalization

        # Normalized CKA
        denominator = torch.sqrt(
            hsic_vv * hsic_ll
        ) + self.eps

        cka = hsic_vl / denominator

        # CKA regularization loss
        loss = 1.0 - cka

        return loss, cka