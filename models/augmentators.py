import torch
import torch.nn as nn
import torch.nn.functional as F


class ChromatogramCropper(nn.Module):
    def __init__(self, scale, size=None, resize=True, mode='linear', p=0.5):
        super(ChromatogramCropper, self).__init__()
        self.scale = scale
        self.size = size
        self.resize = resize
        self.mode = mode
        self.p = p

        if self.mode =='linear':
            self.align_corners = False
        elif self.mode == 'nearest' or self.mode == 'area':
            self.align_corners = None
    
    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch
        
        _, _, l = chromatogram_batch.size()
        target_len = torch.round(
            l * torch.empty(
                1).uniform_(self.scale[0], self.scale[1])).int().item()
        start = torch.randint(low=0, high=l - target_len + 1, size=(1,)).item()
        end = start + target_len
        crop = chromatogram_batch[:, :, start:end]

        if self.resize:
            if not self.size:
                self.size = l

            return F.interpolate(
                crop,
                size=self.size,
                mode=self.mode,
                align_corners=self.align_corners
            )

        return crop


class ChromatogramJitterer(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        augment_precursor=True,
        mean=0,
        std=1,
        p=0.5,
        device='cpu'
    ):
        super(ChromatogramJitterer, self).__init__()
        self.mz_bins = mz_bins
        self.length = None
        self.mean = mean
        self.std = std
        self.p = p
        self.device = device

        if augment_precursor:
            self.mz_bins += self.mz_bins // 6

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        if not self.length:
            self.length = chromatogram_batch.size()[2]

        noise = (
            torch.FloatTensor(
                self.mz_bins, self.length
            ).normal_(self.mean, self.std)
        ).to(self.device)

        chromatogram_batch[:, 0:self.mz_bins] = F.relu(
            chromatogram_batch[:, 0:self.mz_bins] + noise)

        return chromatogram_batch


class ChromatogramNormalizer(nn.Module):
    def __init__(self, mz_bins=6, standardize=False):
        super(ChromatogramNormalizer, self).__init__()
        self.mz_bins = mz_bins
        self.standardize = standardize

    def forward(self, chromatogram_batch):
        M = self.mz_bins // 6
        start = self.mz_bins + M
        end = start + 6

        if self.standardize:
            sigma, mu = torch.std_mean(
                torch.cat(
                    [chromatogram_batch[:, 0:self.mz_bins],
                    chromatogram_batch[:, start:end]], dim=1
                ),
                dim=1,
                keepdim=True
            )
            sigma += 1e-7
            chromatogram_batch[:, 0:self.mz_bins] = (
                (chromatogram_batch[:, 0:self.mz_bins] - mu) / sigma)
            chromatogram_batch[:, start:end] = (
                (chromatogram_batch[:, start:end] - mu) / sigma)

            sigma, mu = torch.std_mean(
                chromatogram_batch[:, self.mz_bins:start], dim=1, keepdim=True)
            sigma += 1e-7
            chromatogram_batch[:, self.mz_bins:start] = (
                (chromatogram_batch[:, self.mz_bins:start] - mu) / sigma)
        else:
            x = torch.cat(
                [chromatogram_batch[:, 0:self.mz_bins],
                chromatogram_batch[:, start:end]],
                dim=1
            )
            x_min, _ = torch.min(x, dim=1, keepdim=True)
            x_max, _ = torch.max(x, dim=1, keepdim=True)
            x_max += 1e-7
            chromatogram_batch[:, 0:self.mz_bins] = (
                (chromatogram_batch[:, 0:self.mz_bins] - x_min) /
                (x_max - x_min))
            chromatogram_batch[:, start:end] = (
                (chromatogram_batch[:, start:end] - x_min) /
                (x_max - x_min))

            x_min, _ = torch.min(
                chromatogram_batch[:, self.mz_bins:start], dim=1, keepdim=True)
            x_max, _ = torch.max(
                chromatogram_batch[:, self.mz_bins:start], dim=1, keepdim=True)
            x_max += 1e-7
            chromatogram_batch[:, self.mz_bins:start] = (
                (chromatogram_batch[:, self.mz_bins:start] - x_min) / 
                (x_max - x_min))

        x_min = torch.min(chromatogram_batch[:, -2:-1])
        x_max = torch.max(chromatogram_batch[:, -2:-1]) + 1e-7
        chromatogram_batch[:, -2:-1] = (
                (chromatogram_batch[:, -2:-1] - x_min) / (x_max - x_min))

        x_min, x_max = 1, 3
        chromatogram_batch[:, -1:] = (
                (chromatogram_batch[:, -1:] - x_min) / 
                (x_max - x_min))

        return chromatogram_batch


class ChromatogramScaler(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        augment_precursor=True,
        scale_independently=False,
        scale=(0.875, 1.125),
        p=0.5,
        device='cpu'
    ):
        super(ChromatogramScaler, self).__init__()
        self.mz_bins = mz_bins
        self.num_factors = 6
        self.scale_independently = scale_independently
        self.scale = scale
        self.p = p
        self.device = device

        if augment_precursor:
            self.mz_bins += self.mz_bins // 6
            self.num_factors += 1

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        if self.scale_independently:
            scaling_factors = (
                torch.FloatTensor(
                    self.num_factors, 1).uniform_(self.scale[0], self.scale[1])
            ).to(self.device)

            if self.mz_bins > 6:
                scaling_factors = (
                    scaling_factors.repeat_interleave(
                        self.mz_bins // self.num_factors, dim=0)
                ).to(self.device)
        else:
            scaling_factors = (
                torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1])
            ).to(self.device)

        chromatogram_batch[:, 0:self.mz_bins] = (
            chromatogram_batch[:, 0:self.mz_bins] * scaling_factors)

        return chromatogram_batch


class ChromatogramShuffler(nn.Module):
    def __init__(self, mz_bins=6, p=0.5):
        super(ChromatogramShuffler, self).__init__()
        self.mz_bins = mz_bins
        self.p = p

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        shuffled_indices = torch.randperm(6)

        M = self.mz_bins // 6
        start = self.mz_bins + M
        end = start + 6

        if M == 1:
            chromatogram_batch[:, 0:self.mz_bins] = (
                chromatogram_batch[:, 0:self.mz_bins][:, shuffled_indices])
        else:
            b, _, n = chromatogram_batch.size()
            chromatogram_batch[:, 0:self.mz_bins] = (
                chromatogram_batch[:, 0:self.mz_bins].reshape(
                    b, 6, M, n)[:, shuffled_indices].reshape(b, -1, n)
            )

        chromatogram_batch[:, start:end] = (
            chromatogram_batch[:, start:end][:, shuffled_indices])

        return chromatogram_batch


class ChromatogramSpectraMasker(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        augment_precursor=False,
        F=1,
        m_F=1,
        p=0.5
    ):
        super(ChromatogramSpectraMasker, self).__init__()
        self.v = mz_bins
        self.F = F
        self.m_F = m_F
        self.p = p

        if augment_precursor:
            self.v += self.v // 6

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        for i in range(self.m_F):
            f = torch.randint(0, self.F + 1, (1,)).item()
            f_0 = torch.randint(0, self.v - f, (1,)).item()
            chromatogram_batch[:, f_0:f_0 + f] = 0

        return chromatogram_batch


class ChromatogramTimeMasker(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        augment_precursor=True,
        T=5,
        m_T=1,
        p=0.5
    ):
        super(ChromatogramTimeMasker, self).__init__()
        self.mz_bins = mz_bins
        self.tau = None
        self.T = T
        self.m_T = m_T
        self.p = p

        if augment_precursor:
            self.mz_bins += self.mz_bins // 6

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch
        
        if not self.tau:
            self.tau = chromatogram_batch.size()[2]

        for i in range(self.m_T):
            t = torch.randint(0, self.T + 1, (1,)).item()
            t_0 = torch.randint(0, self.tau - t, (1,)).item()
            chromatogram_batch[:, 0:self.mz_bins, t_0:t_0 + t] = 0

        return chromatogram_batch


class ChromatogramTraceMasker(nn.Module):
    def __init__(self, mz_bins, min_only=True, p=0.5):
        super(ChromatogramTraceMasker, self).__init__()
        self.mz_bins = mz_bins
        self.min_only = min_only
        self.p = p

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        if self.min_only:
            start = self.mz_bins - self.mz_bins // 6
        else:
            weights = torch.sum(chromatogram_batch[:, -8:-2, 0], dim=0).reshape(6)
            weights_max = torch.max(weights)
            weights = (weights_max - weights) / weights_max
            start = torch.multinomial(weights, 1).item() * self.mz_bins // 6
            
        chromatogram_batch[:, start:start + self.mz_bins // 6, :] = 0

        return chromatogram_batch


class ChromatogramTranslator(nn.Module):
    def __init__(self, mz_bins, augment_precursor=True, dist=10, p=0.5):
        super(ChromatogramTranslator, self).__init__()
        self.mz_bins = mz_bins
        self.dist = dist
        self.p = p

        if augment_precursor:
            self.mz_bins += self.mz_bins // 6

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch
        
        move_by = torch.randint(-self.dist, self.dist + 1, size=(1,)).item()

        if move_by < 0:
            chromatogram_batch[:, 0:self.mz_bins, 0:move_by] = (
                chromatogram_batch[:, 0:self.mz_bins, -move_by:])
            chromatogram_batch[:, 0:self.mz_bins, move_by:] = 0
        elif move_by > 0:
            chromatogram_batch[:, 0:self.mz_bins, move_by:] = (
                chromatogram_batch[:, 0:self.mz_bins, 0:-move_by])
            chromatogram_batch[:, 0:self.mz_bins, 0:move_by] = 0

        return chromatogram_batch


class SelfSupervisedGlobalAugmentatorOne(nn.Module):
    def __init__(self, scale, size, mode, mz_bins, device, mean, std):
        super(SelfSupervisedGlobalAugmentatorOne, self).__init__()
        self.augmentator = nn.Sequential(
            ChromatogramCropper(scale=scale, size=size, mode=mode, p=1),
            ChromatogramScaler(mz_bins=mz_bins, scale=scale, device=device),
            ChromatogramJitterer(
                mz_bins=mz_bins,
                mean=mean,
                std=std,
                device=device
            )
        )
    
    def forward(self, chromatogram_batch):
        return self.augmentator(chromatogram_batch)


class SelfSupervisedGlobalAugmentatorTwo(nn.Module):
    def __init__(self, scale, size, mode, mz_bins, num_F, m_F, T, m_T):
        super(SelfSupervisedGlobalAugmentatorTwo, self).__init__()
        self.augmentator = nn.Sequential(
            ChromatogramCropper(scale=scale, size=size, mode=mode, p=1),
            ChromatogramTraceMasker(mz_bins=mz_bins),
            ChromatogramSpectraMasker(mz_bins=mz_bins, F=num_F, m_F=m_F),
            ChromatogramTimeMasker(mz_bins=mz_bins, T=T, m_T=m_T)
        )
    
    def forward(self, chromatogram_batch):
        return self.augmentator(chromatogram_batch)


class SelfSupervisedLocalAugmentator(nn.Module):
    def __init__(
        self,
        scale,
        size,
        mode,
        mz_bins,
        device,
        mean,
        std,
        num_F,
        m_F,
        T,
        m_T
    ):
        super(SelfSupervisedLocalAugmentator, self).__init__()
        self.augmentator = nn.Sequential(
            ChromatogramCropper(scale=scale, size=size, mode=mode, p=1),
            ChromatogramScaler(mz_bins=mz_bins, scale=scale, device=device, p=0.8),
            ChromatogramJitterer(
                mz_bins=mz_bins,
                mean=mean,
                std=std,
                device=device,
                p=0.8
            ),
            ChromatogramTraceMasker(mz_bins=mz_bins, p=0.2),
            ChromatogramSpectraMasker(mz_bins=mz_bins, F=num_F, m_F=m_F, p=0.2),
            ChromatogramTimeMasker(mz_bins=mz_bins, T=T, m_T=m_T, p=0.2)
        )
    
    def forward(self, chromatogram_batch):
        return self.augmentator(chromatogram_batch)


class SemiSupervisedStrongAugmentator(nn.Module):
    def __init__(self, mz_bins, device, scale, mean, std, num_F, m_F, T, m_T):
        super(SemiSupervisedStrongAugmentator, self).__init__()
        self.augmentator = nn.Sequential(
            ChromatogramScaler(mz_bins=mz_bins, scale=scale, device=device),
            ChromatogramJitterer(
                mz_bins=mz_bins,
                mean=mean,
                std=std,
                device=device
            ),
            ChromatogramTraceMasker(mz_bins=mz_bins),
            ChromatogramSpectraMasker(mz_bins=mz_bins, F=num_F, m_F=m_F),
            ChromatogramTimeMasker(mz_bins=mz_bins, T=T, m_T=m_T)
        )
    
    def forward(self, chromatogram_batch):
        return self.augmentator(chromatogram_batch)


class SemiSupervisedWeakAugmentator(nn.Module):
    def __init__(self, mz_bins, scale, device):
        super(SemiSupervisedWeakAugmentator, self).__init__()
        self.augmentator = ChromatogramScaler(
            mz_bins=mz_bins,
            scale=scale,
            p=1,
            device=device
        )
    
    def forward(self, chromatogram_batch):
        return self.augmentator(chromatogram_batch)
