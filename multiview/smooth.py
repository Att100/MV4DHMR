import numpy as np

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = None
        self.dx_prev = 0.0
        self.last_time = None

    @staticmethod
    def alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    @staticmethod
    def lowpass(x, x_prev, alpha):
        return alpha * x + (1 - alpha) * x_prev

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x

        # dx
        dx = (x - self.x_prev) * self.freq
        alpha_d = self.alpha(self.d_cutoff, self.freq)
        dx_hat = self.lowpass(dx, self.dx_prev, alpha_d)

        # cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self.alpha(cutoff, self.freq)

        x_hat = self.lowpass(x, self.x_prev, alpha)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
