import logging
import numpy as np

from uncoverml import mpiops
log = logging.getLogger(__name__)


class CentreTransform:
    def __init__(self):
        self.mean = None

    def __call__(self, x):
        x = x.astype(float)
        if self.mean is None:
            self.mean = mpiops.mean(x)
        x -= self.mean
        return x


class StandardiseTransform:
    def __init__(self):
        self.mean = None
        self.sd = None

    def __call__(self, x):
        x = x.astype(float)
        if self.sd is None or self.mean is None:
            self.mean = mpiops.mean(x)
            self.sd = mpiops.sd(x)

        # Centre
        x -= self.mean

        # remove dimensions with no st. dev. (and hence no info)
        zero_mask = self.sd == 0.
        if zero_mask.sum() > 0:
            x = x[:, ~zero_mask]
            sd = self.sd[~zero_mask]
        else:
            sd = self.sd
        x /= sd
        return x


class PositiveTransform:

    def __init__(self, stabilizer=1.0e-6):
        self.min = None
        self.stabilizer = stabilizer

    def __call__(self, func, x):
        x = x.astype(float)
        if self.min is None:
            self.min = mpiops.minimum(x)

        # remove min
        x -= self.min

        # add small +ve value for stable log
        x += self.stabilizer
        return func(x)


class LogTransform(PositiveTransform):

    def __init__(self, stabilizer=1.0e-6):
        super(LogTransform, self).__init__(stabilizer)

    def __call__(self, *args):
        return super(LogTransform, self).__call__(np.ma.log, *args)


class SqrtTransform(PositiveTransform):

    def __init__(self, stabilizer=1.0e-6):
        super(SqrtTransform, self).__init__(stabilizer)

    def __call__(self, *args):
        return super(SqrtTransform, self).__call__(np.ma.sqrt, *args)


class WhitenTransform:
    def __init__(self, keep_fraction=None, n_components=None, variation_fraction=None):
        self.mean = None
        self.eigvals = None
        self.eigvecs = None
        self.keep_fraction = keep_fraction
        self.n_components = n_components
        self.variation_fraction = variation_fraction
        null_check = [x is None for x in [keep_fraction, n_components, variation_fraction]]
        assert np.sum(list(null_check)) == 2, "only one of keep_fraction, n_components, " \
                                              "variation_fraction can be non null"
        self.explained_ratio = {}

    def __call__(self, x):
        x = x.astype(float)
        if self.mean is None or self.eigvals is None or self.eigvecs is None:
            self.mean = mpiops.mean(x)
            num_points = np.sum(mpiops.comm.gather(x.shape[0]))
            num_covariates = x.shape[1]
            log.info(f"Matrix for eigenvalue decomposition has shape {num_points, num_covariates}")
            self.eigvals, self.eigvecs = mpiops.eigen_decomposition(x)
            ndims = x.shape[1]
            # make sure 1 <= keepdims <= ndims
            self.explained_ratio = {
                ndims - i: r * 100 for i, r in enumerate(np.abs(self.eigvals[-ndims:])/np.sum(np.abs(
                    self.eigvals)))
            }

            if self.n_components is None and self.variation_fraction is None:
                self.keepdims = min(max(1, int(ndims * self.keep_fraction)), ndims)
            elif self.n_components is None and self.keep_fraction is None:
                explained_ratio_cum_sum = np.cumsum(list(self.explained_ratio.values())[::-1])
                self.keepdims = np.searchsorted(explained_ratio_cum_sum > self.variation_fraction*100, True) + 1
                log.info(f"Variation percentage {self.variation_fraction*100} requires only {self.keepdims} PCs")
            else:
                self.keepdims = self.n_components
                assert self.n_components < ndims, "More components demanded than features. Not possible! \n " \
                                                  "Please reduce n_components or increase the number of features"

        mat = self.eigvecs[:, - self.keepdims:]
        vec = self.eigvals[np.newaxis, - self.keepdims:]
        x = np.ma.dot(x - self.mean, mat, strict=True) / np.sqrt(vec)

        return x
