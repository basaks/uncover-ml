import logging
import pickle

import numpy as np
from mpi4py import MPI

log = logging.getLogger(__name__)

# We're having trouble with the MPI pickling and 64bit integers
# MPI.pickle.dumps = pickle.dumps
# MPI.pickle.loads = pickle.loads

comm = MPI.COMM_WORLD
"""module-level MPI 'world' object representing all connected nodes
"""

chunks = comm.Get_size()
"""int: the total number of nodes in the MPI world
"""

chunk_index = comm.Get_rank()
"""int: the index (from zero) of this node in the MPI world. Also known as
the rank of the node.
"""


def run_once(f, *args, **kwargs):
    """Run a function on one node, broadcast result to all
    This function evaluates a function on a single node in the MPI world,
    then broadcasts the result of that function to every node in the world.
    Parameters
    ----------
    f : callable
        The function to be evaluated. Can take arbitrary arguments and return
        anything or nothing
    args : optional
        Other positional arguments to pass on to f
    kwargs : optional
        Other named arguments to pass on to f
    Returns
    -------
    result
        The value returned by f
    """
    if chunk_index == 0:
        f_result = f(*args, **kwargs)
    else:
        f_result = None
    result = comm.bcast(f_result, root=0)
    return result


def sum_axis_0(x, y, dtype):
    s = np.ma.sum(np.ma.vstack((x, y)), axis=0)
    return s


def max_axis_0(x, y, dtype):
    s = np.amax(np.array([x, y]), axis=0)
    return s


def min_axis_0(x, y, dtype):
    s = np.amin(np.array([x, y]), axis=0)
    return s


def unique(sets1, sets2, dtype):
    per_dim = zip(sets1, sets2)
    out_sets = [np.unique(np.concatenate(k, axis=0)) for k in per_dim]
    return out_sets

unique_op = MPI.Op.Create(unique, commute=True)
sum0_op = MPI.Op.Create(sum_axis_0, commute=True)
max0_op = MPI.Op.Create(max_axis_0, commute=True)
min0_op = MPI.Op.Create(min_axis_0, commute=True)


def count(x):
    x_n_local = np.ma.count(x, axis=0).ravel()
    x_n = comm.allreduce(x_n_local, op=sum0_op)
    still_masked = np.ma.count_masked(x_n)
    if still_masked != 0:
        log.info('Reported subcounts: ' + ', '.join([str(s) for s in x_n]))
        raise ValueError("Can't compute count: subcounts are still masked")
    if hasattr(x_n, 'mask'):
        x_n = x_n.data
    return x_n


def outer_count(x):

    xnotmask = (~x.mask).astype(float)
    x_n_outer_local = np.dot(xnotmask.T, xnotmask)
    x_n_outer = comm.allreduce(x_n_outer_local)

    return x_n_outer


def mean(x):
    x_n = count(x)
    x_sum_local = np.ma.sum(x, axis=0)
    x_sum = comm.allreduce(x_sum_local, op=sum0_op)
    still_masked = np.ma.count_masked(x_sum)
    if still_masked != 0:
        log.info('Reported x_sum: ' + ', '.join([str(s) for s in x_sum]))
        raise ValueError("Can't compute mean: At least 1 column has nodata")
    if hasattr(x_sum, 'mask'):
        x_sum = x_sum.data
    mean = x_sum / x_n
    return mean


def minimum(x):
    x_min_local = np.ma.min(x, axis=0)
    x_min = comm.allreduce(x_min_local, op=min0_op)
    still_masked = np.ma.count_masked(x_min)
    if still_masked != 0:
        log.info('Reported x_min: ' + ', '.join([str(s) for s in x_min]))
        raise ValueError("Can't compute mean: At least 1 column has nodata")
    if hasattr(x_min, 'mask'):
        x_min = x_min.data
    return x_min


def sd(x):
    x_mean = mean(x)
    delta_mean = mean(power((x - x_mean), 2))
    sd = np.sqrt(delta_mean)
    return sd


def power(x, exp):
    if np.ma.count_masked(x) == 0:
        return np.ma.masked_array(x.data**2, mask=False)
    m = np.where(~x.mask)
    xe = x[m]
    xe = xe**exp
    result = x.copy()
    result[m] = xe
    return result


def outer(x):
    x_outer_local = np.ma.dot(x.T, x)
    out = comm.allreduce(x_outer_local)
    still_masked = np.ma.count_masked(out)
    if still_masked != 0:
        log.info("=========still_masked =================")
        if chunk_index == 0:
            for s in out:
                log.info(s)
        # log.info('Reported out: ' + ', '.join([str(s) for s in out]))
        raise ValueError("Can't compute outer product:"
                         " completely missing columns!")
    if hasattr(out, 'mask'):
        out = out.data
    return out


def covariance(x):
    x_mean = mean(x)
    cov = outer(x - x_mean) / outer_count(x)
    return cov


def eigen_decomposition(x):
    log.info(f"Eigenvalue matrix shape in process {chunk_index}: {x.shape}")
    eigvals, eigvecs = np.linalg.eigh(covariance(x))
    return eigvals, eigvecs


def random_full_points(x, Napprox):
    npernode = int(np.round(Napprox / chunks))
    npernode = min(npernode, len(x))  # Make sure the dataset is upper bound

    rinds = np.random.permutation(len(x))  # random choice of indices

    # Get random points per node
    x_p_node = []
    count = 0
    for i in rinds:
        if np.ma.count_masked(x[i]) > 0:
            continue
        if count >= npernode:
            break
        x_p_node.append(x[i])
        count += 1

    # one chunk can have all of one or more covariates masked
    x_p_node = np.vstack(x_p_node) if len(x_p_node) else None

    all_x_p_node = comm.allgather(x_p_node)
    # filter out the None chunks
    filter_all_x_p_node = [x for x in all_x_p_node if x is not None]

    # Gather all random points
    x_p = np.vstack(filter_all_x_p_node)
    return x_p
