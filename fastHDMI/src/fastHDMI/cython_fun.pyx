# cython: language_level=3


from libc.math cimport log, isfinite, pow, fmax
from libc.stdlib cimport calloc, free, qsort, malloc
cimport cython
from cython cimport floating
from cython.parallel import prange
import numpy as _np
ctypedef fused floating_float_double:
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def joint_to_mi_cython(floating_float_double[:, ::1] joint, floating_float_double forward_euler_a=1., floating_float_double forward_euler_b=1.):
    cdef int i, j
    cdef int joint_shape0 = joint.shape[0]
    cdef int joint_shape1 = joint.shape[1]
    cdef floating_float_double *log_a_marginal = <floating_float_double*>calloc(joint_shape0, sizeof(floating_float_double))
    cdef floating_float_double *log_b_marginal = <floating_float_double*>calloc(joint_shape1, sizeof(floating_float_double))
    cdef floating_float_double temp_sum, log_val, log_temp_sum, log_forward_euler_a, log_forward_euler_b, log_joint, output

    if log_a_marginal == NULL or log_b_marginal == NULL:
        raise MemoryError("Failed to allocate memory.")

    log_forward_euler_a = log(forward_euler_a)
    log_forward_euler_b = log(forward_euler_b)
    temp_sum = 0.0

    for i in prange(joint_shape0, nogil=True):
        for j in prange(joint_shape1):
            log_a_marginal[i] += joint[i, j]
            log_b_marginal[j] += joint[i, j]
        temp_sum += log_a_marginal[i]

    temp_sum *= forward_euler_a * forward_euler_b
    log_temp_sum = log(temp_sum)

    for i in prange(joint_shape0, nogil=True):
        log_val = log(log_a_marginal[i])
        log_a_marginal[i] = log_val + log_forward_euler_b if isfinite(log_val) else 0.0

    for j in prange(joint_shape1, nogil=True):
        log_val = log(log_b_marginal[j])
        log_b_marginal[j] = log_val + log_forward_euler_a if isfinite(log_val) else 0.0

    output = 0.0
    for i in prange(joint_shape0, nogil=True):
        for j in prange(joint_shape1):
            log_joint = log(joint[i, j]) if isfinite(log(joint[i, j])) else 0.0
            output += joint[i, j] * (log_joint - log_a_marginal[i] - log_b_marginal[j]) * forward_euler_a * forward_euler_b

    output = max(output, 0.0)

    free(log_a_marginal)
    free(log_b_marginal)

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double hist_obj_cython(floating_float_double[:] x, int D):
    """
    Calculate the value of the objective function for histogram binning.
    """
    cdef int N = x.shape[0]
    cdef int *N_j
    cdef int i, bin_index
    cdef double x_min, x_max, bin_width, result = 0.0, total_sum = 0.0
    cdef floating_float_double xi

    # Initialize min and max
    x_min = x_max = x[0]
    for i in range(N):
        if x[i] < x_min: x_min = x[i]
        if x[i] > x_max: x_max = x[i]

    # Allocate memory for histogram counts
    N_j = <int*>calloc(D, sizeof(int))
    if N_j == NULL:
        raise MemoryError("Failed to allocate memory for histogram counts.")

    # Calculate bin width
    bin_width = (x_max - x_min) / D

    # Histogram calculation
    for i in range(N):
        xi = x[i]
        bin_index = <int>((xi - x_min) / bin_width)
        if bin_index >= D:
            bin_index = D - 1
        N_j[bin_index] += 1

    # Objective function calculation
    for i in range(D):
        if N_j[i] > 0:
            result += N_j[i] * log(N_j[i])
            total_sum += N_j[i]

    result += total_sum * log(D) - (D - 1 + log(D) ** 2.5)

    # Free allocated memory
    free(N_j)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int num_of_bins_cython(floating_float_double[:] x):
    """
    Calculate the optimal number of bins for histogram binning.
    """
    cdef int D, max_D = 2
    cdef double max_obj = -1e20, current_obj
    cdef int N = x.shape[0]

    # Search for the optimal number of bins from 2 to 100
    for D in range(2, 100):
        current_obj = hist_obj_cython(x, D)
        if current_obj > max_obj:
            max_obj = current_obj
            max_D = D

    return max_D

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double binning_MI_cython(floating_float_double[:] a, floating_float_double[:] b):
    """
    Calculate mutual information using binning for two 1-D arrays a and b.
    """
    cdef int N = a.shape[0]
    cdef int D_a = num_of_bins_cython(a)
    cdef int D_b = num_of_bins_cython(b)
    cdef int *histogram
    cdef double *normalized_histogram_ptr
    cdef double[:, ::1] joint_2d
    cdef int i, j, bin_index_a, bin_index_b
    cdef double a_min, a_max, b_min, b_max, bin_width_a, bin_width_b, total = 0.0
    cdef floating_float_double ai, bi

    # Initialize min and max for a and b
    a_min, a_max = a[0], a[0]
    b_min, b_max = b[0], b[0]
    for i in range(N):
        if a[i] < a_min: a_min = a[i]
        if a[i] > a_max: a_max = a[i]
        if b[i] < b_min: b_min = b[i]
        if b[i] > b_max: b_max = b[i]

    # Allocate memory for 2D histogram
    histogram = <int*>calloc(D_a * D_b, sizeof(int))
    if histogram == NULL:
        raise MemoryError("Failed to allocate memory for 2D histogram.")

    # Calculate bin widths
    bin_width_a = (a_max - a_min) / D_a
    bin_width_b = (b_max - b_min) / D_b

    # 2D Histogram calculation
    for i in range(N):
        ai = a[i]
        bi = b[i]
        bin_index_a = <int>((ai - a_min) / bin_width_a)
        bin_index_b = <int>((bi - b_min) / bin_width_b)
        if bin_index_a >= D_a: bin_index_a = D_a - 1
        if bin_index_b >= D_b: bin_index_b = D_b - 1
        histogram[bin_index_a * D_b + bin_index_b] += 1

    # Allocate memory for normalized histogram
    normalized_histogram_ptr = <double *> calloc(D_a * D_b, sizeof(double))
    if normalized_histogram_ptr == NULL:
        raise MemoryError("Failed to allocate memory for normalized histogram.")

    # Normalize histogram to get joint probability
    total = 0.0
    for i in range(D_a * D_b):
        total += histogram[i]
    for i in range(D_a):
        for j in range(D_b):
            normalized_histogram_ptr[i * D_b + j] = histogram[i * D_b + j] / total

    # Wrap the normalized histogram pointer as a 2D memory view
    joint_2d = <double[:D_a, :D_b]>normalized_histogram_ptr

    # Calculate mutual information
    mi = joint_to_mi_cython(joint_2d)

    # Free allocated memory
    free(histogram)
    free(normalized_histogram_ptr)

    return mi

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double binning_MI_discrete_cython(floating_float_double[:] a, floating_float_double[:] b):
    """
    Calculate mutual information using binning for two 1-D arrays a and b.
    """
    cdef int N = a.shape[0]
    cdef int D_a = len(_np.unique(a))
    cdef int D_b = num_of_bins_cython(b)
    cdef int *histogram
    cdef double *normalized_histogram_ptr
    cdef double[:, ::1] joint_2d
    cdef int i, j, bin_index_a, bin_index_b
    cdef double a_min, a_max, b_min, b_max, bin_width_a, bin_width_b, total = 0.0
    cdef floating_float_double ai, bi

    # Initialize min and max for a and b
    a_min, a_max = a[0], a[0]
    b_min, b_max = b[0], b[0]
    for i in range(N):
        if a[i] < a_min: a_min = a[i]
        if a[i] > a_max: a_max = a[i]
        if b[i] < b_min: b_min = b[i]
        if b[i] > b_max: b_max = b[i]

    # Allocate memory for 2D histogram
    histogram = <int*>calloc(D_a * D_b, sizeof(int))
    if histogram == NULL:
        raise MemoryError("Failed to allocate memory for 2D histogram.")

    # Calculate bin widths
    bin_width_a = (a_max - a_min) / D_a
    bin_width_b = (b_max - b_min) / D_b

    # 2D Histogram calculation
    for i in range(N):
        ai = a[i]
        bi = b[i]
        bin_index_a = <int>((ai - a_min) / bin_width_a)
        bin_index_b = <int>((bi - b_min) / bin_width_b)
        if bin_index_a >= D_a: bin_index_a = D_a - 1
        if bin_index_b >= D_b: bin_index_b = D_b - 1
        histogram[bin_index_a * D_b + bin_index_b] += 1

    # Allocate memory for normalized histogram
    normalized_histogram_ptr = <double *> calloc(D_a * D_b, sizeof(double))
    if normalized_histogram_ptr == NULL:
        raise MemoryError("Failed to allocate memory for normalized histogram.")

    # Normalize histogram to get joint probability
    total = 0.0
    for i in range(D_a * D_b):
        total += histogram[i]
    for i in range(D_a):
        for j in range(D_b):
            normalized_histogram_ptr[i * D_b + j] = histogram[i * D_b + j] / total

    # Wrap the normalized histogram pointer as a 2D memory view
    joint_2d = <double[:D_a, :D_b]>normalized_histogram_ptr

    # Calculate mutual information
    mi = joint_to_mi_cython(joint_2d)

    # Free allocated memory
    free(histogram)
    free(normalized_histogram_ptr)

    return mi