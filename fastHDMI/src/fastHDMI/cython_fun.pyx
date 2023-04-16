from libc.math cimport log, isfinite
from libc.stdlib cimport calloc, free
cimport cython
from cython cimport floating

ctypedef fused floating_float_double:
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
def joint_to_mi_cython(floating_float_double[:, ::1] joint, floating_float_double forward_euler_a=1., floating_float_double forward_euler_b=1.):
    cdef int i, j
    cdef int joint_shape0 = joint.shape[0]
    cdef int joint_shape1 = joint.shape[1]
    cdef floating_float_double *log_a_marginal = <floating_float_double*>calloc(joint_shape0, sizeof(floating_float_double))
    cdef floating_float_double *log_b_marginal = <floating_float_double*>calloc(joint_shape1, sizeof(floating_float_double))
    cdef floating_float_double temp_sum, log_temp_sum, log_forward_euler_a, log_forward_euler_b, log_joint, output
    
    temp_sum = 0.0
    for i in range(joint_shape0):
        for j in range(joint_shape1):
            log_a_marginal[i] += joint[i, j]
            log_b_marginal[j] += joint[i, j]
        temp_sum += log_a_marginal[i]
    temp_sum *= forward_euler_a * forward_euler_b
    log_temp_sum = log(temp_sum)
    log_forward_euler_a = log(forward_euler_a)
    log_forward_euler_b = log(forward_euler_b)
    
    for i in range(joint_shape0):
        log_a_marginal[i] = log(log_a_marginal[i]) + log_forward_euler_b if isfinite(log(log_a_marginal[i])) else 0.0
    for j in range(joint_shape1):
        log_b_marginal[j] = log(log_b_marginal[j]) + log_forward_euler_a if isfinite(log(log_b_marginal[j])) else 0.0
    
    output = 0.0
    for i in range(joint_shape0):
        for j in range(joint_shape1):
            log_joint = log(joint[i, j]) if isfinite(log(joint[i, j])) else 0.0
            output += joint[i, j] * (log_joint - log_a_marginal[i] - log_b_marginal[j]) * forward_euler_a * forward_euler_b

    output = max(output, 0.0)

    free(log_a_marginal)
    free(log_b_marginal)

    return output
