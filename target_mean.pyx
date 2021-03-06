# distutils: language=c++
import cython
import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map


ctypedef np.int_t INT
ctypedef np.double_t DOUBLE


cpdef target_mean_v3(data, y_name, x_name):
  cdef long nrow = data.shape[0]
  cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
  cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
  cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

  target_mean_v3_impl(result, y, x, nrow)
  return result


cdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x, const long nrow):
  cdef dict value_dict = dict()
  cdef dict count_dict = dict()

  cdef long i
  for i in range(nrow):
    if x[i] not in value_dict.keys():
      value_dict[x[i]] = y[i]
      count_dict[x[i]] = 1
    else:
      value_dict[x[i]] += y[i]
      count_dict[x[i]] += 1

  i=0
  for i in range(nrow):
    result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)


@cython.boundscheck(False)
@cython.wraparound(False)
def target_mean_np2(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[DOUBLE, ndim=1] result = np.zeros((nrow,), dtype=np.float64)
    cdef np.ndarray[INT, ndim=1] y = data[y_name].values
    cdef np.ndarray[INT, ndim=1] x = data[x_name].values

    target_mean_np2_impl(result, y, x, nrow)
    return result


cdef void target_mean_np2_impl(np.ndarray[DOUBLE, ndim=1] result, np.ndarray[INT, ndim=1] y, np.ndarray[INT, ndim=1] x, const long nrow):
    cdef unordered_map[INT, INT] value_dict
    cdef unordered_map[INT, INT] count_dict

    cdef long i
    for i in range(nrow):
      value_dict[x[i]] += y[i]
      count_dict[x[i]] += 1

    i=0
    for i in range(nrow):
      result[i] = <double>(value_dict[x[i]] - y[i]) / <double>(count_dict[x[i]]-1)
