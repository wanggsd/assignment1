import time
from collections import defaultdict
import numpy as np
import pandas as pd
from target_mean import target_mean_np2, target_mean_v3


def timeit(n_rep):
  def _timeit(f):
    def time_and_run(*args, **kwargs):
      times = list()
      print(f"Testing {f.__name__} {n_rep} time(s)...")
      for _ in range(n_rep):
        t0 = time.time()
        res = f(*args, **kwargs)
        times.append(time.time() - t0)
      print(f"Mean time consumption: {np.mean(times) * 1000:.5f}ms\n")
      return res
    return time_and_run
  return _timeit

@timeit(1)
def target_mean_v1(data, y_name, x_name):
  result = np.zeros(data.shape[0])
  for i in range(data.shape[0]):
    groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
    result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
  return result


@timeit(100)
def target_mean_np1(data, y_name, x_name):
  """A little optimization pure Python"""
  xs = data[x_name].values
  ys = data[y_name].values
  res = np.zeros(data.shape[0])
  value_dict = defaultdict(int)
  count_dict = defaultdict(int)
  for x, y in zip(xs, ys):
    value_dict[x] += y
    count_dict[x] += 1
  for idx, x in enumerate(xs):
    res[idx] = (value_dict[x] - ys[idx]) / (count_dict[x] - 1)
  return res


if __name__ == "__main__":
  nr = 10000
  print("# of rows in test data:", nr)
  y = np.random.randint(2, size=(nr, 1))
  x = np.random.randint(10, size=(nr, 1))
  data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

  print("\n====================")
  print("Demos shown in class")
  print("====================")
  res1 = target_mean_v1(data, 'y', 'x')
  res0 = timeit(100)(target_mean_v3)(data, 'y', 'x')
  assert np.allclose(np.linalg.norm(res1 - res0), 0)

  print("\n========================")
  print("Pure Python Optimization")
  print("========================")
  res2 = target_mean_np1(data, 'y', 'x')
  assert np.allclose(np.linalg.norm(res1 - res2), 0)

  print("\n========================")
  print("Optimization with Cython")
  print("========================")
  res3 = timeit(100)(target_mean_np2)(data, 'y', 'x')
  assert np.allclose(np.linalg.norm(res1 - res3), 0)
