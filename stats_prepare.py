import numpy as np

# taken from https://www.econstor.eu/bitstream/10419/222316/1/GLO-DP-0601.pdf
pandemic = [1520, 1665, 1629, 1817, 1885, 1899, 1889, 1918, 1957, 1968, 1981, 2009, 2002, 2014, 2015, 2020]
durations = [15, 3, 2, 4, 1, 3, 6, 1, 1, 2, 2, 2, 3, 2, 2, 3, 5, 2]

sorted(pandemic)
pandemic_diffrance = [pandemic[i+1] - pandemic[i] for i in range(len(pandemic)-1)]

print("pandemic:\n mean =  {:.2f}\n std = {:.2f}"
      .format(np.mean(pandemic_diffrance),
              np.std(pandemic_diffrance)))

print("pandemic durations:\n mean =  {:.2f}\n std = {:.2f}"
      .format(np.mean(durations),
              np.std(durations)))

print(np.random.normal(np.mean(pandemic_diffrance), np.std(pandemic_diffrance), 1))