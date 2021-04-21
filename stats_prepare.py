import numpy as np

# taken from https://en.wikipedia.org/wiki/List_of_epidemics
pandemic = [-1200, -426, 165, 217, 250, 541, 590, 627, 638, 664, 698, 735, 746, 1346, 1489, 1510, 1519, 1545, 1557, 1561, 1563, 1576, 1582, 1592, ]
durations = [3, 3, 15, 5, 16, 8, 1, 2, 2, 25, 3, 2, 11, 7, 1, 1, 2, 3, 2, 2, 2, 4, 2, 4]
die_rate = [0.01 for i in range(len(durations))]

pandemic_diffrance = [pandemic[i+1] - pandemic[i] for i in range(len(pandemic)-1)]

print("pandemic:\n mean =  {:.2f}\n std = {:.2f}"
      .format(np.mean(pandemic_diffrance),
              np.std(pandemic_diffrance)))

print("pandemic durations:\n mean =  {:.2f}\n std = {:.2f}"
      .format(np.mean(durations),
              np.std(durations)))
