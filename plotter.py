import numpy as np
import matplotlib.pyplot as plt
import sys
import os

"""
The goal is to plot serialized runs after the fact.
What we want is to take as arguments a list of runFiles.
Where a runfile contains the xs (the number of steps)
and ys (the cost at that step) and the label for the graph.

So the two tasks we have are: 
save this run thing into a pkl file from the training
read this out and generate the plot.
"""

max = int(sys.argv[1]) # the xrange of the model

files = sys.argv[2:] # all other arguments are results files
  
print files

results = []

for x in files:
  with open(x) as f:
    results.append(f.readlines())

xs = []
ys = []
for result in results:
  x = []
  y = []
  for line in result:
    if line.startswith("trailing"):
      words = line.split()
      y.append(float(words[-1]))
    elif line.startswith("iter"):
      words = line.split()
      x.append(words[-1])
    else:
      pass
  xs.append(x)
  ys.append(y)


for x,y,f in zip(xs,ys,files):
  a = np.array(x[:max])
  b = np.array(y[:max])

  plt.plot(a,b, label=os.path.basename(f))

plt.xlabel("training sequences")
plt.ylabel("BPC")
plt.ylim(ymin=0)
plt.legend(loc='upper right')
plt.show()

