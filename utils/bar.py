import matplotlib.pyplot as plt

data = [3, 3.3, 5.7, 6.2, 4.1]
labels = ['Proposed', '50%GPS', '20%GPS', '10%GPS', 'Markov Chain']

colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
plt.bar(range(len(data)), data,color=colorlist, tick_label=labels)
plt.ylabel("Average Discrete Trajectory Distance")
plt.title("Comparison of average discrete trajectory distance")
plt.show()