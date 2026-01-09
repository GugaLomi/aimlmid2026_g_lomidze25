import numpy as np
import matplotlib.pyplot as plt

x = np.array([-9.8, -5.8, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0])
y = np.array([-8.5, -6.7, -4.8, -3.1, -0.9, 0.8, 2.9, 4.7, 6.5, 8.3])

# Pearson correlation
r = np.corrcoef(x, y)[0, 1]
print("Pearson correlation coefficient:", r)

# Best-fit line
m, b = np.polyfit(x, y, 1)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, m*x + b, color="red", label="Best-fit line")

plt.title("Correlation Between X and Y")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.grid(True)

plt.savefig("correlation_plot.png")
plt.show()
