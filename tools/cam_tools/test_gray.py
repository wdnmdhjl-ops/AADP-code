import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


grayscale_cam = np.load('grayscale_cam.npy')
plt.figure(figsize=(8, 6))
sns.heatmap(grayscale_cam, annot=True, cmap='viridis', fmt=".1f")
plt.title('heatmap')
plt.show()

