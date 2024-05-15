import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the model's embeddings
embeddings = np.load('/home/ubuntu/CG-Labs-HAN/embeddings.npy')

# Perform t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=300)
tsne_results = tsne.fit_transform(embeddings)

# Plot the results
plt.figure(figsize=(16,10))
plt.scatter(tsne_results[:,0], tsne_results[:,1])
plt.xlabel('t-SNE feature 0')
plt.ylabel('t-SNE feature 1')
plt.title('t-SNE visualization of embeddings')

# Save the plot as an image file
plt.savefig('tsne_visualization.png')
