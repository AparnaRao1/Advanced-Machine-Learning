from data_generation import generate_separable, generate_overlap
from training import train_svm
from visualization import plot_data, plot_decision_boundary, plot_margin_lines
import matplotlib.pyplot as plt

# Hard margin case
X, y = generate_separable()
plot_data(X, y, "Linearly Separable Data")

w, b = train_svm(X, y)

plt.scatter(X[y==1][:,0], X[y==1][:,1])
plt.scatter(X[y==-1][:,0], X[y==-1][:,1])
plot_decision_boundary(w, b, X)
plot_margin_lines(w, b, X)
plt.title("Hard Margin SVM")
plt.show()

# Soft margin case
X2, y2 = generate_overlap()
plot_data(X2, y2, "Overlapping Data (Soft Margin)")

w2, b2 = train_svm(X2, y2)

plt.scatter(X2[y2==1][:,0], X2[y2==1][:,1])
plt.scatter(X2[y2==-1][:,0], X2[y2==-1][:,1])
plot_decision_boundary(w2, b2, X2)
plot_margin_lines(w2, b2, X2)
plt.title("Soft Margin SVM")
plt.show()