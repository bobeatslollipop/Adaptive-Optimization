import numpy as np

def generate_synthetic_data(num_samples):
    # Set the dimension of each sample
    dim = 12

    # Generate random weights and bias from a standard Gaussian distribution
    w = np.random.randn(dim)
    b = np.random.randn()

    # Generate samples: each x is a 12-dimensional vector from a standard Gaussian
    X = np.random.randn(num_samples, dim)

    # Compute the linear threshold function <w, x> + b
    LTF = X.dot(w) + b

    # Determine labels by applying the threshold; y=1 if non-negative, else y=0
    Y = (LTF >= 0).astype(int)

    # Introduce noise: flip labels with a probability of 0.1
    noise = np.random.rand(num_samples) < 0.1  # Boolean array where flips occur
    Y[noise] = 1 - Y[noise]  # Flip the labels

    return X, Y


"""Tests."""
# # Generate 100 samples
# num_samples = 100
# X, Y = generate_synthetic_data(num_samples)
#
# # Print some of the data
# for i in range(5):  # Print the first 5 samples
#     print(f"Sample {i + 1}: x={X[i]}, y={Y[i]}")


def generate_and_save_data(n_train=10000, n_test=1000):
    num_samples = n_train + n_test
    X, Y = generate_synthetic_data(num_samples)

    # Split the data into training and testing sets
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]

    # Saving them using numpy.
    np.savetxt("X_train.txt", X_train)
    np.savetxt("Y_train.txt", Y_train)
    np.savetxt("X_test.txt", X_test)
    np.savetxt("Y_test.txt", Y_test)
    print("Finished saving the training and testing data. ")


generate_and_save_data(5000, 500)