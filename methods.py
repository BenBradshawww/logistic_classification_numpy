import numpy as np


# Returns column vector of probabilities
def probability_prediction(X: list, w: list):

    return 1 / (1 + np.exp(-1 * X.dot(w)))


def predict(X, w):

    probs = probability_prediction(X, w)

    return class_prediction(probs)


def class_prediction(probs) -> list:

    n = len(probs)
    classes = np.zeros(shape=(n, 1))

    for i in range(0, n):

        if probs[i] >= 0.5:
            classes[i, 0] = 1

    return classes


# Returns the float loss
def cross_entropy_loss_function(X, w, y) -> float:

    probs = probability_prediction(X, w)

    ones = np.ones(shape=(len(y), 1))

    loss = -np.dot(y.T, np.log(probs)) - np.dot((ones - y).T, np.log(ones - probs))

    return loss[0][0]


# Returns a columns vector of the gradient
def grad_cross_entropy_loss_function(X: list, probs: list, y: list):

    n, d = X.shape

    temp = probs - y

    sums = np.zeros(shape=(d, 1))

    for i in range(n):
        sums = np.add(sums, temp[i] * np.reshape(X[i, :], newshape=(d, 1)))

    return sums / n


def gradient_descent(X, y, w=None, num_iter=100, step_size=0.01, stopping_criteria=1e-5):

    d = np.shape(X)[1]  # Dimension
    w = np.transpose([np.random.normal(0, 2, d)])  # Column Vector of Weights
    w_array = w
    loss_list = [cross_entropy_loss_function(X, w, y)]

    for i in range(num_iter):

        probs = probability_prediction(X, w)

        w = w - step_size * grad_cross_entropy_loss_function(X, probs, y)

        w_array = np.append(w_array, w, axis=1)

        loss_list.append(cross_entropy_loss_function(X, w, y))

        if np.linalg.norm(w_array[:, -1] - w_array[:, -2]) < stopping_criteria:
            break

    return w, loss_list


def lipschitz_generator(X, y, m, seed=42):

    np.random.seed(seed)

    d = np.shape(X)[1]
    max_l = float("-Inf")

    for i in range(0, m):

        w1 = np.transpose([np.random.normal(0, 0.01, d)])
        w2 = np.transpose([np.random.normal(0, 0.01, d)])

        probs1 = probability_prediction(X, w1)
        probs2 = probability_prediction(X, w2)

        grad1 = grad_cross_entropy_loss_function(X, probs1, y)
        grad2 = grad_cross_entropy_loss_function(X, probs2, y)

        l = np.linalg.norm(grad2 - grad1) / np.linalg.norm(w2 - w1)

        max_l = max(max_l, l)

    return max_l


# Returns a columns vector of the gradient
def sgd_cross_entropy_loss_function(X: list, probs: list, y: list):

    if X.ndim == 1:
        d = len(X)
        n = 1
    else:
        n, d = X.shape

    temp = probs - y

    sums = np.zeros(shape=(d, 1))

    if n == 1:
        return temp * np.reshape(X, newshape=(d, 1))
    else:
        for i in range(n):
            sums = np.add(sums, temp[i] * np.reshape(X[i, :], newshape=(d, 1)))

        return sums / n


def sgd(X, y, w=None, num_iter=100, step_size=0.01, stopping_criteria=1e-5, batch_size=None, seed=42):

    np.random.seed(seed)

    n, d = np.shape(X)  # Dimension
    w = np.transpose([np.random.normal(0, 2, d)])  # Column Vector of Weights
    w_array = w
    loss_list = [cross_entropy_loss_function(X, w, y)]

    for i in range(num_iter):

        if batch_size:
            indexs = np.random.randint(0, n, batch_size)
        else:
            indexs = np.random.randint(0, n, 1)

        probs = probability_prediction(X[indexs], w)

        w = w - step_size * sgd_cross_entropy_loss_function(X[indexs], probs, y[indexs])

        w_array = np.append(w_array, w, axis=1)

        loss_list.append(cross_entropy_loss_function(X, w, y))

        if np.linalg.norm(w_array[:, -1] - w_array[:, -2]) < stopping_criteria:
            break

    return w, loss_list


def sgd_momentum(
    X, y, w=None, num_iter=100, step_size=0.01, stopping_criteria=1e-5, batch_size=None, momentum_term=None, seed=42
):

    np.random.seed(seed)

    n, d = np.shape(X)  # Dimension
    w = np.transpose([np.random.normal(0, 2, d)])  # Column Vector of Weights
    w_array = w
    loss_list = [cross_entropy_loss_function(X, w, y)]
    grad = np.zeros(shape=(d, 1))

    for i in range(num_iter):

        if batch_size:
            indexs = np.random.randint(0, n, batch_size)
        else:
            indexs = np.random.randint(0, n, 1)

        probs = probability_prediction(X[indexs], w)

        grad = momentum_term * grad + (1 - momentum_term) * sgd_cross_entropy_loss_function(X[indexs], probs, y[indexs])

        w = w - step_size * grad

        w_array = np.append(w_array, w, axis=1)

        loss_list.append(cross_entropy_loss_function(X, w, y))

    return w_array, loss_list


def nestrov_accelerated_gradient(
    X, y, w=None, num_iter=100, step_size=0.01, stopping_criteria=1e-5, batch_size=None, momentum_term=None, seed=42
):
    np.random.seed(seed)

    n, d = np.shape(X)  # Dimension
    w = np.transpose([np.random.normal(0, 2, d)])  # Column Vector of Weights
    w_array = w
    loss_list = [cross_entropy_loss_function(X, w, y)]
    grad = np.zeros(shape=(d, 1))

    for i in range(num_iter):

        if batch_size:
            indexs = np.random.randint(0, n, batch_size)
        else:
            indexs = np.random.randint(0, n, 1)

        probs = probability_prediction(X[indexs], w - step_size * grad)

        grad = momentum_term * grad + (1 - momentum_term) * sgd_cross_entropy_loss_function(X[indexs], probs, y[indexs])

        w = w - step_size * grad

        w_array = np.append(w_array, w, axis=1)

        loss_list.append(cross_entropy_loss_function(X, w, y))

    return w_array, loss_list


def AdaGrad(X, y, w=None, num_iter=100, step_size=0.01, stopping_criteria=1e-5, batch_size=None, epsilon=None, seed=42):

    np.random.seed(seed)

    n, d = np.shape(X)  # Dimension
    w = np.transpose([np.random.normal(0, 2, d)])  # Column Vector of Weights
    w_array = w  # Array to store our weights
    loss_list = [cross_entropy_loss_function(X, w, y)]  # Loss List
    grad = np.zeros(shape=(d, 1))
    G = np.zeros(shape=(d, d))
    v = np.zeros(shape=(d, 1))
    m = np.zeros(shape=(d, 1))

    # Epsilon
    if not epsilon:
        epsilon = 1e-8

    epsilon_list = [epsilon] * d
    epsilon_diag = np.diag(epsilon_list)

    for i in range(1, num_iter + 1):

        if batch_size:
            indexs = np.random.randint(0, n, batch_size)
        else:
            indexs = np.random.randint(0, n, 1)

        # Calculating probabilities
        probs = probability_prediction(X[indexs], w - step_size * grad)

        # Calculating the Stochastic Gradient
        grad = sgd_cross_entropy_loss_function(X[indexs], probs, y[indexs])

        # Element-wise gradient square.
        grad_squared = grad**2

        # Updating gradient matrix
        G += np.diag(grad_squared.T[0])

        # Updating the weights
        w = w - step_size * np.linalg.inv(np.sqrt(G + epsilon_diag)).dot(grad)

        # Storing Weights
        w_array = np.append(w_array, w, axis=1)

        # Appending new Loss
        loss_list.append(cross_entropy_loss_function(X, w, y))

    return w_array, loss_list


def RMSProp(
    X,
    y,
    w=None,
    momentum=0.9,
    num_iter=100,
    step_size=0.01,
    stopping_criteria=1e-5,
    batch_size=None,
    epsilon=None,
    seed=42,
):

    np.random.seed(seed)

    n, d = np.shape(X)  # Dimension
    w = np.transpose([np.random.normal(0, 2, d)])  # Column Vector of Weights
    w_array = w  # Array to store our weights
    loss_list = [cross_entropy_loss_function(X, w, y)]  # Loss List
    grad = np.zeros(shape=(d, 1))
    s = 0

    # Epsilon
    if not epsilon:
        epsilon = 1e-8

    for i in range(1, num_iter + 1):

        # Batch Size
        if batch_size:
            indexs = np.random.randint(0, n, batch_size)
        else:
            indexs = np.random.randint(0, n, 1)

        # Calculating probabilities
        probs = probability_prediction(X[indexs], w)

        # Calculating the Stochastic Gradient
        grad = sgd_cross_entropy_loss_function(X[indexs], probs, y[indexs])

        # Element-wise gradient square.
        grad_squared = grad**2

        # Momentum Equation
        s = momentum * s + (1 - momentum) * grad_squared

        # Updating the weights
        w = w - step_size * (1 / (epsilon + np.sqrt(s))) * grad

        # Storing Weights
        w_array = np.append(w_array, w, axis=1)

        # Appending new Loss
        loss_list.append(cross_entropy_loss_function(X, w, y))

    return w_array, loss_list


def ADAM(
    X,
    y,
    w=None,
    num_iter=100,
    step_size=0.01,
    stopping_criteria=1e-5,
    batch_size=None,
    b1=None,
    b2=None,
    epsilon=None,
    seed=42,
):

    np.random.seed(seed)

    n, d = np.shape(X)  # Dimension
    w = np.transpose([np.random.normal(0, 2, d)])  # Column Vector of Weights
    w_array = w  # Array to store our weights
    loss_list = [cross_entropy_loss_function(X, w, y)]  # Loss List
    grad = np.zeros(shape=(d, 1))
    v = np.zeros(shape=(d, 1))
    m = np.zeros(shape=(d, 1))

    # Beta 1
    if not b1:
        b1 = 0.9
    if not b2:
        b2 = 0.999
    if not epsilon:
        epsilon = 1e-8

    for i in range(1, num_iter + 1):

        if batch_size:
            indexs = np.random.randint(0, n, batch_size)
        else:
            indexs = np.random.randint(0, n, 1)

        # Calculating probabilities
        probs = probability_prediction(X[indexs], w)

        # Calculating the Stochastic Gradient
        grad = sgd_cross_entropy_loss_function(X[indexs], probs, y[indexs])

        # Updating biased first moment estimate
        m = b1 * m + (1 - b1) * grad

        # Updating biased seconds raw moment estimate.
        v = b2 * v + (1 - b2) * (grad**2)

        # Computing bias-corrected first moment
        m_corrected = m / (1 - (b1**i))

        # Computing bias-corrected first moment
        v_corrected = v / (1 - (b2**i))

        # Updating the weights
        w = w - step_size * m_corrected / (np.sqrt(v_corrected) + epsilon)

        # Storing Weights
        w_array = np.append(w_array, w, axis=1)

        # Appending new Loss
        loss_list.append(cross_entropy_loss_function(X, w, y))

    return w_array, loss_list


def AdaDelta(
    X,
    y,
    w=None,
    momentum=0.9,
    num_iter=100,
    step_size=0.01,
    stopping_criteria=1e-5,
    batch_size=None,
    epsilon=None,
    seed=42,
):

    np.random.seed(seed)

    n, d = np.shape(X)  # Dimension
    w = np.transpose([np.random.normal(0, 2, d)])  # Column Vector of Weights
    w_array = w  # Array to store our weights
    loss_list = [cross_entropy_loss_function(X, w, y)]  # Loss List
    grad = np.zeros(shape=(d, 1))
    s = 0
    v = 0

    # Epsilon
    if not epsilon:
        epsilon = 1e-8

    for i in range(1, num_iter + 1):

        # Batch Size
        if batch_size:
            indexs = np.random.randint(0, n, batch_size)
        else:
            indexs = np.random.randint(0, n, 1)

        # Calculating probabilities
        probs = probability_prediction(X[indexs], w - step_size * grad)

        # Calculating the Stochastic Gradient
        grad = sgd_cross_entropy_loss_function(X[indexs], probs, y[indexs])

        # Element-wise gradient square.
        grad_squared = grad.T.dot(grad)[0]

        # Momentum Equation
        s = momentum * s + (1 - momentum) * grad_squared

        # Updating the RMS_g
        RMS_g = np.sqrt(s + epsilon)

        # 2nd Momentum Equation
        v = momentum * v + (1 - momentum) * RMS_g.T.dot(RMS_g)

        # Updating
        RMS_w = np.sqrt(v + epsilon)

        # Updating weights
        w = w - (RMS_w / RMS_g) * grad

        # Storing Weights
        w_array = np.append(w_array, w, axis=1)

        # Appending new Loss
        loss_list.append(cross_entropy_loss_function(X, w, y))

    return w_array, loss_list
