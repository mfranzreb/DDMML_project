import numpy as np


def evaluate_hypothesis(X, theta):
    # evaluate the linear hypothesis: h = X * theta
    h_theta = np.zeros((X.shape[0], 1))  # initialization

    # evaluate hypothesis
    # >>> enter your code here
    h_theta = np.dot(X, theta)
    return h_theta


def compute_cost(h, y):
    # compute cost for linear regression
    m = y.shape[0]  # number of observations
    J = 0  # initialization

    # >>> enter your code here
    J = (1 / m) * np.sum((h - y) ** 2)
    return J


def solve_normaleqn(X, y):
    # initialize theta
    theta = np.zeros((X.shape[1], 1))

    # solve the normal equation
    # >>> enter your code here
    theta = np.dot(X.T, X)
    try:
        theta = np.linalg.inv(theta)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(theta)
    theta = np.dot(np.dot(theta, X.T), y)
    return theta


def solve_gradientdescent(X, y, theta, alpha, itercount):
    # initialize values
    m = X.shape[0]
    J_log = np.zeros((itercount, 1))
    costgrad = np.zeros(theta.shape)

    # initialize hypothesis
    h = evaluate_hypothesis(X, theta)

    for i in range(itercount):
        # compute cost gradient
        # >>> enter your code here
        costgrad = (2 / m) * np.dot(X.T, (h - y))

        # update theta
        # >>> enter your code here
        theta -= alpha * costgrad

        # evaluate hypothesis
        h = evaluate_hypothesis(X, theta)

        # evaluate cost J and save every iteration
        J_log[i, 0] = compute_cost(h, y)
    return theta, J_log


def cost_fun(theta, X, y):
    # compute cost and cost gradient for linear hypothesis
    m = X.shape[0]  # number of observations
    J = 0.0  # initialization
    # >>> enter your code here
    J = (1 / m) * np.sum((evaluate_hypothesis(X, theta) - y) ** 2)
    return J


def cost_grad(theta, X, y):
    # compute cost gradient for linear hypothesis
    m = X.shape[0]  # number of observations
    costgrad = np.zeros(theta.shape)
    # >>> enter your code here
    costgrad = (2 / m) * np.dot(X.T, (evaluate_hypothesis(X, theta) - y))
    return costgrad.flatten()


def create_featurepoly(X, p):
    # creates a feature matrix of polynomials [X, X.^2, ... X.^p]
    # input: X = m x n matrix
    # output: out = m x (n*p)
    m = X.shape[0]  # initialization
    n = X.shape[1]  # initialization
    poly_features = np.zeros((m, n * p))  # initialization

    # >>> enter your code here
    for i in range(p):
        poly_features[:, i * n : (i + 1) * n] = X ** (i + 1)
    return poly_features


def solve_normaleqn_reg(X, y, lambda_reg):
    # solve Tikhonov regularized normal equation
    theta = np.zeros((X.shape[1], 1))  # initialization

    # >>> enter your code here
    ident = np.eye(X.shape[1])
    ident[0, 0] = 0
    theta = np.linalg.pinv(X.T @ X + lambda_reg * ident) @ X.T @ y
    return theta
