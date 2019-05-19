import sys
import numpy as np


def normal_x_train(train_x):
    to_col = np.transpose(train_x)
    for i in range(len(to_col)):
        row = to_col[i]
        min = row.min()
        max = row.max()
        to_div = max - min
        for j in range(len(row)):
            if to_div != 0:
                to_col[i][j] = (row[j] - min) / to_div
            else:
                to_col[i][j] = 1 / len(train_x)
    train_x = np.transpose(to_col)
    return train_x


def to_matrix(file):
    get_file_value = open(file, 'r')
    x_train = np.array([[]])
    flag_first = 0
    line = (get_file_value.readline())

    while line:
        i = 0
        m = line[i]
        num = ""
        temp_x_train = np.array([])
        while ((m != '\n') & (i < len(line) - 1)):
            if (m == 'M'):
                temp_x_train = np.append(temp_x_train, 0.25)
            elif (m == 'F'):
                temp_x_train = np.append(temp_x_train, 0.5)
            elif (m == 'I'):
                temp_x_train = np.append(temp_x_train, 0.75)
            else:
                while ((m != ",") & (i < len(line) - 1)):
                    num += m
                    i += 1
                    m = line[i]
                if (num != ""):
                    temp_x_train = np.append(temp_x_train, float(num))
            if (i + 1 == len(line)):
                break
            i += 1
            m = line[i]
            num = ""
        temp1 = np.array([temp_x_train])
        if (flag_first != 0):
            x_train = np.concatenate((x_train, temp1))
        else:
            x_train = np.copy(temp1)
            flag_first = 1
        line = (get_file_value.readline())
    return x_train


def read_y_train(file):
    get_file_value = open(file, 'r').readlines()
    arr = []
    for line in get_file_value:
        arr.append(int(float(line.replace("\n", ""))))
    return arr


def predict(test_x, perceptron_w, svm_w, pa_w):
    # loop through the examples
    for x in test_x:
        # predictions for each algorithm
        perceptron_y_hat = np.argmax(np.dot(x, np.transpose(perceptron_w)))
        svm_y_hat =  np.argmax(np.dot(x, np.transpose(svm_w)))
        pa_y_hat =  np.argmax(np.dot(x, np.transpose(pa_w)))
        print("perceptron: " + str(perceptron_y_hat) +", svm: "+str(svm_y_hat)+", pa: " + str(pa_y_hat))


def perceptron(x_info, y_info):
    # preparation
    m = len(x_info)
    # initialise eta and weight vector
    eta = 0.1
    w = np.zeros((3, 8))
    # bad predictions counter
    bad_y_hat = 0

    epochs = 15
    for e in range(epochs):
        # choose a random example
        zip_info = list(zip(x_info, y_info))
        np.random.shuffle(zip_info)
        x_example, y_example = zip(*zip_info)
        for x, y in zip(x_example, y_example):
            # prediction
            y_hat = np.argmax(np.dot(w, x))
            # update w in case our predication is wrong
            if y_hat != y:
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
                bad_y_hat = bad_y_hat + 1
        err_avg = float((bad_y_hat) / m)
        bad_y_hat = 0
        eta = eta / (e + 100)
    return w


def pa(x_info, y_info):
    # preparation
    m = len(x_info)
    # initialise tau and weight vector
    tau = 0
    w = np.zeros((3, 8))
    # bad predictions counter
    bad_y_hat = 0
    eta = 0.001

    epochs = 10
    for e in range(epochs):
        # choose a random example
        zip_info = list(zip(x_info, y_info))
        np.random.shuffle(zip_info)
        x_example, y_example = zip(*zip_info)
        for x, y in zip(x_example, y_example):
            # prediction
            y_hat = np.argmax(np.dot(w, x))
            # calculate tau (by calculating hinge loss function and the norm of x powered by 2)
            hinge_loss = np.maximum(0, 1 - w[y, :] * x + w[y_hat, :] * x)
            x_norm = 2 * (np.linalg.norm(x) ** 2)
            if (x_norm != 0):
                tau = (hinge_loss / x_norm) * eta
            # update w in case our predication is wrong
            if y_hat != y:
                w[y, :] = w[y, :] + tau * x
                w[y_hat, :] = w[y_hat, :] - tau * x
                bad_y_hat = bad_y_hat + 1
        err_avg = float((bad_y_hat) / m)
        bad_y_hat = 0
        eta = eta / (e + 100)
    return w


def svm(x_info, y_info):
    # preparation
    m = len(x_info)
    # initialise eta, lamda and weight vector
    eta = 0.1
    lamda= 0.0001
    w = np.zeros((3, 8))
    # bad predictions counter
    bad_y_hat = 0

    epochs = 10
    for e in range(epochs):
        # choose a random example
        zip_info = list(zip(x_info, y_info))
        np.random.shuffle(zip_info)
        x_example, y_example = zip(*zip_info)
        for x, y in zip(x_example, y_example):
            # prediction
            y_hat = np.argmax(np.dot(w, x))
            # update w in case our predication is wrong
            if y_hat != y:
                w[y, :] = ((1-(eta*lamda)) * w[y, :]) + eta * x
                w[y_hat, :] = (1-eta*lamda) * w[y_hat, :] - eta * x
                w[(3 - (y+y_hat)), :] = (1-eta*lamda)*w[(3 - (y+y_hat)), :]
                bad_y_hat = bad_y_hat + 1
        err_avg = float((bad_y_hat) / m)
        bad_y_hat = 0
        eta = eta / (e + 100)
    return w


arg1 = sys.argv[1]
x_train = to_matrix(arg1)

arg2 = sys.argv[2]
y_train = read_y_train(arg2)

arg3 = sys.argv[3]
test_x = to_matrix(arg3)

perceptron_w = perceptron(x_train, y_train)
svm_w = svm(x_train, y_train)
pa_w = pa(x_train, y_train)

predict(test_x, perceptron_w, svm_w, pa_w)