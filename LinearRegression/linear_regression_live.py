#coding:utf-8
__author__ = 'Administrator'


import numpy as np

#
def compute_error_for_line_given_points(b, m, points):
    error_value = 0
    for it in ((y - (m*x + b))**2 for x,y in points):
        error_value += it
    return error_value / float(len(points))

def gradient_descent_runner(points, start_b, start_m, learning_rate, num_iterations):

    b = start_b
    m = start_m

    error_val = compute_error_for_line_given_points(b, m, points)
    #indicator = [b, m]
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
        tmp_error_val = compute_error_for_line_given_points(b, m, points)
        #if tmp_error_val < error_val:
            #error_val = tmp_error_val
           #indicator = [b, m]
    return [b, m]



def step_gradient(current_b, current_m, points, learning_rate):

    gradient_b = 0
    gradient_m = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        gradient_b += (current_m*x + current_b - y) * 2 / len(points)
        gradient_m += (current_m*x + current_b - y) * x * 2 / len(points)

    new_b = current_b - (learning_rate * gradient_b)
    new_m = current_m - (learning_rate * gradient_m)

    # print new_m, new_b

    return [new_b, new_m]


def Run():
    points = np.genfromtxt('data.csv', delimiter=',')

     ##
    learning_rate = 0.0001

    #y = mx + b
    init_b = 0
    init_m = 0

    num_iterations = 1000

    print "b: {0}, m: {1}, error: {2}".format(init_b, init_m, compute_error_for_line_given_points(init_b, init_m, points))
    result = gradient_descent_runner(points, init_b, init_m, learning_rate, num_iterations)
    print "b: {0}, m: {1}, error: {2}".format(result[0], result[1], compute_error_for_line_given_points(result[0], result[1], points))


if __name__ == '__main__':
    Run()
