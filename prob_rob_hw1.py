import random 

def pi_estimate(num_points: int):
    # random.random() generates a random floating point number in the range [0, 1]
    # the probability that the point is within a unit circle and the square is the ratio of the areas
    points_in_circle = 0 
    for i in range(num_points):
        x = (random.random() * 2) - 1
        y = (random.random() * 2) - 1
        # scaled to be in the range [-1, 1] 
        # square has area = 2 * 2 = 4 
        if x**2 + y**2 <= 1:
            points_in_circle +=1 
    # ratio of points in circle vs generated should be = ratio of the areas = pi/4 
    pi_estimate = (points_in_circle/num_points)*4
    return pi_estimate

pi = pi_estimate(50000)
print(pi)





