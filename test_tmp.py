import numpy as np

n = 10
pts_y_rect = np.zeros((n,1))
y = pts_y_rect[0,0]
print("y:", y)
pts_y_rect[0] = [1]
y = pts_y_rect[0,0]
print("y:", y)
print("type:", type(pts_y_rect[0]))