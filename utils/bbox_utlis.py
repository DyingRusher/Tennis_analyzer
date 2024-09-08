def find_center_bbox(bbox):
    x1,y1,x2,y2 = bbox
    c_x = int((x1+x2)/2)
    c_y = int((y1+y2)/2)

    return (c_x,c_y)

def measure_dis(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_dis(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_center_of_bbox(point):
    return (int((point[0]+point[2])/2),int((point[1]+point[3])/2))