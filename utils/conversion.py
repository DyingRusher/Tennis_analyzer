def pixal_to_m(pixel_d,refrence_height_in_p,refrence_height_in_m):
    return (pixel_d * refrence_height_in_m)/refrence_height_in_p

def meter_to_p(pixel_d,refrence_height_in_m,refrence_height_in_p):
    return pixel_d * (refrence_height_in_p/refrence_height_in_m)

def get_foot_position(bbox):

    x1,y1,x2,y2 = bbox
    return (int((x1+ x2)/2),y2)

def get_closest_kp_index(pt,kp,kp_index):

    closest_dis = float('inf')
    key_points_ind = kp_index[0]

    for  kp_i in kp_index:
        # print("Kp index in conversion",kp_i*2 + 1,len(kp))
        kp_p = kp[kp_i*2],kp[kp_i*2 + 1]
        distance = abs(pt[1] - kp_p[1])

        if distance < closest_dis:
            closest_dis = distance
            key_points_ind = kp_i

    return key_points_ind