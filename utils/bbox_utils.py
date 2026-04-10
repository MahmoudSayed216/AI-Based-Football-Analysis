def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)



def get_bbox_width(bbox):
    x1, x2 = bbox[0], bbox[2]
    return int(x2-x1)