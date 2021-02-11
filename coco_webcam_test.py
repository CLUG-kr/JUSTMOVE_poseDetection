import cv2

protoFile = ".\\coco\\pose_deploy_linevec.prototxt"
weightsFile = ".\\coco\\pose_iter_440000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

image_height = 368
image_width = 368
threshold = 0.1

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FPS, 0.3)
cap.set(3,800)
cap.set(4,800)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)


    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()

    out = out[:, :19, :, :]

    assert(len(BODY_PARTS_COCO) == out.shape[1])

    points = []

    for i in range(len(BODY_PARTS_COCO)):
        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        x = (frameWidth * point[0]) / out.shape[3]
        x = int(x)
        y = (frameHeight * point[1]) / out.shape[2]
        y = int(y)
        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((x, y))

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)
            points.append(None)

    for pair in POSE_PAIRS_COCO:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        assert(part_a in BODY_PARTS_COCO)
        assert(part_b in BODY_PARTS_COCO)

        if points[part_a] and points[part_b]:
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)

    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("webcam pose estimation with COCO model", frame)