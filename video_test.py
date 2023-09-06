import cv2
import argparse
def test_camera(source=1):
    cap = cv2.VideoCapture(source)
    
    while True:
        ret, frame = cap.read()
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h,w = rotated.shape[:2]
        print(h,w)
        cv2.line(rotated, (0, int(h/2)), (w,int(h/2)), (255,0,0), thickness=3)
        cv2.line(rotated, (0, int(h/2)-300), (w,int(h/2)-300), (255,0,0), thickness=3)
        cv2.imshow('frame', rotated)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test camera')
    parser.add_argument('--source', type=int, default=1, help='camera source')
    args = parser.parse_args()
    test_camera(args.source)