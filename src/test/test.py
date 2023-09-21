import cv2
import time
import numpy as np



def main():
    cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)

    img = cv2.imread("image_blocks.png")
    d_img = cv2.imread("depth_blocks.png", cv2.IMREAD_ANYDEPTH)
    mask = np.zeros_like(d_img, dtype=np.uint8)
    cv2.rectangle(mask, (275,120),(1100,720), 255, cv2.FILLED)
    cv2.rectangle(mask, (575,414),(723,720), 0, cv2.FILLED)
    cv2.rectangle(img, (275,120),(1100,720), (255, 0, 0), 2)
    cv2.rectangle(img, (575,414),(723,720), (255, 0, 0), 2)
    lower = -10
    upper = 500
    thresh = cv2.bitwise_and(cv2.inRange(d_img, lower, upper), mask)
    # depending on your version of OpenCV, the following line could be:
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,255), thickness=1)
    for contour in contours:
        color = retrieve_area_color(rgb_image, contour, colors)
        theta = cv2.minAreaRect(contour)[2]
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(img, color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=2)
        cv2.putText(img, str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=2)
        print(color, int(theta), cx, cy)
    #cv2.imshow("Threshold window", thresh)
    cv2.imshow("Image window", img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()




if __name__ == '__main__':
    main()