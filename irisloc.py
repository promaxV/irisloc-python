import cv2
import os
import numpy as np
from time import time

from scipy.signal import find_peaks

def find_iris(source: cv2.Mat, center: tuple, 
              mode: str = 'iris',
              adjust_BRI_CONTR: bool = True, 
              adjust_BLUR: bool = True, 
              calc_time: bool = False):
    
    if mode=='iris':
        mode_func = np.max
    elif mode =='pupil':
        mode_func = np.min
    else:
        raise KeyError("Wrong key mode. Keys: 'iris', 'pupil'")
    
    if calc_time:
        start = time()
        
    # _, threshholded = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # inpainted = cv2.inpaint(source, threshholded, 2, cv2.INPAINT_TELEA)

    img = source.astype(np.float32)
    maxRad = 90.50966799187809
    
    polar = cv2.linearPolar(img, (center[1], center[0]), maxRad, cv2.WARP_FILL_OUTLIERS)
    
    polar = polar.astype(np.uint8)
    
    if adjust_BRI_CONTR:
        polar = cv2.convertScaleAbs(polar, alpha = 1.5, beta=-127)
    if adjust_BLUR:
        polar = cv2.blur(polar, (3,3))
    
    # angles = [0, 8, 16, 47, 55, 63, 67, 77, 113, 123]
    angles = [0, 8, 16, 24, 39, 47, 55, 63, 67, 77, 113, 123]
    peaks = []
    for ang in angles:
        deriv = np.gradient(polar[ang])
        maximas, _ = find_peaks(deriv)
        ind_max = np.argpartition(deriv[maximas], -2)[-2:]
        peaks.append(mode_func(maximas[ind_max]))
        
    normalized_peaks = []
    for peak in peaks:
        normalized_peaks.append(peak*maxRad/128)
    normalized_peaks = np.array(normalized_peaks)
    
    # normalized_peaks = np.vstack([normalized_peaks.max(axis=1), normalized_peaks.min(axis=1)]).T
        
    angles_rad = np.array(angles)*2*np.pi/128
    
    xs = np.full((len(angles),), center[0])+np.sin(angles_rad)*normalized_peaks
    ys = np.full((len(angles),), center[1])+np.cos(angles_rad)*normalized_peaks
    points_iris=np.round(np.vstack([ys, xs]).T).astype(int)
    
    med = np.median(normalized_peaks)
    std = normalized_peaks.std()
    points_iris = points_iris[np.abs(normalized_peaks-med)<std]
    
    points_iris = points_iris[points_iris[:,0]>0]
    points_iris = points_iris[points_iris[:,0]<127]
    
    # cv2.ellipse(image, cv2.fitEllipse(iris_points), color=[0,255,0])
    
    (xc_i, yc_i), r_i = cv2.minEnclosingCircle(points_iris)
    center_c = (int (xc_i), int(yc_i))
    r_i = int(r_i)
    
    if calc_time:
        end = time()
        print(end-start, "seconds")
    return (int (xc_i), int(yc_i)), r_i

def find_iris_by_path(path: str, output_path: str, center: tuple, mode: str):
    if mode not in ['iris', 'pupil', 'both']:
        raise KeyError("mode parameter should be one of 'iris', 'pupil' or 'both'")
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mode=="iris" or mode == "both":
        center_i, r_i = find_iris(image, center, mode='iris', adjust_BLUR=True, adjust_BRI_CONTR=True)
    if mode == "pupil" or mode == "both":
        center_p, r_p = find_iris(image, center, mode='pupil', adjust_BLUR=True, adjust_BRI_CONTR=True)
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if mode=="iris" or mode == "both":
        cv2.circle(image, center_i, r_i, [0,255,0])
    if mode == "pupil" or mode == "both":
        cv2.circle(image, center_p, r_p, [0,0,255])
    cv2.imwrite(output_path, image)
    
    match mode:
        case "iris":
            return (center_i, r_i)
        case "pupil":
            return (center_p, r_p)
        case "both":
            return [(center_i, r_i), (center_p, r_p)]

def find_irises_in_dir(dir_path: str, blur: bool, bri_contr: bool):
    try:
        os.makedirs("./output")
    except FileExistsError:
        pass
    start = time()
    for filename in os.scandir(dir_path):
        if filename.is_file():
            # print(filename.name)
            image = cv2.imread(filename.path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            center_point = (int(filename.name[:-4].split("_")[1]), int(filename.name[:-4].split("_")[2]))
            
            center_i, r_i = find_iris(image, center_point, mode='iris', adjust_BLUR=blur, adjust_BRI_CONTR=bri_contr)
            center_p, r_p = find_iris(image, center_point, mode='pupil', adjust_BLUR=blur, adjust_BRI_CONTR=bri_contr)
            
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.circle(image, center_i, r_i, [0,255,0])
            cv2.circle(image, center_p, r_p, [0,0,255])
            cv2.imwrite("./output/out_"+filename.name, image)
    end = time()
    print("Total", end-start, "seconds")
    
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global mouse_X, mouse_Y  
        mouse_X = x
        mouse_Y = y
        print(mouse_Y, mouse_X)
        
def get_center_on_image(path: str):
    img = cv2.imread(path)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (mouse_Y, mouse_X)
        
def mark_centers_in_dir(dir_path: str):
    for filename in os.scandir(dir_path):
        if filename.is_file() and filename.name[-4:]==".png" and filename.name[:3]!="eye":
            print(filename.path)
            img = cv2.imread(filename.path)
            cv2.imshow('image', img)
            cv2.setMouseCallback('image', click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            os.rename(filename.path, dir_path+"/eye_"+str(mouse_Y)+"_"+str(mouse_X)+".png")
            
def cut_and_rotate(dir_path: str, rotate: bool = True):
    for filename in os.scandir(dir_path):
        if filename.is_file() and filename.name[-4:]==".png":
            print(filename.path)
            img = cv2.imread(filename.path)
            if rotate:
                img = cv2.rotate(img, cv2.ROTATE_180)
            _, width = img.shape[:2]
            half_width = width // 2
            left_image = img[:, :half_width]
            right_image = img[:, half_width:]
            cv2.imwrite(dir_path+"/"+filename.name[:-4]+"left.png", left_image)
            cv2.imwrite(dir_path+"/"+filename.name[:-4]+"right.png", right_image)
            os.remove(filename.path)
            
def make_collage(dir_path: str, size: tuple):
    collage = np.array([])
    strip = np.array([])
    i = 0
    cur = 0
    for filename in os.scandir(dir_path):
        if filename.is_file() and filename.name[-4:]==".png" and i<size[0]*size[1]:
            print(filename.path)
            img = cv2.imread(filename.path)
            if i%size[0]==0:
                if i//size[0]==1:
                    collage = strip
                elif i//size[0]>1:
                    collage = np.vstack([collage, strip])
                strip = np.asarray(img)
            else:
                strip = np.hstack([strip, np.asarray(img)])
                
            i+=1
    cv2.imwrite("./collage.png", collage)