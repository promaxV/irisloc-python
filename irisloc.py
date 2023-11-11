import cv2
import os
import imageio
import numpy as np
import pandas as pd
from time import time

from scipy.signal import find_peaks
from scipy import optimize

def find_iris(source: cv2.Mat, center: tuple, 
              mode: str = 'iris',
              angles = list([*range(0, 16), *range(48, 80), *range(112, 128)]),
              filtering = "median",
              sigmaK = 1,
              fitting_method = "least_squares",
              adjust_contrast: bool = True,
              blur_mode: str = "gauss", 
              calc_time: bool = False):
    
    def auto_contrast(image):
        min_v = image.min()
        max_v = image.max()
        a = 255/(max_v-min_v)
        b = 255*(1-max_v/(max_v-min_v))
        return np.clip(a*image+b, 0, 255)
    
    if mode=='iris':
        mode_func = np.max
    elif mode =='pupil':
        mode_func = np.min
    else:
        raise KeyError("Wrong key mode. Keys: 'iris', 'pupil'")
    
    if calc_time:
        start = time()
        
    if len(source.shape) == 3:
        img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img = source.astype(np.float32)
        
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    
    if adjust_contrast:
        img = auto_contrast(img)
    
    maxRad = 90.50966799187809
    
    polar = cv2.linearPolar(img, (center[1], center[0]), maxRad, cv2.WARP_FILL_OUTLIERS)
    polar = polar.astype(np.uint8)

    if blur_mode == "gauss":
        polar = cv2.GaussianBlur(polar, (3, 3), 0)
    elif blur_mode == "average":
        polar = cv2.blur(polar, (3, 3))
    
    peaks = []
    deriv_sum = np.zeros(128, )
    for ang in angles:
        deriv = np.gradient(polar[ang])
        deriv_sum += deriv
        maximas, _ = find_peaks(deriv)
        if len(maximas) > 2:
            ind_max = np.argpartition(deriv[maximas], -2)[-2:]
            peaks.append(mode_func(maximas[ind_max]))
        else:
            peaks.append(0)
        
    sum_maximas, _ = find_peaks(deriv_sum)
    ind_max_sum = np.argpartition(deriv_sum[sum_maximas], -2)[-2:]
    sum_peak = mode_func(sum_maximas[ind_max_sum])*maxRad/128

    normalized_peaks = []
    for peak in peaks:
        normalized_peaks.append(peak*maxRad/128)
    normalized_peaks = np.array(normalized_peaks)
    
    angles_rad = np.array(angles)*2*np.pi/128
    
    med = np.median(normalized_peaks[normalized_peaks != 0])
    std = normalized_peaks.std()
    
    xs = np.full((len(angles),), center[0])+np.sin(angles_rad)*normalized_peaks
    ys = np.full((len(angles),), center[1])+np.cos(angles_rad)*normalized_peaks
    points_iris = np.vstack([ys, xs]).T
    
    if filtering == "median":
        points_iris = points_iris[np.abs(normalized_peaks - med) < sigmaK*std]
    elif filtering == "sum_peak":
        points_iris = points_iris[np.abs(normalized_peaks - sum_peak) < sigmaK*std]
    
    # points_iris = points_iris[normalized_peaks != 0]
    # points_iris = points_iris[normalized_peaks != 0]
    
    points_iris = points_iris[points_iris[:, 0] > 0]
    points_iris = points_iris[points_iris[:, 1] > 0]
    points_iris = points_iris[points_iris[:, 0] < 127]
    points_iris = points_iris[points_iris[:, 1] < 127]
    
    # points_iris = points_iris[np.sqrt((points_iris[:, 0] - center[0])**2 + (points_iris[:, 1] - center[1])**2) > 10]
   
    if len(points_iris)<3:
        return (0, 0), 0, []
   
    if fitting_method == "least_squares":
        x = points_iris[:, 0]
        y = points_iris[:, 1]
        
        def calc_R(xc, yc):
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = x.mean(), y.mean()
        center, _ = optimize.leastsq(f_2, center_estimate)

        xc, yc = center
        Ri = calc_R(*center)
        r = Ri.mean()
        
    elif fitting_method == "enclosing_circle":
        (xc, yc), r = cv2.minEnclosingCircle(points_iris.astype(np.float32))
        
     # cv2.ellipse(image, cv2.fitEllipse(iris_points), color=[0,255,0])
    
    center_c = (round(xc), round(yc))
    r = round(r)
    
    if calc_time:
        end = time()
        print(end-start, "seconds")
        
    return center_c, r, points_iris

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
    
def test_with_images_and_csv(images_folder: str, csv_path: str):
    df = pd.read_csv(csv_path)
    df.drop(columns=df.columns[-1], inplace=True)
    df = df[["Time","Lx", "Ly", "Ls", "Rx", "Ry", "Rs", "Pitch"]]
    df.loc[:,["Lx", "Ly", "Ls", "Rx", "Ry", "Rs"]] = df.loc[:,["Lx", "Ly", "Ls", "Rx", "Ry", "Rs"]].interpolate()
    df = df[df["Pitch"].notna()].reset_index().drop(columns=["index", "Pitch"])
    
    os.makedirs(images_folder.rstrip("/")+"_output", exist_ok=True)
    i=0
    for filename in sorted(os.listdir(images_folder), key=lambda x: int(x[x.find("-")+1:x.rfind(".")])):
            # print(filename.name)
            image = cv2.imread(images_folder+filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            left_image = image[:, :image.shape[1]//2]
            right_image = image[:, image.shape[1]//2:]
            
            left_center = (df.loc[i, "Ly"], df.loc[i, "Lx"])
            left_radius = np.sqrt(df.loc[i, "Ls"]/np.pi)
            right_center = (df.loc[i, "Ry"], df.loc[i, "Rx"]-128)
            right_radius = np.sqrt(df.loc[i, "Rs"]/np.pi)
            
            left_iris_c, left_iris_r = find_iris(left_image, left_center, mode='iris', adjust_BLUR=True, adjust_BRI_CONTR=False)
            left_pupil_c, left_pupil_r = find_iris(left_image, left_center, mode='pupil', adjust_BLUR=True, adjust_BRI_CONTR=False)
            
            right_iris_c, right_iris_r = find_iris(right_image, right_center, mode='iris', adjust_BLUR=True, adjust_BRI_CONTR=False)
            right_pupil_c, right_pupil_r = find_iris(right_image, right_center, mode='pupil', adjust_BLUR=True, adjust_BRI_CONTR=False)
            
            
            
            left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
            right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
            
            cv2.circle(left_image, (round(left_center[1]), round(left_center[0])), round(left_radius), [255,0,0])
            cv2.circle(left_image, np.round(left_pupil_c).astype(int), round(left_pupil_r), [0,0,255])
            cv2.circle(left_image, np.round(left_iris_c).astype(int), round(left_iris_r), [0,255,0])
            
            cv2.circle(right_image, (round(right_center[1]), round(right_center[0])), round(right_radius), [255,0,0])
            cv2.circle(right_image, np.round(right_iris_c).astype(int), round(right_iris_r), [0,255,0])
            cv2.circle(right_image, np.round(right_pupil_c).astype(int), round(right_pupil_r), [0,0,255])
            
            output = np.concatenate([left_image, right_image], axis=1)
            
            cv2.imwrite(images_folder.rstrip("/")+"_output/out_"+filename, output)
            
            i+=1
            
def gif_from_images(images_folder, output_path):
    with imageio.get_writer(output_path, mode='I') as writer:
        for filename in sorted(os.listdir(images_folder), key=lambda x: int(x[x.find("-")+1:x.rfind(".")])):
            image = imageio.imread(images_folder+filename)
            writer.append_data(image)