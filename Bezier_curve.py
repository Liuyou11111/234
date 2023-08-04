# from matplotlib import axis
# from plyfile  import PlyData
import numpy as np
import math
import matplotlib.pyplot as plt
# import pandas as pd
from PIL import Image
import cv2 as cv
import random
import os
# import pyrealsense2 as rs
from convert_depth_to_pc import Camera_intrinsics
import cv2
# B样条曲面拟合
import copy

Path = './img/label/'
img_dir = os.listdir(Path)


for img in img_dir:
    if img.endswith('1_color.png'):
        print(00)
        # 图片大小
        img_size_x = 848
        img_size_y = 480
        i = img[0:img.rfind('_color.png')]
        work_dir = './img5/'
        mask_path = work_dir + 'label/1_color_pseudo.png'.format(i)
        depth_img_path = work_dir + '1_depth.png'.format(i)
        intr_path = work_dir + 'camera-intr.npy'

        # 读取相机内参
        intr = np.load(intr_path, allow_pickle=True)
        camera = Camera_intrinsics(intr)
        print(camera.fx)

        mask = Image.open(mask_path)
        depth_img = Image.open(depth_img_path)
        mask_arr = np.array(mask)
        depth_img_arr = np.array(depth_img, dtype=np.float32)

        mask_r = mask_arr[:, :, 0]
        mask_r = np.uint8((mask_r - np.min(mask_r)) / (np.max(mask_r) - np.min(mask_r)))
        depth_img_arr = mask_r * depth_img_arr

        img=copy.deepcopy(mask_arr)
        for i in range(0,img_size_y):
            for j in range(0,img_size_x ):
                px = img[i, j]
                if px[2] ==255:
                    img[i,j]=(255,255,255)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        # 提取轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        x_min = img_size_x + 100
        x_max = 0
        y_min = img_size_y + 100
        y_max = 0
        depth_img_arr_max=0

        for i in range(0, img_size_y - 1):
            for j in range(0, img_size_x - 1):
                px = mask_r[i, j]
                px2 = mask_r[i, j + 1]
                px3 = mask_r[i + 1, j]
                if px == 0 and px2 != 0:
                    x_min = min(x_min, j + 1)
                if px != 0 and px2 == 0:
                    x_max = max(x_max, j)
                if px == 0 and px3 != 0:
                    y_min = min(y_min, i + 1)
                if px != 0 and px3 == 0:
                    y_max = max(y_max, i)

        print("x_min:{}".format(x_min))
        print("x_max:{}".format(x_max))
        print("y_min:{}".format(y_min))
        print("y_max:{}".format(y_max))


        need_fix_depth_img_arr=np.zeros((y_max  - y_min,x_max- x_min),np.float32)
        for i in range(x_min,x_max):
            for j in range(y_min,y_max):
                need_fix_depth_img_arr[j-y_min][i-x_min]=depth_img_arr[j][i]
        leaf_point_zhongxin = need_fix_depth_img_arr[(y_max - y_min) // 2][(x_max - x_min) // 2]

        Z_yuzhi=30
        need_fix_depth_point = []
        for i in range(0, img_size_y):
            for j in range(0, img_size_x):
                px = mask_r[i, j]
                if i>=y_min and i<y_max and j>=x_min and j<x_max:
                    if px != 0 and need_fix_depth_img_arr[i-y_min][j-x_min]>leaf_point_zhongxin-Z_yuzhi and need_fix_depth_img_arr[i-y_min][j-x_min]<leaf_point_zhongxin+Z_yuzhi:
                        need_fix_depth_point.append([j-x_min, i-y_min, need_fix_depth_img_arr[i-y_min][j-x_min]])
                    else:
                        need_fix_depth_img_arr[i - y_min][j - x_min] = 0
                        need_fix_depth_point.append([j - x_min, i - y_min, need_fix_depth_img_arr[i - y_min][j - x_min]])



        a = np.array(need_fix_depth_point)
        np.savetxt("needfiximg1.txt", np.c_[a])

        new_depth_arr = np.zeros((img_size_y, img_size_x))
        for i in range(x_max-x_min):
            for j in range(y_max-y_min):
                new_depth_arr[j+y_min][i+x_min]=need_fix_depth_img_arr[j,i]

        for j in range(0, img_size_y):
            for i in range(0, img_size_x):
                depth_img_arr_max = max(depth_img_arr_max, new_depth_arr[j][i])

        need_fix_depth_img_arr_gray = np.zeros((img_size_y, img_size_x), np.float32)
        for j in range(0, img_size_y):
            for i in range(0, img_size_x):
                need_fix_depth_img_arr_gray[j][i] =new_depth_arr[j][i]/depth_img_arr_max*255

        yanmo=[[ [0,0,0] for col in range(img_size_x)] for row in range(img_size_y)]
        yanmo=np.array(yanmo)






        for i in range(0,img_size_x):
            for j in range(0,img_size_y):
                if need_fix_depth_img_arr_gray[j][i]!=0:
                    yanmo[j][i] = [255,255,255]
                else:
                    yanmo[j][i] = [0,0,0]
                need_fix_depth_point.append([i, j, depth_img_arr[j][i]])

        yanmo=yanmo.astype(np.uint8)
        gray=cv2.cvtColor(yanmo,cv2.COLOR_BGR2GRAY)
        _,yan_mask=cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)
        #bilateral=cv2.bilateralFilter(need_fix_depth_img_arr_gray,15,75,75)
        dst = cv2.inpaint(need_fix_depth_img_arr_gray,yan_mask, 10,cv2.INPAINT_NS)
        fix_depth_img = np.zeros((img_size_y, img_size_x))
        fix_depth_point =[]
        for i in range(0, img_size_y):
            for j in range(0, img_size_x):
                fix_depth_img[i,j]=depth_img_arr[i,j]
        for i in range(img_size_x):
            for j in range(img_size_y):
                fix_depth_img[j, i]=float(dst[j, i])/255*depth_img_arr_max
                fix_depth_point.append([i, j, fix_depth_img[j, i]])

        a = np.array(fix_depth_point)
        np.savetxt("fiximg1.txt", np.c_[a])






        # 获取像素值
        mask_list = []
        for i in range(0, img_size_y):
            for j in range(0, img_size_x):
                px = mask_r[i, j]
                if px != 0:
                    mask_list.append([i, j])
        depth_img_arr_2=[]
        for point_x_y in mask_list:
            depth_img_arr[point_x_y[0],point_x_y[1]]=fix_depth_img[point_x_y[0],point_x_y[1]]

        leaf_contours = []

        control_point = []
        trims_2D_l_point_orign = []
        trims_2D_r_point_orign = []

        x_distance = x_max -x_min
        x_extend=0.1*x_distance
        x_min=int(x_min-x_extend)
        x_max=int(x_max+x_extend)
        x_distance=x_max -x_min
        y_distance = y_max - y_min
        y_extend = 0.1 * y_distance
        y_min = int(y_min - y_extend)
        y_max = int(y_max + y_extend)
        y_distance = y_max - y_min
        size_u_tongji=10
        size_v_tongji = 10

        x= np.linspace(x_min,x_max,size_u_tongji)
        y = np.linspace(y_min, y_max, size_v_tongji)
        X,Y=np.meshgrid(x,y)

        print("x={}".format(x))
        print("y={}".format(y))
        print("X={}".format(X))
        print("Y={}".format(Y))

        for i in range(20):
            for j in range(20):
                yy=int(Y[i][j])
                xx=int(X[i][j])
                a = fix_depth_img[yy, xx]
                control_point.append([xx, yy, a])


        # 格子间隔

        for i in range(len(contours[0])):
            leaf_contours.append([(contours[0][i][0][1] - y_min) / (y_max - y_min), (contours[0][i][0][0] - x_min) / x_distance])
        leaf_contours.append( [(contours[0][0][0][1] - y_min) / (y_max - y_min), (contours[0][0][0][0] - x_min) /x_distance])



        #2D轮廓点
        # a = np.array(trims_2D_l_point)
        # np.savetxt("444.txt", np.c_[a])
        #
        # a = np.array(trims_2D_r_point)
        # np.savetxt("555.txt", np.c_[a])

        #用来查看输入控制点数据
        np.savetxt("shuru1.txt", np.c_[control_point])

        print("control point.size:{}".format(len(control_point)))
        print(control_point)

        from geomdl import fitting
        from geomdl.visualization import VisMPL
        from geomdl import exchange
        from matplotlib import cm
        from geomdl import tessellate
        from geomdl import BSpline
        from geomdl import knotvector
        from geomdl.visualization import VisMPL as vis


        # Fix file path
        os.chdir(os.path.dirname(os.path.realpath(__file__)))


        # Create a planar BSpline surface (surface 1)



        surf_bottom = control_point
        size_u = size_u_tongji
        size_v = size_v_tongji
        degree_u = 3
        degree_v = 3

        print("size_u:{}".format(size_u))
        print("size_v:{}".format(size_v))
        row = size_u-5
        column = size_v-15

        surf_bottom = fitting.approximate_surface(surf_bottom, size_u, size_v, degree_u, degree_v , ctrlpts_size_u=row, ctrlpts_size_v=column)
        exchange.export_txt(surf_bottom, "surf1.txt")
        from geomdl import abstract
        #vis_config = VisMPL.VisConfig(ctrlpts=False)


        surf_bottom.sample_size = 50
        surf_bottom.tessellator = tessellate.TrimTessellate()
        vis_conf = vis.VisConfig(ctrlpts=False, legend=False,trims=True)
        surf_bottom.vis = vis.VisSurface(vis_conf)

        #归一化后裁剪曲线
        trim1 = BSpline.Curve()
        trim1.degree = 1
        trim1.ctrlpts = leaf_contours
        trim1.knotvector = knotvector.generate(trim1.degree, trim1.ctrlpts_size)
        trim1.delta = 0.1
        trim1.opt = ['reversed', 1]
        surf_bottom.trims=[trim1]


        # Fix trimming curves
        #trimming.fix_trim_curves(surf_bottom)
        #print(surf_bottom.tessellator)
        #surf_bottom.render(colormap=cm.cool)
        exchange.export_obj(surf_bottom, "trim1.obj")
        # surf_bottom.vis = VisMPL.VisSurface(trims=True, ctrlpts=False)
        print(surf_bottom.data)
        trim1.vis=VisMPL.VisCurve2D(ctrlpts=False)
        trim1.render(colormap=cm.cool)
        #
