import cv2
import time
import numpy as np

def deHaze(img_with_haze, r_Dark_Patch, r_GUIDED,  eps, w=0.95, t0=0.1):
    print('看 r_Dark_Patch: ',  r_Dark_Patch)
    deHazed_img = np.zeros(img_with_haze.shape)
    map_T, A = get_map_T_and_A(img_with_haze, r_Dark_Patch, r_GUIDED, eps, w)
    #  we restrict the transmission map_T(x) to a lower bound t0, \
    #  which means that a small certain amount of haze are preserved in very dense haze regions
    map_T_with_bound = np.maximum(map_T, t0)
    for k in range(3):
        #  J = (I – A) / map_T  + A
        deHazed_img[:,:,k] = (img_with_haze[:,:,k] - A[k])/ map_T_with_bound + A[k]
    deHazed_img = np.clip(deHazed_img, 0, 1)
    return map_T, deHazed_img,A

def get_map_T_and_A(img_with_haze, r_Dark_Patch, r_GUIDED, eps, w):                 # 输入rgb图像，值范围[0,1]
    #  J = (I – A) / map_T  + A

    #--------------------  min  option 1：-----------------------------------------------------------------
    min_among_channel = np.min(img_with_haze, 2)
    Dark_Channel = MinFilter_gray(min_among_channel, r_Dark_Patch)
    #--------------------------end

    #--------------------  min  option 2：-----------------------------------------------------------------
    #  min_among_patch = np.zeros(img_with_haze.shape)
    #  for k in range(3):
        #  min_among_patch[:,:,k] = MinFilter_gray(img_with_haze[:,:,k] ,r_Dark_Patch)
    #  Dark_Channel = np.min(min_among_patch,2)
    #--------------------------end

    cv2.imwrite('dark.jpg',Dark_Channel*255 )
    #  keep a very small amount of haze \
    #  for the distant objects by introducing a constant parameter w (0<ω≤1)

    gray_img_with_haze =cv2.cvtColor(img_with_haze, cv2.COLOR_BGR2GRAY)


    bins = 255*2
    #  返回直方图的纵坐标，横坐标
    y_ht,x_ht= np.histogram(Dark_Channel, bins)
    #  累加后的柱子高度/　ｍａｐ的总像素个数
    y_cumsum = np.cumsum(y_ht) / float(Dark_Channel.size)
    #  记录暗通道图中灰度最大的前0.1%的像素所在的位置num_max
    #  for循环得到num_max
    for num_max in range(bins - 1, 0, -1):
        if y_cumsum[num_max] <= 0.999:
            break
    cv2.imwrite ('mask_for_大气.jpg', 255*np.array(Dark_Channel>= x_ht[num_max],np.uint8))

    img_with_haze_2 = np.zeros(img_with_haze.shape)
    A = []
    for c in range(3):
        #大气亮度 = 带雾图像的灰度图　在这些位置中的         最大值
        A.append( img_with_haze[:,:,c][ Dark_Channel>= x_ht[num_max] ].max() )
        img_with_haze_2[:,:,c] = img_with_haze[:,:,c]/A[c]

    min_among_channel_2 = np.min(img_with_haze, 2)
    Dark_Channel_2 = MinFilter_gray(min_among_channel_2, r_Dark_Patch)

    map_T = 1 - w*Dark_Channel_2
    map_T = guided_filter_for_map_T(gray_img_with_haze, map_T, r_GUIDED, eps)

    return map_T, A

def MinFilter_gray(src, r_Dark_Patch):
    # 腐蚀，对于每个像素，提取方形kernel覆盖下的像素最小值
    kernel = np.ones((2*r_Dark_Patch + 1, 2*r_Dark_Patch + 1) )
    return cv2.erode(src, kernel )


def guided_filter_for_map_T(guide_pic, input_pic, r_GUIDED, eps):
    height, width = guide_pic.shape

    #  r_GUIDED: 导向滤波的窗口半径。合适的才能保证使得公式中的a和b是常数
    #  eps: smoothing

    #  cv2.boxfilter(img, -1, (3, 3), normalize=True) 表示进行方框滤波，
    #  当normalize=True时，与均值滤波结果相同

    # 论文公式的I和p分别是guide_pic和input_pic
    #(r_GUIDED, r_GUIDED) 是一个  mean filter with a window radius r
    m_I = cv2.boxFilter(guide_pic, -1, (r_GUIDED, r_GUIDED))
    m_p = cv2.boxFilter(input_pic, -1, (r_GUIDED, r_GUIDED))
    m_Ip = cv2.boxFilter(guide_pic * input_pic, -1, (r_GUIDED, r_GUIDED))
    m_II = cv2.boxFilter(guide_pic * guide_pic, -1, (r_GUIDED, r_GUIDED))
    #  乘积的期望-期望的乘积
    cov_Ip = m_Ip - m_I * m_p
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    print('看 eps: ',  eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r_GUIDED, r_GUIDED))
    m_b = cv2.boxFilter(b, -1, (r_GUIDED, r_GUIDED))
    return m_a * guide_pic + m_b


#  if __name__ == '__main__':

#  for i in ['1']:
for i in ['2']:
#  for i in ['4']:
    imageName = "win{}.jpg".format(i)
    #  imageName = "win{}.jpg".format(i)

    #  my_img = cv2.imread('images/' + imageName)
    my_img = cv2.imread('images/' + imageName).astype(np.float32)
    #  opencv只支持float32的图像显示和操作，float64是numpy的默认数据类型，opencv中不支持。
    #  指定np.float32即可
    #  my_img = cv2.resize(my_img,(500,500),interpolation = cv2.INTER_AREA)

    h,w,_ = my_img.shape

    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1200,600)

    #  trackbar可以用callback函数，但遇到bug，貌似用getTrackbarPos的人更多
    def nothing(x):
            pass

    cv2.createTrackbar('r_Dark_Patch_bar','img',5,200, nothing)

    cv2.createTrackbar('smoothing','img',1,40,nothing)
    cv2.createTrackbar('r_GUIDED_bar +1','img',1,30,nothing)

    cv2.createTrackbar('1 - 0.01*far_dehaze_strength','img',0,6,nothing)
    cv2.createTrackbar('bright_add','img',1,60,nothing)


    while(1):
        #  这样导致滚动条用不了
        #  keyb = int(input())
        #  _eps = 0.01*(keyb)

        k =cv2.waitKey(200) & 0xFF
        #  if k!=255:
        _radius  = cv2.getTrackbarPos('r_Dark_Patch_bar','img')
        #  _radius = (int(min([h,w])*_radius/300) - 1)//2
        #  _radius = 7
        _eps = 0.001*cv2.getTrackbarPos('smoothing','img')
        _w = 1- 0.05*cv2.getTrackbarPos('1 - 0.01*far_dehaze_strength','img')
        _b = cv2.getTrackbarPos('bright_add','img')/255
        _radius_guided = cv2.getTrackbarPos('r_GUIDED_bar +1','img')+1

        map_T, deHazed_img,A =  deHaze(my_img/255.0, r_Dark_Patch=_radius ,r_GUIDED=_radius_guided, eps=_eps, w=_w,t0=0.1)
        #  不行
        #  map_T = cv2.cvtColor(map_T,cv2.COLOR_GRAY2RGB)

        #  imshow的图片必须是归一化的！
        #  cv2.imshow('iqmg', map_T)
        #  print('看A : ', A)
        cv2.imshow('img',my_img)
        #  imgs = np.hstack([deHazed_img,map_T])
        #  cv2.imshow('img',map_T)
        if k ==ord('q'):
            break

    cv2.destroyAllWindows()

