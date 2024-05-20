import os
import datetime
import pandas
import pandas as pd
import numpy as np
import pywt
cit1=5
cit2=50

def get_5_feature(path,final_path):
    files = os.listdir(path)
    for i in range(len(files)):
        data = pd.read_csv(path + "/" + files[i])
        try:
            lon_list = data['经度']
            lat_list = data['纬度']
            speed_list = data['速度']
            angular_list = data['方向']
            time_list = data['时间']
            try:
                tag_list = data['标签']
            except:
                tag_list = data['标记']
        except:
            lon_list=data['longitude']
            lat_list=data['latitude']
            speed_list = data['speed']
            angular_list = data['dir']
            time_list = data['time']
            tag_list=data['tags']
        speed_diff = [0 for x in range(len(speed_list))]
        time_diff = [0 for x in range(len(speed_list))]
        acclec = [0 for x in range(len(speed_list))]
        angular_speed_diff = [0 for x in range(len(speed_list))]
        angular_acclec = [0 for x in range(len(speed_list))]
        angular_diff = [0 for x in range(len(speed_list))]
        angular_speed = [0 for x in range(len(speed_list))]
        for j in range(len(list(speed_list))):
            if j == 0:
                speed_diff[j] = 0
                time_diff[j] = 0
                acclec[j] = 0
            else:
                speed_diff[j] = speed_list[j] - speed_list[j - 1]
                try:
                    d1 = datetime.datetime.strptime(str(time_list[j - 1]), '%Y/%m/%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[j]), '%Y/%m/%d %H:%M:%S')
                except:
                    d1 = datetime.datetime.strptime(str(time_list[j - 1]), '%Y-%m-%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[j]), '%Y-%m-%d %H:%M:%S')
                time_diff[j] = (d2 - d1).seconds
                acclec[j] = float(speed_diff[j]) / time_diff[j]
        for temp in range(len(list(speed_list))):
            if temp == 0:
                angular_speed[temp] = 0
            else:
                if speed_list[temp] == 0:
                    angular_speed[temp] = 0
                else:
                    angular_speed[temp] = angular_list[temp] / time_diff[temp]
        for k in range(len(list(speed_list))):
            if k == 0:
                angular_speed_diff[k] = 0
                time_diff[k] = 0
                angular_acclec[k] = 0
            else:
                angular_speed_diff[k] = angular_speed[k] - angular_speed[k - 1]
                try:
                    d1 = datetime.datetime.strptime(str(time_list[k - 1]), '%Y/%m/%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[k]), '%Y/%m/%d %H:%M:%S')
                except:
                    d1 = datetime.datetime.strptime(str(time_list[k - 1]), '%Y-%m-%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[k]), '%Y-%m-%d %H:%M:%S')
                time_diff[k] = (d2 - d1).seconds
                angular_acclec[k] = angular_speed_diff[k] / time_diff[k]
        for l in range(len(list(speed_list))):
            if l == 0:
                angular_diff[l] = 0
            else:
                angular_diff[l] = angular_list[l] - angular_list[l - 1]
        columns = ['lon','lat','speed', 'acceleration', 'angular_speed', 'angular_acceleration', 'angle_diff','tag']
        df = pd.DataFrame(columns=columns)
        df["lon"] = lon_list
        df["lat"] = lat_list
        df["speed"] = speed_list
        df['acceleration'] = acclec
        df['angular_speed'] = angular_speed
        df['angular_acceleration'] = angular_acclec
        df['angle_diff'] = angular_diff
        df['tag'] = tag_list
        x = pd.DataFrame(df)
        x.to_csv(final_path+"/" + files[i], index=False)

def get_feature(path,final_path):
    files = os.listdir(path)
    for i in range(len(files)):
        data = pd.read_csv(path + "/" + files[i])
        try:
            lon_list = data['经度']
            lat_list = data['纬度']
            speed_list = data['速度']
            angular_list = data['方向']
            time_list = data['时间']
            try:
                tag_list = data['标签']
            except:
                tag_list = data['标记']
        except:
            lon_list=data['longitude']
            lat_list=data['latitude']
            speed_list = data['speed']
            angular_list = data['dir']
            time_list = data['time']
            tag_list=data['tags']

        speed_diff = [0 for x in range(len(speed_list))]
        time_diff = [0 for x in range(len(speed_list))]
        acclec = [0 for x in range(len(speed_list))]
        angular_speed_diff = [0 for x in range(len(speed_list))]
        angular_acclec = [0 for x in range(len(speed_list))]
        angular_diff = [0 for x in range(len(speed_list))]
        angular_speed = [0 for x in range(len(speed_list))]
        jerk = [0 for x in range(len(speed_list))]
        jerk_diff = [0 for x in range(len(speed_list))]
        lon_diff = [0 for x in range(len(speed_list))]
        lat_diff = [0 for x in range(len(speed_list))]

        #加速度
        for j in range(len(list(speed_list))):
            if j == 0:
                speed_diff[j] = 0
                time_diff[j] = 0
                acclec[j] = 0
            else:
                speed_diff[j] = speed_list[j] - speed_list[j - 1]
                try:
                    d1 = datetime.datetime.strptime(str(time_list[j - 1]), '%Y/%m/%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[j]), '%Y/%m/%d %H:%M:%S')
                except:
                    d1 = datetime.datetime.strptime(str(time_list[j - 1]), '%Y-%m-%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[j]), '%Y-%m-%d %H:%M:%S')
                time_diff[j] = (d2 - d1).seconds
                acclec[j] = float(speed_diff[j]) / time_diff[j]

        #经纬度差
        for j in range(len(list(speed_list))):
            if j == 0:
                lon_diff[j] = 0
                lat_diff[j] = 0
            else:
                lon_diff[j] = lon_list[j] - lon_list[j - 1]
                lat_diff[j] = lat_list[j] - lat_list[j - 1]

        #jerk
        for j in range(len(list(speed_list))):
            if j == 0:
                jerk[j] = 0
            else:
                jerk_diff[j] = acclec[j] - acclec[j - 1]
                jerk[j] = jerk_diff[j] / time_diff[j]


        #角速度
        for temp in range(len(list(speed_list))):
            if temp == 0:
                angular_speed[temp] = 0
            else:
                if speed_list[temp] == 0:
                    angular_speed[temp] = 0
                else:
                    angular_speed[temp] = angular_list[temp] / time_diff[temp]

        #角加速度
        for k in range(len(list(speed_list))):
            if k == 0:
                angular_speed_diff[k] = 0
                time_diff[k] = 0
                angular_acclec[k] = 0
            else:
                angular_speed_diff[k] = angular_speed[k] - angular_speed[k - 1]
                try:
                    d1 = datetime.datetime.strptime(str(time_list[k - 1]), '%Y/%m/%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[k]), '%Y/%m/%d %H:%M:%S')
                except:
                    d1 = datetime.datetime.strptime(str(time_list[k - 1]), '%Y-%m-%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(time_list[k]), '%Y-%m-%d %H:%M:%S')
                time_diff[k] = (d2 - d1).seconds
                angular_acclec[k] = angular_speed_diff[k] / time_diff[k]

        #角度差
        for l in range(len(list(speed_list))):
            if l == 0:
                angular_diff[l] = 0
            else:
                angular_diff[l] = angular_list[l] - angular_list[l - 1]

        columns = ['lon','lat','speed', 'acceleration','angular_speed','angular_acceleration',
                   'jerk', 'angle_diff', 'lon_diff', 'alt_diff', 'tag']
        df = pd.DataFrame(columns=columns)
        df["lon"] = lon_list
        df["lat"] = lat_list
        df["speed"] = speed_list
        df['acceleration'] = acclec
        df['angular_speed'] = angular_speed
        df['angular_acceleration'] = angular_acclec
        df['jerk'] = jerk
        df['angle_diff'] = angular_diff
        df['lon_diff'] = lon_diff
        df['lat_diff'] = lat_diff
        df['tag'] = tag_list
        x = pd.DataFrame(df)
        x.to_csv(final_path+"/" + files[i], index=False)

def db(lst):
    ya=[[0 for x in range(len(lst))] for i in range(4)]
    wavename=['db1','db2','db3','db4']
    for i in range(4):
        ca, cd = pywt.dwt(lst[:len(lst)], wavename[i])
        if(len(lst) % 2 == 0):
            ya[i] = pywt.idwt(ca, None, wavename[i])
        else:
            ya[i] = pywt.idwt(ca, None, wavename[i])
            ya[i] = ya[i][0:len(lst)]
    return ya[0],ya[1],ya[2],ya[3]


def sym(lst):
    ya = [[0 for x in range(len(lst))] for i in range(4)]
    wavename=['sym2','sym3','sym4','sym5']
    for i in range(4):
        ca, cd = pywt.dwt(lst[:len(lst)], wavename[i])
        if (len(lst) % 2 == 0):
            ya[i] = pywt.idwt(ca, None, wavename[i])
        else:
            ya[i] = pywt.idwt(ca, None, wavename[i])
            ya[i] = ya[i][0:len(lst)]
    return ya[0], ya[1], ya[2], ya[3]


def get_median(data):
   data = sorted(data)
   size = len(data)
   if size % 2 == 0: # 判断列表长度为偶数
    #median = (data[size//2]+data[size//2-1])/2
    median =  data[size//2]
    data[0] = median
   if size % 2 == 1: # 判断列表长度为奇数
    median = data[(size-1)//2]
    data[0] = median
   return data[0]

def med(num,lst):
    med=[0 for x in range(len(lst))]
    med[0]=lst[0]
    for i in range(1,len(lst)):
        if(i<=num):
            med[i]=get_median(lst[:i+1])
        else:
            med[i] = get_median(lst[i-num:i+1])
    return med

def SD(num,lst):
    SD=[0 for x in range(len(lst))]
    SD[0]=np.std([lst[0]])
    for i in range(1,len(lst)):
        if(i<=num):
            SD[i]=np.std(lst[:i+1])
        else:
            SD[i] = np.std(lst[i-num:i+1])
    return SD
# 计算特征
if __name__=="__main__":
    path = r"data/data_file"
    if not os.path.exists(path):
        raise Exception("文件夹不存在")
    if not os.path.exists(path):
        os.mkdir(path)
    middle_path = path+"_cal"
    if not os.path.exists(middle_path):
        os.mkdir(middle_path)
    final_path=path+"_5"
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    end_path=path+"_25"
    if not os.path.exists(end_path):
        os.mkdir(end_path)
    staticremove = 0
    alllen = 0
    allfile = os.listdir(path)
    #去除重复
    for q in range(len(allfile)):
        data1 = pd.read_csv(path +'/'+ allfile[q], header=0)
        try:
            data2 = data1.loc[:, ['time', 'longitude', 'latitude', 'speed', 'dir', 'tags']]
            x = data2['longitude']
            y = data2['latitude']
            speed = data2['speed']
        except:
            try:
                data2 = data1.loc[:, ['时间', '经度', '纬度', '速度', '方向', '标签']]
            except:
                data2 = data1.loc[:, ['时间', '经度', '纬度', '速度', '方向', '标记']]
            x = data2['经度']
            y = data2['纬度']
            speed = data2['速度']
        alllen = alllen + len(speed)
        # 清理掉经纬度速度重复的
        waitdelete = []
        allpoint = []
        for j in range(len(speed)):
            point = []
            point.append([x[j], y[j], speed[j]])
            if point in allpoint:
                waitdelete.append(j)
            else:
                allpoint.append(point)
        data2 = data2.drop(waitdelete)
        data2 = data2.reset_index()
        # 清理掉时间重复的
        try:
            time = data2['time']
        except:
            time = data2['时间']
        timedelete = []
        all_time_point = []
        for j in range(len(time)):
            if time[j] in all_time_point:
                timedelete.append(j)
            else:
                all_time_point.append(time[j])
        data2 = data2.drop(timedelete)
        data2 = data2.reset_index()
        del data2['index']
        staticremove = staticremove + len(waitdelete)
        data = pd.DataFrame(data2)
        data.to_csv(middle_path+"/"+allfile[q],index=False)
    get_5_feature(middle_path, final_path)

    files=os.listdir(final_path)
    for i in files:
        data=pd.read_csv(final_path+"/"+i)
        tag_list=data["tag"]
        lon_list = data['lon']
        lat_list = data['lat']

        angular_speed = data['angular_speed']
        angular_acclec = data['angular_acceleration']

        speed_list=data['speed']
        speed_med_5=med(cit1,speed_list)
        speed_med_20=med(cit2,speed_list)
        speed_SD_5=SD(cit1,speed_list)
        speed_SD_20 = SD(cit2, speed_list)
        speed_db1, speed_db2, speed_db3, speed_db4 = db(speed_list)
        speed_sym1, speed_sym2, speed_sym3, speed_sym4 = sym(speed_list)

        acc_list = data['acceleration']
        acc_med_5 = med(cit1, acc_list)
        acc_med_20 = med(cit2, acc_list)
        acc_SD_5 = SD(cit1, acc_list)
        acc_SD_20 = SD(cit2, acc_list)
        acc_db1, acc_db2, acc_db3, acc_db4 = db(acc_list)
        acc_sym1, acc_sym2, acc_sym3, acc_sym4 = sym(acc_list)

        ang_speed_list = data['angular_speed']
        ang_speed_med_5 = med(cit1, ang_speed_list)
        ang_speed_med_20 = med(cit2, ang_speed_list)
        ang_speed_SD_5 = SD(cit1, ang_speed_list)
        ang_speed_SD_20 = SD(cit2, ang_speed_list)

        ang_acc_list = data['angular_acceleration']
        ang_acc_med_5 = med(cit1, ang_acc_list)
        ang_acc_med_20 = med(cit2, ang_acc_list)
        ang_acc_SD_5 = SD(cit1, ang_acc_list)
        ang_acc_SD_20 = SD(cit2, ang_acc_list)

        ang_diff_list = data['angle_diff']
        ang_diff_med_5 = med(cit1, ang_diff_list)
        ang_diff_med_20 = med(cit2, ang_diff_list)
        ang_diff_SD_5 = SD(cit1, ang_diff_list)
        ang_diff_SD_20 = SD(cit2, ang_diff_list)
        ang_diff_db1, ang_diff_db2, ang_diff_db3, ang_diff_db4 = db(ang_diff_list)
        ang_diff_sym1, ang_diff_sym2, ang_diff_sym3, ang_diff_sym4 = sym(ang_diff_list)

        columns = ['speed', 'speed_med_5', 'speed_med_20','speed_SD_5','speed_SD_20',
                   'acceleration','acceleration_med_5', 'acceleration_med_20','acceleration_SD_5','acceleration_SD_20',
                   'angular_speed','angular_speed_med_5', 'angular_speed_med_20','angular_speed_SD_5','angular_speed_SD_20',
                   'angular_acceleration','angular_acceleration_med_5','angular_acceleration_med_20','angular_acceleration_SD_5','angular_acceleration_SD_20',
                   'angle_diff','angle_diff_med_5','angle_diff_med_20','angle_diff_SD_5','angle_diff_SD_20',
                   'tag','lon','lat']


        df = pd.DataFrame(columns=columns)
        df["speed"] = speed_list
        df["speed_med_5"] = speed_med_5
        df["speed_med_20"] = speed_med_20
        df["speed_SD_5"] = speed_SD_5
        df["speed_SD_20"] = speed_SD_20

        df['acceleration'] = acc_list
        df['acceleration_med_5'] = acc_med_5
        df['acceleration_med_20'] = acc_med_20
        df['acceleration_SD_5'] = acc_SD_5
        df['acceleration_SD_20'] = acc_SD_20

        df['angular_speed'] = ang_speed_list
        df['angular_speed_med_5'] = ang_speed_med_5
        df['angular_speed_med_20'] = ang_speed_med_20
        df['angular_speed_SD_5'] = ang_speed_SD_5
        df['angular_speed_SD_20'] = ang_speed_SD_20

        df['angular_acceleration'] = ang_acc_list
        df['angular_acceleration_med_5'] = ang_acc_med_5
        df['angular_acceleration_med_20'] = ang_acc_med_20
        df['angular_acceleration_SD_5'] = ang_acc_SD_5
        df['angular_acceleration_SD_20'] = ang_acc_SD_20

        df['angle_diff'] = ang_diff_list
        df['angle_diff_med_5'] = ang_diff_med_5
        df['angle_diff_med_20'] = ang_diff_med_20
        df['angle_diff_SD_5'] = ang_diff_SD_5
        df['angle_diff_SD_20'] = ang_diff_SD_20


        df['tag']=tag_list
        df['lon']=lon_list
        df['lat']=lat_list

        x = pd.DataFrame(df)
        x.to_csv(end_path+"/"+ i, index=False)

