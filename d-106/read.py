import sys
import xlrd
#from nn_bp import * 
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors

#read all xls files and write data into a csv file in order
def combine (name_list, seat):
    all_data = []
    for filename in name_list:
        ele = []
        loc = filename.split ("_")
        ele.append (float (loc[0]))
        order = seat[int (loc[0])][loc[1]]
        ele.append (float (order))
        xls_file = xlrd.open_workbook(filename)
        data_set = xls_file.sheets ()[0]
        for i in range (1, data_set.nrows):
            line = data_set.row_values (i)
            ele.append (line[1])
            ele.append (float (line[3]))
        all_data.append (ele)
    
    #print (all_data)
    with open ("dat.csv", "w") as f:
        for rec in all_data:
            for i in range (len (rec) - 1):
                f.write (str (rec[i]) + ",")
            f.write (str (rec[-1]) + "\n")

#read the csv file 
def read_file (filename):
    all = []
    with open (filename, "r") as f:
        while True:
            line = f.readline ()
            if not line:
                break
            line = line.replace ("\n", "")
            line = line.split (",")
            all.append (line)

    return (all)

#from the train data set find all unique mac, n defs the minimum times that each mac occurs
def get_allMac (all, min_times, min_stren):
    all_mac = {}
    mac_avg = {}
    for line in all:
        for i in range (2, len (line), 2):
            if not all_mac.__contains__ (line[i]):
                all_mac[line[i]] = 1
                mac_avg[line[i]] = float (line[i + 1])
            else:
                all_mac[line[i]] = all_mac[line[i]] + 1
                mac_avg[line[i]] = mac_avg[line[i]] + float (line[i + 1])

    dict_keys = (all_mac.copy ()).keys ()
    for key in dict_keys:
        mac_avg[key] = mac_avg[key] / all_mac[key]
        if all_mac[key] < min_times:
            all_mac.pop (key)
        elif mac_avg[key] < min_stren:
            all_mac.pop (key)
    print (len (all_mac), "unique mac")
    
    return (all_mac)

#transform the data based on the mac list 
def convert_dat (raw_dat, all_mac, NA):
    location = []
    rss = []
    mac_RSSI = []
    #act_num = []
    for line in raw_dat:
        ele = {}
        for i in range (2, len (line), 2):
            ele[line[i]] = float (line[i+1])
        mac_RSSI.append (ele)
        location.append ([float (line[0]), float (line[1])])
    
    for i in range (len (raw_dat)):
        ele = []
        tmp = 0
        for key in all_mac.keys ():
            if mac_RSSI[i].__contains__ (key):
                ele.append (mac_RSSI[i][key])
                tmp = tmp + 1
            else:
                ele.append (NA)
        rss.append (ele)
        #act_num.append (tmp)
    
    #print (act_num)
    return rss, location

#read seat chart
def seat_chart ():
    seat = {}
    row = 0
    with open ("seat.csv") as f:
        while True:
            row = row + 1
            tmp = {}
            line = f.readline ()
            if not line:
                break
            line = line.replace ("\n", "")
            line = line.split (",")
            for i in range (len (line)):
                tmp[line[i]] = i+1
            seat[row] = tmp
    
    return seat   

def norm (data_set):
    c_num = len (data_set[0])
    for i in range (c_num):
        for j in range (len (data_set)):
            data_set[j][i] = (data_set[j][i] + 50) / (30)
    
    return data_set

def dimen_tran (loc, flag):
    loc_pro = []
    if flag:
        for rec in loc:
            loc_pro.append ((rec[0] - 1) * 25 + rec[1])
    else:
        for rec in loc:
            loc_pro.append ([int (rec / 25 + 1), rec % 25])

    return loc_pro 

def loc_narrow (loc_li, mult_x, mult_y):
    for i in range (len (loc_li)):
        loc_li[i][0] = int (loc_li[i][0] / mult_x)
        loc_li[i][1] = int (loc_li[i][1] / mult_y)
    
    return loc_li

def accuracy(predictions, labels):
    sqr_sum = 0.0
    for i in range (len (predictions)):
        tmp = 0.0
        for j in range (len (predictions[0])):
            tmp = tmp + (predictions[i][j] - labels[i][j]) ** 2
        sqr_sum = sqr_sum + tmp ** 0.5
    
    return (sqr_sum / len (predictions))

def main ():
    #seat = seat_chart ()
    #combine (sys.argv[1:], seat)
    train_raw = read_file ("all.csv")
    test_raw = read_file ("test.csv")
    all_mac = get_allMac (train_raw, 10, -100)
    train_rss, train_loc = convert_dat (train_raw, all_mac, -100)
    test_rss, test_loc = convert_dat (test_raw, all_mac, -100)

    #"""
    clf = MLPRegressor (hidden_layer_sizes = (200, 200), max_iter = 10000)
    acc_li = []
    for i in range (5):
        for j in range (20):
            clf.fit (train_rss, train_loc)
            predict = clf.predict(test_rss)
            """
            print ("ex", i, ":")
            for j in range (len (predict)):
                print (test_loc[j], "->", predict[j])
            print ()
            """
            acc_li.append (accuracy (predict, test_loc))
        print (sum (acc_li) / len (acc_li))
    #"""
    
    """
    knn_cls = neighbors.KNeighborsRegressor(4, weights='uniform', metric='euclidean')
    predict = knn_cls.fit(train_rss, train_loc).predict(test_rss)
    for i in range (len (predict)):
        print ("predict:", predict[i], "\t", "fact:", test_loc[i])
    """
    
    """
    clf = MLPClassifier (hidden_layer_sizes = (200, 200), max_iter = 10000)
    clf.fit (train_rss, dimen_tran (train_loc, 1))
    predict = clf.predict(test_rss)
    predict = dimen_tran (predict, 0)
    for i in range (len (predict)):
        print (test_loc[i], "->", predict[i])
    """

    """
    clf_x = MLPClassifier (hidden_layer_sizes = (100, 100), max_iter = 10000)
    clf_x.fit (train_rss, [rec[0] for rec in train_loc])
    predict_x = clf_x.predict(test_rss)
    clf_y = MLPClassifier (hidden_layer_sizes = (100, 100), max_iter = 10000)
    clf_y.fit (train_rss, [rec[1] for rec in train_loc])
    predict_y = clf_y.predict(test_rss)
    predict = []
    for i in range (len (predict_x)):
        ele = [predict_x[i], predict_y[i]]
        predict.append(ele)
        print (test_loc[i], "->", predict[i])
    """

    """
    train_rss = norm (train_rss)
    test_rss = norm (test_rss)

    nn = NN (61, 100, 100, 2)
    nn.train (train_rss, train_loc, 100)
    nn.test (test_rss, test_loc)
    """

    """
    for rec in test_loc:
        num = train_loc.count (rec)
        if num != 0:
            key1 = train_loc.index (rec)
            rss1 = train_rss [key1]
            key2 = test_loc.index (rec)
            rss2 = test_rss [key2]
            rss3 = train_rss[0]
            for i in range (len (rss1)):
                print (rss1[i], "\t", rss2[i], "\t\t", rss3[i])
            print ()
    """

main ()