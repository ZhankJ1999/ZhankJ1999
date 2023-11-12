import cv2 as cv
from PIL import Image, ImageFilter
import PIL.ImageOps
import numpy as np
from math import gcd, sqrt, ceil, floor
# from huaxue1215 import zhuaqu
from matplotlib import pyplot as plt
#定义区
path = "D:/121.png"
Prep = cv.imread(path, 0)
X, Y = Prep.shape[0:2]
Near = [[0] * 10] * 10 #邻接表
Points = [] #每个点的坐标，dot格式
Ts = 0
class bingchaji:
    parent = {}
    def __init__(self, n):
        for i in range(n):
            self.parent[i] = i
    def find(self, k):
        if k == self.parent[k]:
            return k
        return self.find(self.parent[k])
    def equal(self, a, b):
        return self.find(a) == self.find(b)
    def merge(self, U, D):
        if self.equal(U, D):
            return
        self.parent[self.find(U)] = self.find(D)

    def count(self, n):
        cnt = 0
        Q = []
        for i in range(n):
            if self.parent[i] == i:
                cq = 0
                for j in range(n):
                    if self.find(j) == i:
                        cq += 1
                Q.append((i, cq))
        return Q
    def Prt(self, n):
        for i in range(n):
            print(self.parent[i], end=" ")
        print("\n")
class Dot:
    Ele = "C"
    Num = 0
    Zx = 0
    Zy = 0
    def __init__(self, Cx, Cy):
        Ts += 1
        self.Num = Ts
        self.Zx = Cx
        self.Zy = Cy
    def bond(self, ele, linker):
        if Near[self.Num][ele.Num] != 0:
            return False
        Near[self.Num][ele.Num] = Near[ele.Num][self.Num] = linker
        return True

def check(Px, Py):
    for i in Points:
        D = sqrt((i.Zx - Px) ** 2 + (i.Zy - Py) ** 2)
        if D < 3.0:
            return False
    return True


Prep = cv.resize(Prep, (Y * 3, X * 3))
Prep = cv.Canny(Prep, 100, 100)
N = cv.imread(path, 0)
N = cv.Canny(N, 100, 200)
P = Image.fromarray(cv.cvtColor(Prep, cv.COLOR_BGR2RGB)) #一开始的描边图（黑底）
P = PIL.ImageOps.invert(P)
#P为每次处理之后的图像
F = Image.new("RGB", (P.width, P.height), (255, 255, 255)) #最终处理的单线图（白底）

R = 10

def check(x, Logic = 1): #logic为指定识别白底黑线和黑底白线的逻辑变量，0为白线，1为黑线
    if Logic == 0:
        return ((x[0] > 128) & (x[1] > 128) & (x[2] > 128)) == True
    else:
        return ((x[0] <= 128) & (x[1] <= 128) & (x[2] <= 128)) == True
#找白点
def Findnear(Po, Log = 1):#查找格子周围的点，R为半径 在后续处理白底黑字时用1
    Ret = []
    for Y in range(Po[1]-R, Po[1]+R + 1):
        if check(P.getpixel((Po[0] - R, Y)), Log):
            Ret.append((Po[0] - R, Y))
        if check(P.getpixel((Po[0] + R, Y)), Log):
            Ret.append((Po[0] + R, Y))
    for X in range(Po[0] - R, Po[0] + R + 1):
        if check(P.getpixel((X, Po[1] - R)), Log):
            Ret.append((X, Po[1] - R))
        if check(P.getpixel((X, Po[1] + R)), Log):
            Ret.append((X, Po[1] + R))
    return Ret
def Text(N):
    if N == "Cl":
        return -1
    elif N == "Br":
        return -1
    elif N == "I":
        return -1
    elif N == "CH3":
        return -1
    elif N == "CI":
        return -1
    elif N == "OH":
        return -1
    elif N == "O":
        return -2
    elif N == "NH":
        return -2
    elif N == "NH2":
        return -1
    else:
        return 0
# def Clear():
#     Nx = [1, 0, -1, 0, 1, 1, -1, -1]
#     Ny = [0 ,1, 0, -1, 1, -1, 1, -1]
#     for i in range(1, P.width - 1):
#         for j in range(1, P.height - 1):
#             cnt = 0
#             for k in range(8):
#                 if check(F.getpixel((i + Nx[k] * 1, j + Ny[k] * 1))):
#                     cnt += 1
#             if cnt < 1:
#                 F.putpixel((i, j), (255, 255, 255))         
lines = cv.HoughLinesP(Prep, 1.0, np.pi / 180, 20, np.array([]), minLineLength=100, maxLineGap=10)
lineImg = np.zeros((Prep.shape[0], Prep.shape[1], 3), dtype=np.uint8)
print(lines)
jianchang = 0
xiuzheng = 100
for line in lines:
    for x1, y1, x2, y2 in line:
    #    print(x1, x2, y1, y2)
    #    cv.line(Prep, (x1, y1), (x2, y2), [255, 255, 0], 2)
        cv.circle(lineImg, (x1, y1), 10, color=(255, 255, 0), thickness= 3)
        cv.circle(lineImg, (x2, y2), 10, color=(255, 255, 0), thickness= 3)
        
        tmp = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # if tmp < xiuzheng:
        #     continue
        Points.append((x1, y1))
        Points.append((x2, y2))
        jianchang += tmp
jianchang /= len(lines)
print(jianchang)
plt.subplot(121),plt.imshow(Prep, cmap= "gray")
plt.subplot(122),plt.imshow(lineImg,cmap = 'gray')
plt.show()
cv.waitKey(0)
#print(len(lines))
A = len(Points)
Analy = bingchaji(A)
for i in range(A):
    for j in range(i + 1, A):
        if (Points[i][0] - Points[j][0]) ** 2 + (Points[i][1] - Points[j][1]) ** 2 <= (jianchang / 2) ** 2:
            Analy.merge(i, j)
Analy.Prt(A)
W = Analy.count(A)
Ts = len(W)
Qs = 4 * Ts
B = []
print(W)
for i in range(Ts):
    B.append(W[i][1])
B.sort()
print(B)
sum = 0
F = 4
for i in range(Ts):
    B[i] = ceil(B[i] / F)
    sum += B[i]
print(B)
Qs -= sum
Qs = ceil(Qs / 2) * 2
print("C%dH%d" % (Ts, Qs))
# Txt = zhuaqu(path)
# print(Txt)
# for k in Txt:
#     Name = k["words"]
#     Ts -= 1
#     Qs -= 2
#     Qs += Text(Name)
#print("C%dH%d" % ( Ts, Qs), end="") 
# for k in Txt:
#     Name = k["words"]
#     print(Name, end="")
#P.show() 
#找周围的点
# def manage(SP, Dir):
#     Now = Findnear(SP)
#     for j in Now:
#         if (Now[1] - j[1]) * Dir[0] - (Now[0] - j[0]) * Dir[1] <= 0.1:
#             return 0      
# #P = P.filter(ImageFilter.SMOOTH_MORE)
# Lines = []
# #P.show()
# #Lines以ax+by+c=0形式存储线条
# for i in range(P.width):
#     for j in range(P.height):
#         Point = P.getpixel((i, j))
#         if check(Point):
#         #    P.putpixel((i, j), (255, 0, 0))
#             V = Findnear((i, j))
#             for k in V:
#                 if k[0] == i:
#                     Lines.append((1, 0, -i))
#                 elif k[1] == j:
#                     Lines.append((0, 1, -j))
#                 else:
#                     M1 = k[1] - j
#                     M2 = i - k[0]
#                     M3 = j * k[0] - i * k[1]
#                     Mb = gcd(abs(M1), gcd(abs(M2), abs(M3)))
#                     M1 /= Mb; M2 /= Mb; M3 /= Mb; 
#                     Gap = 0.2687
#                     if (abs(abs(M2 / M1) - sqrt(3)) > Gap) & (abs(abs(M1 / M2) - sqrt(3)) > Gap):
#                         continue
#                     Lines.append((M1, M2, M3))
#             Points.append((i, j)) #抓取所有黑色像素，描边
#         #    Q.putpixel((i, j), (0, 0, 0))
# #P.show() 
# print(Lines)
# Bn = len(Lines)
# # for m in range(Bn):
# #     if m == len(Lines) - 2:
# #         break
# #     h = Lines.count(Lines[m])
# #     if h <= 2:
# #         Lines = Lines[:m] + Lines[m+1:]
# #         m -= 1
# #Lines = set(Lines)
# #print(len(Lines))
# for K in Lines:
#     for X in range(P.width):
#         if K[1] == 0:
#         #    print("case1")
#             # if abs(K[2]) < P.height:
#             #     P.putpixel((X, abs(K[2])), (255, 0, 0))
#             continue
#         else:
#         #    print("case2")
#             Y = int((-K[0] / K[1]) * X - (K[2] / K[1]))
#             if Y <= 0:
#                 continue
#             if Y >= P.height:
#                 continue
#             P.putpixel((X, Y), (255, 0, 0))
# P.show()

######

# for i in Points:
#     G.putpixel(i, (0, 0, 0))
# #黑线
# G.show()
# Sx = 0
# Sy = 0
# for Sx in range(G.width):
#     for Sy in range(G.height):
#         if check(G.getpixel((Sx, Sy))):
#             break
# print(Sx, Sy)
# Slsd = []
# for k in range(G.height):
#     if check(G.getpixel((Sx + R, k))):
#         Slsd.append(k)

# for i in range(Sx + 1, G.width):
#     for j in range(G.height)