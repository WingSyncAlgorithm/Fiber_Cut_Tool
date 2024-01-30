import math

def trans_x(s1x, s2x, s1y, s2y, sin_val, cos_val, len_change):
    Re = (cos_val * s2x) - (cos_val * s1x) - (sin_val * s2y) + (sin_val * s1y) # 旋轉矩陣轉換
    s3x = s1x + len_change * Re
    return s3x

def trans_y(s1x, s2x, s1y, s2y, sin_val, cos_val, len_change):
    Re = (sin_val * s2x) - (sin_val * s1x) + (cos_val * s2y) - (cos_val * s1y) # 旋轉矩陣轉換
    s3y = s1y + len_change * Re
    return s3y

def calculate_third_point(s1,s2,length_12,length_13,length_23):
    cos312 = (length_12**2 + length_13**2 - length_23**2)/(2.0 * length_12 * length_13) # 餘弦定理
    sin312_1 = math.sqrt(1 - cos312**2)
    sin312_2 = -math.sqrt(1 - cos312**2)
    length_change = length_13 / length_12

    s3x1 = trans_x(s1[0], s2[0], s1[1], s2[1], sin312_1, cos312, length_change)
    s3y1 = trans_y(s1[0], s2[0], s1[1], s2[1], sin312_1, cos312, length_change) 
    s3x2 = trans_x(s1[0], s2[0], s1[1], s2[1], sin312_2, cos312, length_change)
    s3y2 = trans_y(s1[0], s2[0], s1[1], s2[1], sin312_2, cos312, length_change)

    print(f"s3 的x,y坐標為 ({s3x1:.3f},{s3y1:.3f}) 或 ({s3x2:.3f},{s3y2:.3f})")

if __name__ == "__main__":
    s1 = 0,0 #欲測試的s1座標
    s2 = 3,0 #欲測試的s2座標
    calculate_third_point(s1,s2,3,4,5) #給定s1,s2座標及三邊邊長（依序為s1s2,s1s3,s2s3），計算s3座標