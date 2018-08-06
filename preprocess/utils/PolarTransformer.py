# -*- coding: utf-8 -*-

class PolarTransformer():

    def __init__(self, h_src, w_src):
        self.h_src = h_src
        self.w_src = w_src
        
        self.x_center, self.y_center = h_src // 2, w_src//2
        
        self.w_dst = w_src * 2
        self.h_dst = h_src // 2
        
        # dst中のpxからsrc中のpxへのmappingをあらかじめ計算しておく
        self.__create_polar_map()
    
    def __create_polar_map(self):
        self.polar_map = [[0 for i in range(self.w_dst)] for j in range(self.h_dst)]
        
        for h_px in range(self.h_dst):
            r = h_px
            for w_px in range(self.w_dst):
                t = 2 * math.pi * (float(w_px + 1) / self.w_dst)
                x = r * math.cos(t)
                y = -r * math.sin(t)
                self.polar_map[h_px][w_px] = (y, x)
      
    def transform(self, src, h_top=0, h_bottom=1, w_left=0, w_right=1, interpolation=True):
        '''
        height * h_topからheight * h_bottomまでが変換対象になる
        width * w_leftからwidth * w_leftまでが変換対象になる
        '''
        assert h_top < h_bottom <= 1
        assert w_left < w_right <= 1
        dst = np.zeros((self.h_dst, self.w_dst, 3), np.uint8)
        
        top = int(self.h_dst * h_top)
        bottom = int(self.h_dst * h_bottom)
        left = int(self.w_dst * w_left)
        right = int(self.w_dst * w_right)
        for h_px in range(top, bottom):
            for w_px in range(left, right):
                src_y, src_x = self.polar_map[h_px][w_px]
                src_iy, src_ix = int(src_y), int(src_x)
                y = self.y_center + src_iy
                x = self.x_center + src_ix
                if not interpolation:
                    dst[h_px][w_px] = src[y][x]
                else:
                    if y > src.shape[0] - 2: y = src.shape[0] - 2
                    if x > src.shape[1] - 2: x = src.shape[1] - 2
                    dx = abs(src_x - src_ix)
                    dy = abs(src_y - src_iy)
                    dst[h_px][w_px] = (1-dx) * (1-dy) * src[y][x] \
                        + dx * (1-dy) * src[y][x+1] \
                        + (1-dx) * dy * src[y+1][x] \
                        + dx * dy * src[y+1][x+1]
        return dst[top:bottom, left:right]