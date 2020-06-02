#!/usr/bin/python
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

import matplotlib
print(matplotlib.matplotlib_fname())

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["figure.titlesize"] = 'medium'
plt.subplots_adjust(wspace =0, hspace =0)
with open('./shandong.csv', mode='r', encoding='utf8') as fin:
  arr = fin.readlines()
# 4*3
row = 5
col = 8
figure, axes = plt.subplots(row,col, figsize=(15, 10)) # facecolor=(0.5,0.5,0.5)
figure.suptitle('蒙牛纯甄vs伊利安慕希(山东)', fontsize=16)
figure.set_dpi(300)
# figure.tight_layout()
figure.tight_layout(pad=4, w_pad=1, h_pad=1)


for j in range(0,row):
  for i in range(0,col):
      if j == 0:
          axes[j][i].set_title('Week {:02d}'.format( i + 10), fontsize='14')
      # ts = arr[j * col + i].strip().split(',')
      ts = tuple(map(int, arr[j * col + i].strip().split(',')))
      venn2(ts,
            set_labels = ('', ''),
            set_colors=('#00A13A','#D72836'),
            alpha=0.7,
            ax=axes[j][i])
plt.subplots_adjust(wspace =0, hspace =0)
plt.savefig('demo.png')