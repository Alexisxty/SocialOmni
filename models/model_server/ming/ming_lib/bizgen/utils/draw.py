import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
import os
import copy
def draw_bbox(index,layout,dir):
    color_list=['red', 'green', 'blue', 'purple', 'orange', 'pink', 'brown', 'black', 'gray', 'cyan', 'magenta', 'olive', 'lime', 'teal', 'navy', 'maroon', 'aqua',  'silver', 'gold', 'indigo', 'violet', 'tan', 'khaki', 'coral', 'salmon', 'tomato', 'orangered', 'darkorange', 'darkred', 'darkgoldenrod', 'darkkhaki', 'darkolivegreen', 'darkseagreen', 'darkgreen', 'darkcyan', 'darkturquoise', 'darkslategray', 'darkblue', 'darkviolet', 'darkmagenta', 'darkorchid', 'darkpink', 'darksalmon', 'darkslateblue', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'goldenrod', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'ivory', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'thistle', 'turquoise', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
    bboxes=[[x['top_left'][1],x['top_left'][0],x['bottom_right'][1],x['bottom_right'][0]] for x in layout]
    font=ImageFont.truetype('assets/arial.ttf', size=30)
    img=Image.new('RGB', (bboxes[0][3],bboxes[0][2]), (255, 255, 255))
    
    draw=ImageDraw.Draw(img)
    for n,bbox in enumerate(bboxes):
        label=f"Layer {n}"
        target_bbox=[bbox[1],bbox[0],bbox[3],bbox[2]]
        label_size = draw.textsize(label, font)
        text_origin = np.array([target_bbox[0], target_bbox[1]])
        try:
            draw.rectangle([target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]],outline=color_list[n],width=4)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color_list[n])
            draw.text(text_origin, str(label), fill=(255, 255, 255), font=font)
        except:
            pass
    del draw
    img.save(f'{dir}/{index}_bbox.png')




def draw_lcfg(index,bboxes,dir,guidance=7):
    cfg=[x['cfg'] if "cfg" in x.keys() else guidance for x in bboxes]
    if all(guidance==x for x in cfg):
        return
    bboxes=[[x['top_left'][1],x['top_left'][0],x['bottom_right'][1],x['bottom_right'][0]] for x in bboxes]

    data = np.ones((bboxes[0][2], bboxes[0][3])) * guidance
    for n,bbox in enumerate(bboxes):
        data[bbox[0]:bbox[2],bbox[1]:bbox[3]]=cfg[n]
    fig, ax = pyplot.subplots(figsize=(5, 5))
    levels = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15]
    cmap = plt.cm.get_cmap('hot_r', len(levels) - 1)
    norm = plt.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    cax = ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical',boundaries=levels,ticks=levels)
    cbar.set_label('CFG Value') 
    tick_step = 5
    cbar.set_ticks(0.1*np.arange(10, 150, tick_step)) 

    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    img.save(f'{dir}/{index}_lcfg.png')

def get_text_from_prompt(prompt):
    import re
    texts = []
    patterns = [r'\"(.*?)\"', r'‘(.*?)’', r'“(.*?)”']
    for pattern in patterns:
        texts.extend(re.findall(pattern, prompt))

    return " ".join(texts)