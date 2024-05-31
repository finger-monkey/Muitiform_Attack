"""
chatgpt提示词：
请帮我用python写一个代码：遍历文件夹A中的每一个文件夹K(记录下文件夹i的名字，它是由数字组成的文件夹编号pid，
把pid用四位宽度显示，不足四位的部分用零来填充)，计算文件夹K有多少张图片，然后遍历文件夹K下的每一张图片，把图片复制到文件夹B并重命名，
重命名为pid_cam_length_01，其中pid是刚才提到的四位宽度的文件夹编号，其中的cam="c{i}_s1"(其中i是变量，具体值由用户指定)。
最后的length由遍历迭代器器决定，例如如果当前处理的图片是文件夹K中的第3张图片，那么它的值就为000003（用6位宽度来表示）.

第二次修正：
这里的文件夹结构你弄错了，文件夹A下有若干文件夹（包括K）,K下有若干张图片

"""


import os
import shutil

# 指定文件夹A和文件夹B的路径
# folder_A = 'D:/works/studio/data/RegDB/RegDB/split/bounding_box_test/Visible/'
# folder_B = 'D:/works/studio/data/RegDB/RegDB/split/deal/bounding_box_test/Visible/'

# folder_A = 'D:/works/studio/data/RegDB/RegDB/split/combime/bounding_box_test/'
# folder_B = 'D:/works/studio/data/RegDB/RegDB/split/combime/deal/bounding_box_test/'

folder_A = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/thermal/test/cam3/'
folder_B = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/deal/thermal/test/cam3/'

# folder_A = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/visible/test/cam5/'
# folder_B = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/deal/visible/test/cam5/'

# 确保输出根文件夹存在
if not os.path.exists(folder_B):
    os.makedirs(folder_B)

# 用户指定的cam值
# cam_value = input("请输入cam的值（例如c1_s1）：")
cam_value = "c3_s1"
# 遍历文件夹A中的子文件夹
for folder_K in os.listdir(folder_A):
    # 检查子文件夹是否是数字命名的
    if folder_K.isdigit():
        # 计算文件夹K的pid，并格式化为四位宽度的字符串
        pid = folder_K.zfill(4)

        # 获取文件夹K中的所有图片文件
        # image_files = [f for f in os.listdir(os.path.join(folder_A, folder_K)) if f.endswith('.bmp')]  ##注意图片格式!!!!!!!!!!!!!!!!
        image_files = [f for f in os.listdir(os.path.join(folder_A, folder_K)) if f.endswith('.jpg')]##注意图片格式!!!!!!!!!!!!!!!!

        # 遍历文件夹K中的图片文件
        for i, image_file in enumerate(image_files, start=1):
            # 构建新的文件名
            length = str(i).zfill(6)  # 使用六位宽度表示图片序号
            # new_filename = f"{pid}_{cam_value}_{length}_01.bmp"
            new_filename = f"{pid}_{cam_value}_{length}_01.jpg"

            # 构建源文件和目标文件的完整路径
            source_path = os.path.join(folder_A, folder_K, image_file)
            target_path = os.path.join(folder_B, new_filename)

            # 复制文件并重命名
            shutil.copy2(source_path, target_path)

            # 输出复制文件的信息
            print(f"复制文件: {source_path} 到 {target_path}")

print("任务完成！")
