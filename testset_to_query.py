




import os
import random
import shutil


folder_A = 'D:/works/studio/data/RegDB/RegDB/split/bounding_box_test/Visible/'
folder_B = 'D:/works/studio/data/RegDB/RegDB/split/bounding_box_test/Visible2/'


subfolders_A = [f for f in os.listdir(folder_A) if os.path.isdir(os.path.join(folder_A, f))]


for folder_i in subfolders_A:
    
    folder_A_i = os.path.join(folder_A, folder_i)

    
    image_files = [f for f in os.listdir(folder_A_i) if f.endswith('.bmp')] 

    
    num_images_to_cut = len(image_files) // 2

    
    images_to_cut = random.sample(image_files, num_images_to_cut)

    
    folder_B_i = os.path.join(folder_B, folder_i)

    
    if not os.path.exists(folder_B_i):
        os.makedirs(folder_B_i)

    
    for image in images_to_cut:
        source_path = os.path.join(folder_A_i, image)
        target_path = os.path.join(folder_B_i, image)

        
        shutil.move(source_path, target_path)

        
        print(f"剪切文件: {source_path} 到 {target_path}")

print("任务完成！")
