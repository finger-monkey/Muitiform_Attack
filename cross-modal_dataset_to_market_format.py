












import os
import shutil








folder_A = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/thermal/test/cam3/'
folder_B = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/deal/thermal/test/cam3/'





if not os.path.exists(folder_B):
    os.makedirs(folder_B)



cam_value = "c3_s1"

for folder_K in os.listdir(folder_A):
    
    if folder_K.isdigit():
        
        pid = folder_K.zfill(4)

        
        
        image_files = [f for f in os.listdir(os.path.join(folder_A, folder_K)) if f.endswith('.jpg')]

        
        for i, image_file in enumerate(image_files, start=1):
            
            length = str(i).zfill(6)  
            
            new_filename = f"{pid}_{cam_value}_{length}_01.jpg"

            
            source_path = os.path.join(folder_A, folder_K, image_file)
            target_path = os.path.join(folder_B, new_filename)

            
            shutil.copy2(source_path, target_path)

            
            print(f"复制文件: {source_path} 到 {target_path}")

print("任务完成！")
