





import os
import shutil





folder_A = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/thermal/cam6/'  
folder_B = 'D:/works/studio/data/SYSU-MM01/SYSU-MM01/thermal/test/cam6/'  


if not os.path.exists(folder_B):
    os.makedirs(folder_B)


with open('D:/works/studio/data/SYSU-MM01/SYSU-MM01/test_id.txt', 'r') as file:
    test_ids = [int(id.strip()) for id in file.readline().split(',')]


for root, dirs, files in os.walk(folder_A):
    for folder_name in dirs:
        
        folder_number = int(folder_name)

        
        if folder_number in test_ids:
            
            source_folder_path = os.path.join(root, folder_name)
            target_folder_path = os.path.join(folder_B, folder_name)

            
            shutil.move(source_folder_path, target_folder_path)
            print(f"移动子文件夹 {folder_name} 到 {target_folder_path}")

print("任务完成！")
