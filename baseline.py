import os
from PIL import Image
import pytesseract
import re
import Levenshtein

if __name__ == "__main__":

    dataset_dir = "dataset_hdr"
    train_values_file = os.path.join(dataset_dir, "dataset.txt")

    with open(train_values_file, 'r') as f:
        lines = [line.strip('\r\n') for line in f.readlines()]

    image_path_list, value_list = [], []
    
    for i, line in enumerate(lines):
        split = line.split(';')
        img_path = split[0]
        value = split[1]
        train = split[2]
        
        #if train == "1":
        image_path_list.append(os.path.join(dataset_dir, img_path))
        value_list.append(value)
    
    print(len(image_path_list))
    result = [0,0,0,0,0,0,0,0,0]
    custom_oem_psm_config = r'--oem 3 --psm 10'
    
    for i in range(len(image_path_list)):
    
        img_path = image_path_list[i]
        expected_value = value_list[i]
        
        val = pytesseract.image_to_string(Image.open(img_path), config=custom_oem_psm_config).strip()
        val = val.replace("o", "0")
        val = val.replace("O", "0")
        val = val.upper()
        val = re.sub('[^A-Z0-9]+', '', val)
        
        if val == expected_value:
            result[0] += 1
        else:
            distance = Levenshtein.distance(val, expected_value)
            if distance > 8:
                distance = 8
            result[distance] += 1
            
        if i % 20 == 0:
            print(result)
    
    print("Final:")
    print(result)
            
    
