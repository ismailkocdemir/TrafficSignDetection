import os
import numpy as numpy
from math import floor
import glob

def parse_annotations(file_path):
    
    sign_list = []
    sign_cat_list = []
    images_with_sign = []
    images_without_sign = []
    images_with_misc_sign = []
    
    with open(file_path, 'r') as af:
        lines = af.readlines()
        for l in lines:
            fields = l.strip().split(':')
            image_name = fields[0].strip()
            
            if fields[1] != '':
                sign_fields = fields[1].split(';')
                is_misc = True
                for _sign in sign_fields:
                    if _sign == '':
                        break
                    sign_info = _sign.strip().split(',')
                    
                    if sign_info[0] != 'MISC_SIGNS':
                        if is_misc == True:
                            images_with_sign.append(image_name)
                            is_misc = False
                        
                        sign_cat, sign_name = sign_info[-2:]
                        sign_list.append(sign_name.strip())
                        sign_cat_list.append(sign_cat.strip())
                
                if is_misc:
                    images_with_misc_sign.append(image_name)

            else:
                images_without_sign.append(image_name)
            
        sign_counter = set(sign_list)
        sign_cat_counter = set(sign_cat_list)
        print('Images with sign:', len(images_with_sign))
        print('Images without sign:', len(images_without_sign))
        print('Images with Misc. sign:', len(images_with_misc_sign))
        print('Sign categories:', len(sign_cat_counter))
        print('Sign names:', len(sign_counter))


def prepare_haar_data(image_folder, annotation_file, extra_background_folder=None):
    positive_lines = []
    negative_lines = []
    
    with open(annotation_file, 'r') as af:
        lines = af.readlines()
        object_count = 0
        for l in lines:
            fields = l.strip().split(':')
            image_name = fields[0].strip()
            new_line = os.path.join(image_folder, image_name)
            
            if fields[1] != '':

                sign_fields = fields[1].split(';')
                sign_coords = []
                for _sign in sign_fields:
                    if _sign == '':
                        break
                    
                    sign_info = _sign.strip().split(',')                    
                    if sign_info[0] != 'MISC_SIGNS':
                        coords = [int(floor(float(item))) for item in sign_info[1:5]]
                        coords = coords[2], coords[3], coords[0] - coords[2], coords[1] - coords[3] 
                        coords = list(coords)
                        if coords[0] + coords[2] >= 1280:
                            coords[2] = 1280 - coords[0] - 1
                        if coords[1] + coords[3] >= 960:
                            coords[3] = 960 - coords[1] - 1

                        coords = ' '.join([str(c) for c in coords])
                        sign_coords.append(coords)
                
                if len(sign_coords) > 0:
                    object_count += len(sign_coords)
                    num_of_samples = str(len(sign_coords))
                    coords_all = ' '.join(sign_coords)
                    new_line = new_line + ' ' + num_of_samples + ' ' + coords_all
                    positive_lines.append(new_line)

            else:
                negative_lines.append(new_line)
        
        if extra_background_folder:
            for img_path in glob.glob(os.path.join(extra_background_folder, '*')):
                if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
                    continue
                negative_lines.append(img_path)

        with open('positive_samples.txt', 'w+') as ps:
            for l in positive_lines:
                ps.write(l + '\n')
        ps.close()

        with open('negative_samples.txt', 'w+') as ns:
            for l in negative_lines:
                ns.write(l + '\n')
        ns.close()

        print('Number of positive samples:', object_count)
        print('Number of negative samples:', len(negative_lines))


def prepare_svm_data():
    pass


if __name__ == '__main__':
    #parse_annotations('annotations_Set1Part0.txt')
    prepare_haar_data('/home/ismail/ford/code/detection/data/img', 'annotations.txt', '/home/ismail/ford/code/detection/data/bg')




