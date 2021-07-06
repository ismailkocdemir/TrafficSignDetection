import os
from math import floor
import glob
from collections import Counter
import cv2
import numpy as np

# Some extracted stats [Given as (SIGN_TYPE, FREQUENCY) pairs]
SIGNS_VISIBLE_ONLY = dict([(None, -1), ('PRIORITY_ROAD', 330), ('PASS_RIGHT_SIDE', 178), ('70_SIGN', 158), ('PEDESTRIAN_CROSSING', 146), ('GIVE_WAY', 116), ('50_SIGN', 104), ('80_SIGN', 62), ('NO_STOPPING_NO_STANDING', 61), ('100_SIGN', 54), ('90_SIGN', 39), ('110_SIGN', 34), ('60_SIGN', 28), ('30_SIGN', 26), ('PASS_EITHER_SIDE', 24), ('120_SIGN', 24), ('NO_PARKING', 23), ('PASS_LEFT_SIDE', 15), ('STOP', 1)])
SIGNS_ALL =  dict([(None, -1), ('PRIORITY_ROAD', 470), ('PASS_RIGHT_SIDE', 351), ('PEDESTRIAN_CROSSING', 337), ('GIVE_WAY', 261), ('70_SIGN', 255), ('50_SIGN', 223), ('80_SIGN', 106), ('110_SIGN', 98), ('120_SIGN', 92), ('NO_STOPPING_NO_STANDING', 77), ('100_SIGN', 77), ('90_SIGN', 64), ('60_SIGN', 48), ('30_SIGN', 45), ('NO_PARKING', 39), ('PASS_EITHER_SIDE', 31), ('STOP', 21), ('PASS_LEFT_SIDE', 19), ('URDBL', 12)])


def parse_annotations(file_path, visible_only=False):
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
                    if sign_info[0].strip() != 'MISC_SIGNS' and sign_info[-1].strip() != 'OTHER':
                    
                        if visible_only and sign_info[0].strip() != 'VISIBLE':
                            continue

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
            
        sign_counter = Counter(sign_list)
        #sign_cat_counter = Counter(sign_cat_list)
        print('Images with sign:', len(images_with_sign))
        print('Images without sign:', len(images_without_sign))
        print('Images with Misc. sign:', len(images_with_misc_sign))
        #print('Sign categories:', len(sign_cat_counter))
        print('Sign names:', sign_counter.most_common())


def get_dynamic_range(image_file, bbox, remove_outliers=True):
    rgb = cv2.imread(image_file, cv2.IMREAD_COLOR).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    gray += 1e-6
    if remove_outliers:
        darkest = np.percentile(gray, 1)
        brightest = np.percentile(gray, 99)
        gray[gray>brightest] = brightest
        gray[gray<darkest] = darkest

    # simple dynamic range
    #return np.log2(gray.max()) - np.log2(gray.min())

    # image key value
    log_avg = np.mean( np.log(gray) )
    log_min = np.log(np.min(gray))
    log_max = np.log(np.max(gray))
    return (log_avg - log_min)/(log_max - log_min)


def save_bbox_as_image(image_file, bbox, save_path):
    cv2.imwrite(
            save_path,
            cv2.imread(image_file, 
                cv2.IMREAD_COLOR)[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    )


def prepare_haar_data(image_folder, 
                    annotation_file, 
                    extra_background_folder=None,
                    visible_only = False,
                    sign_filter = None,
                    freq_filter = None
                    ):
    
    assert extra_background_folder, "provide background images"

    positive_lines = []
    negative_lines = []
    largest_image = None
    largets_coords = None
    maximum_dr = 0.

    if sign_filter:
        visible_only = True

    
    ########### COLLECT THE POSITIVE SAMPLES FROM THE ANNOTATION FILE ###########
    with open(annotation_file, 'r') as af:
        lines = af.readlines()
        object_count = 0
        for l in lines:
            fields = l.strip().split(':')
            image_name = fields[0].strip()
            new_line = os.path.join(image_folder, image_name)
            # image contains signs
            if fields[1] != '':
                sign_fields = fields[1].split(';')
                sign_coords = []
                for _sign in sign_fields:
                    if _sign == '':
                        break
                    sign_info = _sign.strip().split(',')
                    # image contains signs with descriptive information                    
                    if sign_info[0] != 'MISC_SIGNS':
                        # only collect a single sign
                        if sign_filter and sign_info[-1].strip() != sign_filter:
                            continue
                        # only collect signs that are tagged as visible, rather than blurred or occluded.
                        if visible_only and sign_info[0].strip() != "VISIBLE":
                            continue
                        coords = [int(floor(float(item))) for item in sign_info[1:5]]
                        coords = coords[2], coords[3], coords[0] - coords[2], coords[1] - coords[3] 
                        coords = list(coords)
                        # fix boxes that goes out ouf bound
                        if coords[0] + coords[2] >= 1280:
                            coords[2] = 1280 - coords[0] - 1
                        if coords[1] + coords[3] >= 960:
                            coords[3] = 960 - coords[1] - 1
                        # if sign filter is given, we will find a large enough bbox with most details, which will be used to generate training samples.
                        if sign_filter:
                            dynamic_range = get_dynamic_range(os.path.join(image_folder, image_name), coords)
                            area = coords[2] * coords[3]
                            if dynamic_range > maximum_dr and area > 4000:
                                maximum_dr = dynamic_range
                                largest_image = image_name
                                largets_coords = coords
                        # else, just add the bbox to the positive samples, which will be directly fed to the classifier.
                        else:
                            coords = ' '.join([str(c) for c in coords])
                            sign_coords.append(coords)
                if not sign_filter and len(sign_coords) > 0:
                    object_count += len(sign_coords)
                    num_of_samples = str(len(sign_coords))
                    coords_all = ' '.join(sign_coords)
                    new_line = new_line + ' ' + num_of_samples + ' ' + coords_all
                    positive_lines.append(new_line)

            else:
                pass
    ##################### POSITIVE COLLECTION ENDS ##########################################



    #################### COLLECT THE NEGATIVE SAMPLES (background) FROM GIVEN FOLDER ########
    for img_path in glob.glob(os.path.join(extra_background_folder, '*')):
        if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
            continue
        negative_lines.append(img_path)
    #################### NEGATIVE COLLECTION ENDS ##########################################
    

    vec_filename = 'vec'
    ### if sign filter is given, then save the largest bbox found during collection as an image.
    if sign_filter and largest_image:
        vec_filename += "_{}".format(sign_filter)
        image_with_largest_sign = os.path.join(image_folder, largest_image)
        sign_save_path = "largest_sign/{}.jpg".format(sign_filter)
        save_bbox_as_image( image_with_largest_sign,
                            largets_coords,
                            sign_save_path
        )
    ### else, just write the list of all positive collected samples (from all signs) to the file
    else:
        vec_filename += '_ALLSIGNS'
        positive_filename = "positive_samples_ALLSIGNS"
        if visible_only:
            positive_filename += "_visible"
            vec_filename += "_visible"            
        positive_filename += ".txt"
        with open(positive_filename, 'w+') as ps:
            for l in positive_lines:
                ps.write(l + '\n')
        ps.close()
    ### write the list of negative images to the file
    negative_filename = 'negative_samples.txt' 
    with open(negative_filename, 'w+') as ns:
        for l in negative_lines:
            ns.write(l + '\n')
    ns.close()


    ############ CREATE TRAINING SAMPLES WITH OPENCV'S TOOL, USING THE COLLECTED SAMPLES ############
    print("GENERATING:", vec_filename)
    vec_filename += ".vec"
    if sign_filter:
        if largest_image:
            os.system('opencv_createsamples -img {} -vec {} -num {} -w 32 -h 32'.format(
                        sign_save_path, 
                        vec_filename,
                        3000
                        )
            )
    else:
        os.system('opencv_createsamples -info {} -vec {} -num {} -w 32 -h 32'.format(
                    positive_filename, 
                    vec_filename,
                    object_count
                    )
        )
    print("===================================================")



if __name__ == '__main__':
    #parse_annotations('dataset/annotations.txt', visible_only=True)
    
    script_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(script_path, 'data/dataset'))   
    for sgn,freq in SIGNS_VISIBLE_ONLY.items():
        prepare_haar_data('/home/ismail/git_repos/TrafficSignDetection/data/dataset/img', 
                        '/home/ismail/git_repos/TrafficSignDetection/data/dataset/annotations.txt', 
                        extra_background_folder='/home/ismail/git_repos/TrafficSignDetection/data/dataset/img_bg',
                        visible_only=True,
                        sign_filter=sgn
                        )
