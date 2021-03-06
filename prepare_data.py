import os
from math import floor
import glob
from collections import Counter
import cv2
import numpy as np

# Some extracted stats [Given as (SIGN_TYPE, FREQUENCY) pairs]
# SIGNS_VISIBLE_ONLY = dict([('PRIORITY_ROAD', 330), ('PASS_RIGHT_SIDE', 178), ('70_SIGN', 158), ('PEDESTRIAN_CROSSING', 146), ('GIVE_WAY', 116), ('50_SIGN', 104), ('80_SIGN', 62), ('NO_STOPPING_NO_STANDING', 61), ('100_SIGN', 54), ('90_SIGN', 39), ('110_SIGN', 34), ('60_SIGN', 28), ('30_SIGN', 26), ('PASS_EITHER_SIDE', 24), ('120_SIGN', 24), ('NO_PARKING', 23), ('PASS_LEFT_SIDE', 15), ('STOP', 1)])
# SIGNS_ALL =  dict([('PRIORITY_ROAD', 470), ('PASS_RIGHT_SIDE', 351), ('PEDESTRIAN_CROSSING', 337), ('GIVE_WAY', 261), ('70_SIGN', 255), ('50_SIGN', 223), ('80_SIGN', 106), ('110_SIGN', 98), ('120_SIGN', 92), ('NO_STOPPING_NO_STANDING', 77), ('100_SIGN', 77), ('90_SIGN', 64), ('60_SIGN', 48), ('30_SIGN', 45), ('NO_PARKING', 39), ('PASS_EITHER_SIDE', 31), ('STOP', 21), ('PASS_LEFT_SIDE', 19), ('URDBL', 12)])


def get_average_brightness_from_bbox(image_file, bbox, remove_outliers=True):
    '''calculate a modified version of log-luminance from the bbox area in the image'''
    rgb = cv2.imread(image_file, cv2.IMREAD_COLOR).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    gray += 1e-6
    if remove_outliers:
        darkest = np.percentile(gray, 1)
        brightest = np.percentile(gray, 99)
        gray[gray>brightest] = brightest
        gray[gray<darkest] = darkest

    # bbox avg. brigthness
    log_avg = np.mean( np.log(gray) )
    log_min = np.log(np.min(gray))
    log_max = np.log(np.max(gray))
    return (log_avg - log_min)/(log_max - log_min)


def save_bbox_as_image(image_file, bbox, save_path, resize=0):
    '''save the given bbox to the save_path as image'''
    new_image = cv2.imread(image_file, 
                            cv2.IMREAD_COLOR)[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                        
    if resize:
        new_image = cv2.resize(new_image, (resize, resize))
    cv2.imwrite(
            save_path,
            new_image
    )

def save_random_crops(image_file, min_size, max_size, num_crops):
    '''save random crops from the image'''
    new_file_names = []
    idx = 0
    while idx < num_crops:
        crop_width = np.random.randint(min_size, max_size)
        crop_height = np.random.randint(min_size, max_size)
        
        # do not use very wide or very long crops.
        ratio = crop_width / crop_height
        if ratio < 0.5 or ratio > 2.:
            continue
        
        # the image dimensions are hard-coded (1280 x 960). maybe it should not be.
        max_x = 1280 - crop_width
        max_y = 960 - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = [x, y, crop_width, crop_height]

        # do not use too bright crops. not useful for background data creation.
        avg_br = get_average_brightness_from_bbox(image_file, crop, False)
        if avg_br > 0.70:
            continue

        split_path = image_file.split('/')
        new_file_name = os.path.join('/',
                                *split_path[:-2], 
                                split_path[-2] + "_cropped", 
                                split_path[-1].replace('.', '_{}.'.format(idx))
                            )

        new_file_names.append(new_file_name)
        save_bbox_as_image(image_file, crop, new_file_name, 100)
        idx += 1

    return new_file_names


def parse_annotations(file_path, visible_only=False):
    '''Parses the annotation files provided in the dataset.
        Extracts some stats like names of the signs and frequency of each sign.

        Args:
            file_path       : annotation file path
            visible_only    : only extract stats of signs that are tagged as visible, excluding blurred or occluded ones.
        Returns:
            images_with_sign        : list of images with labeled sign(s)
            images_without_sign     : list of images without any labeled signs
            most_common_signs       : a dictionary of (sign_names, sign_count) pairs

    '''
    sign_list = []
    sign_cat_list = []
    images_with_sign = []
    images_without_sign = []
    #images_with_misc_sign = []
    
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
                    images_without_sign.append(image_name)
            else:
                images_without_sign.append(image_name)
            
    most_common_signs = Counter(sign_list).most_common()
    print('Images with sign:', len(images_with_sign))
    print('Images without sign:', len(images_without_sign))
    print('Signs and freqs:', most_common_signs)
    
    return (
        images_with_sign,
        images_without_sign,
        most_common_signs
    )


def prepare_training_data(
        image_folder, 
        annotation_file, 
        extra_background=False,
        visible_only=False,
        sign_filter=None,
        use_augmented_data=False,
        use_templates=False,
        create_negative_samples=True
    ):
    '''Prepares the training data before feeding into Haar-based Cascaded Classifier available in the OpenCV,
        Pre-compiled opencv_createsamples binary is used for data creation.

        Args:
            image_folder                : path to training images
            annotation_file             : path to annotation file
            extra_background            : if set, prepare a background data by cropping the no-sign images.
            visible_only:               : if set, use only the signs tagged as visible 
            sign_filter:                : If given, prepare the data only for this sign. 
                                            Else, include all signs and unite them under general signness category
            use_augmented_data:         : If set, new data will be generated by pasting real traffic signs on backgorund images.
            use_templates               : If set, new data will be generated by pasting template traffic signs on backgorund images. 
        Returns:
            -

    '''
    #assert extra_background_folder, "provide background images"
    assert not (use_augmented_data and use_templates), "can not use augmented data and templates at the same time"

    positive_lines = []
    negative_lines = []

    # some variables for keeping the best candidate bbox for each sign
    largest_image = None
    largets_coords = None
    maximum_br = 0.

    # skip occluded or blurred bboxes when searching for a candidate bbox. 
    if sign_filter:
        visible_only = True
    else:
        if use_augmented_data or use_templates:
            print("not creating augmented samples when no sign filter is given")
            return
    
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
                is_misc = True
                for _sign in sign_fields:
                    if _sign == '':
                        break
                    sign_info = _sign.strip().split(',')
                    # image contains signs with bbox information                    
                    if sign_info[0] != 'MISC_SIGNS':
                        is_misc = False
                        
                        if use_templates:
                            continue
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
                        
                        # we will find a large enough bbox with good brightness, which will be used to generate augmented samples.
                        if sign_filter and use_augmented_data:
                            avg_br = get_average_brightness_from_bbox(os.path.join(image_folder, image_name), coords)
                            area = coords[2] * coords[3]
                            if avg_br > maximum_br and area > 4000:
                                maximum_br = avg_br
                                largest_image = image_name
                                largets_coords = coords
                        # else, just add the bbox to the positive samples, which will be directly fed to the classifier.
                        else:
                            coords = ' '.join([str(c) for c in coords])
                            sign_coords.append(coords)
                
                if not (sign_filter and use_augmented_data) and len(sign_coords) > 0:
                    object_count += len(sign_coords)
                    num_of_samples = str(len(sign_coords))
                    coords_all = ' '.join(sign_coords)
                    new_line = new_line + ' ' + num_of_samples + ' ' + coords_all
                    positive_lines.append(new_line)
                
                # no bbox label found
                if is_misc:
                    negative_lines.append(new_line)
            
            # no sign found
            else:
                negative_lines.append(new_line)
    ##################### POSITIVE COLLECTION ENDS ##########################################


    #################### CREATE THE NEGATIVE SAMPLES (background) FROM GIVEN FOLDER ########
    if extra_background and create_negative_samples:
        new_negative_lines = []
        for i, img_path in enumerate(negative_lines):
            print("Creating negative samples by cropping {}/{}".format(i, len(negative_lines)), end='\r')
            new_negative_lines += save_random_crops(img_path, 50, 300, 5)
        
        negative_lines = new_negative_lines
        '''
        for img_path in glob.glob(os.path.join(extra_background_folder, '*')):
            if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
                continue
            negative_lines.append(img_path)
        '''
    #################### NEGATIVE COLLECTION ENDS ##########################################
    

    vec_filename = 'vec'
    ### if sign filter is given, then save the largest bbox found during collection as an image.
    if sign_filter and (use_augmented_data or use_templates):
        if use_templates:
            vec_filename += "_{}_template".format(sign_filter)
        elif largest_image:
            vec_filename += "_{}_augmented".format(sign_filter)
            image_with_largest_sign = os.path.join(image_folder, largest_image)
            sign_save_path = "largest_sign/{}.jpg".format(sign_filter)
            save_bbox_as_image( image_with_largest_sign,
                                largets_coords,
                                sign_save_path
            )
    ### else, just write the list of all positive collected samples (from all signs) to the file
    elif object_count > 10:
        vec_filename += "_{}_real".format(sign_filter) if sign_filter else'_ALLSIGNS_real'
        positive_filename = "positive_samples" + ("_{}".format(sign_filter) if sign_filter else'_ALLSIGNS')               
        positive_filename += ".txt"
        with open(positive_filename, 'w+') as ps:
            for l in positive_lines:
                ps.write(l + '\n')
        ps.close()
    
    if create_negative_samples:
        ### write the list of negative images to the file
        negative_filename = 'negative_samples.txt' 
        with open(negative_filename, 'w+') as ns:
            for l in negative_lines:
                ns.write(l + '\n')
        ns.close()


    ############ CREATE TRAINING SAMPLES WITH OPENCV'S TOOL, USING THE COLLECTED SAMPLES ############
    print("GENERATING:", vec_filename)
    vec_filename += ".vec"
    if sign_filter and (use_augmented_data or use_templates):
        if largest_image or use_templates:
            os.system('opencv_createsamples -img {} -vec {} -num {} -w 64 -h 64'.format(
                        sign_save_path if largest_image else "templates/{}.png".format(sign_filter), 
                        vec_filename,
                        3000
                        )
            )

    elif object_count > 10:
        os.system('opencv_createsamples -info {} -vec {} -num {} -w 32 -h 32'.format(
                    positive_filename, 
                    vec_filename,
                    object_count
                    )
        )

        sign_name = sign_filter if sign_filter else "ALLSIGNS"
        with open("count_{}.txt".format(sign_name), "w+") as cf:
            cf.write(str(object_count))

    print("===================================================")


if __name__ == '__main__':
    _,_, COMMON_SIGNS = parse_annotations('data/dataset/annotations_test.txt', visible_only=True) # just for extracting some info.
    COMMON_SIGNS = dict(COMMON_SIGNS)

    options = [(0,0)]
    script_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(script_path, 'data/dataset'))
    idx = 0   
    for sgn,freq in COMMON_SIGNS.items():
        if freq < 50:
            continue
        for a,t in options:
            prepare_training_data(
                '/home/ismail/git_repos/TrafficSignDetection/data/dataset/img', 
                '/home/ismail/git_repos/TrafficSignDetection/data/dataset/annotations.txt', 
                extra_background=True,
                visible_only=True,
                sign_filter=sgn,
                use_augmented_data=a,
                use_templates=t,
                create_negative_samples=False #(idx == 0)  
            )
            idx += 1
