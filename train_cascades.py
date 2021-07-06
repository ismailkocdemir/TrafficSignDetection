import os
import glob

def train_cascades_all_signs():
    vec_files = glob.glob('data/dataset/*.vec')

    assert len(vec_files), "No training samples found. run prepare_data.py to create training samples."

    for vec_f in vec_files:
        cascade_dir = ['data/cascades/']
        cascade_name = vec_f.split('/')[-1].split('.')[0].split('_')
        if "real" in cascade_name:
            cascade_dir.append("real")
        else:
            cascade_dir.append("augmented")

        sign_name = cascade_name[1:-1]
        if len(sign_name) == 1:
            cascade_dir.append(sign_name[0])
        else:
            cascade_dir.append("_".join(sign_name))
        
        overall_dir = os.path.join(*cascade_dir)
        if not os.path.exists(overall_dir):
            os.makedirs(overall_dir)
        
        background_image_list = "data/dataset/negative_samples.txt"
        num_pos = 2000
        if 'real' in cascade_dir:
            with open("data/dataset/count_{}.txt".format(cascade_dir[-1]), 'r') as f:
                num_pos = int(f.readlines()[0])

        os.system("opencv_traincascade -data {} -vec {} -bg {} -numPos {} \
            -w 32 -h 32 \
            -mode ALL -acceptanceRatioBreakValue 10e-5 \
            -numStages 20  -maxFalseAlarmRate 0.5".format(
                                                        overall_dir,
                                                        vec_f,
                                                        background_image_list,
                                                        num_pos
                                                    )
        )

if __name__ == '__main__':
    train_cascades_all_signs()