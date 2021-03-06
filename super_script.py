import re
import os
import glob
import cv2
import time

from termcolor import colored
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import matplotlib.pyplot as plt

BATCH_SIZE = 2
TRAINING_IMGS = 10 
EPOCHS = 2
LR = 1e-4

def display_img(rgb_img):
    plt.imshow(rgb_img)
    # to hide tick values on X and Y axis
    plt.xticks([]), plt.yticks([])
    plt.show()

def get_img(img):
    bgr_img = cv2.imread(img)
    # get bgr and switch to rgb
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

def get_sorted_images(folder):
    files = sorted(glob.glob(folder + "*.png"), key=natural_keys)
    if (len(files) == 0):
        files = sorted(glob.glob(folder + "*.jpg"), key=natural_keys)

    return files

def run(train_root, test_root, skip_to_metrics, verbose):

    test_name = test_root.split("/")[-1] if test_root.split("/")[-1] != "" else test_root.split("/")[-2]
    cmd_dims = " --height=256 --width=256"
    model_type = "lstm"
    model_chkp = model_type+"/checkpoints/"

    train_blur = train_root + "blur/"
    train_sharp = train_root + "sharp/"

    test_blur = test_root + "blur/"
    test_sharp = test_root + "sharp/"

    test_results = test_root

    train_prefix = "/training_set/"
    test_prefix = "/testing_set/"

    curr_dir = os.getcwd()
    full_train_root_blur = curr_dir + train_prefix + train_root + "blur/"
    full_train_root_sharp = curr_dir + train_prefix + train_root + "sharp/"

    full_test_root_sharp = curr_dir + test_prefix + test_root + "sharp/"
    full_test_root_blur = curr_dir + test_prefix + test_root + "blur/"

    training_elapsed_time = 0
    testing_elapsed_time = 0

    start_time = time.time()

    if not skip_to_metrics:

        print()
        print("============== STARTING {} RUN ==============".format(colored(test_name, "red")))
        print("="*(43+len(test_name)))

        # #################### CREATING DATALIST ####################

        print("[+]: Creating datalist")
        files = get_sorted_images(full_train_root_blur)

        f = open("datalist_gopro.txt", "w")
        for i, file in enumerate(files):
            name = file.split("/")[-1]
            line = train_sharp + name + " " + train_blur + name + "\n"
            f.write(line)

            if i == TRAINING_IMGS:
                print("[+]: Breaking at iteration {}".format(i))
                break
        f.close()
        print("[+]: Datalist created")
        print("=========================================")

        # ###################### MODEL TRAINING ######################

        training_start_time = time.time()

        print("[+]: Training model")
        cmd_train = "python run_model.py --phase=train --model "+model_type+" --batch=" + str(BATCH_SIZE) + " --lr="+str(LR)+" --epoch=" + str(EPOCHS) + cmd_dims
        print("[+]: "+cmd_train)
        print("=========================================")

        ret = os.popen(cmd_train).read()
        if verbose:
            print(ret)

        if ("Resource exhausted" in ret) or ("loss = " not in ret):
            print("[-]: Resource Exhausted error when training, exiting")
            exit()

        print("=========================================")
        print("[+]: Model trained successfully")

        training_elapsed_time = time.time() - training_start_time

        # ###################### MODEL TESTING ######################
        testing_start_time = time.time()

        print("[+]: Creating test images")
        cmd_test = "python run_model.py --input_path=./testing_set/" + test_blur + " --output_path=./testing_res/" + test_results + cmd_dims + " --model "+model_chkp
        print("[+]: " + cmd_test)
        print("=========================================")

        ret = os.popen(cmd_test).read()
        if verbose:
            print(ret)

        print("[+]: Test images created")

        testing_elapsed_time = time.time() - testing_start_time


    # ###################### METRICS CALCULATION ######################

    print("[+]: Calculating Metrics ")

    ground_truth_images = get_sorted_images(full_test_root_sharp)
    result_test_images = get_sorted_images("./testing_res/" + test_results)

    ssim_sum = 0
    psnr_sum = 0
    total_imgs = len(ground_truth_images)
    for gt_img, res_img in zip(ground_truth_images, result_test_images):
        left = get_img(gt_img)
        right = get_img(res_img)

        ssim_sum += ssim(left, right, data_range=right.max() - right.min(), multichannel=True)
        psnr_sum += psnr(left, right)

    ssim_avg, psnr_avg = ssim_sum/total_imgs, psnr_sum/total_imgs


    print("\n// ===================================================== //")
    print("// ---------------------- Results ---------------------- //")
    print("// ===================================================== //\n")

    print("Folder: {} | # of imgs: {}\n".format("./testing_res/" + test_results, total_imgs))
    print("======== Averages for {} test set ======== ".format(colored(test_name, "red")))
    print("- SSIM: {} \n- PSNR: {}".format(ssim_avg, psnr_avg))
    print("="*(40+len(test_name)))

    print()
    print("============== SUMMARY ==============")
    print("- Training epochs: {}".format(EPOCHS))
    print("- Batch size: {}".format(BATCH_SIZE))
    print("- Learning rate: {}".format(LR))
    print("- Training images: {}".format(TRAINING_IMGS))
    print("=====================================")
    print()

    if training_elapsed_time > 0:
        print("Training elapsed time: {}".format(time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))))

    if testing_elapsed_time > 0:
        print("Testing elapsed time: {}".format(time.strftime("%H:%M:%S", time.gmtime(testing_elapsed_time))))

    total_elapsed_time = time.time() - start_time
    print("Total elapsed time: {}".format(time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))))


if __name__ == '__main__':

    datasets = ["flickr", "dogs", "gopro"]
    defects = ["full_blur", "inpaint_blur", "inpaint_white"]

    for dataset in datasets:
        for defect in defects:
            # NOTE: assuming under ./training_set/ and ./testing_set/ respectively
            train_root = "{}/{}_{}/".format(dataset.upper(), dataset, defect)
            test_root = "{}/{}_{}/".format(dataset.upper(), dataset, defect)
            skip_to_metrics = False
            verbose = True

            # NOTE: The output images will go under ./testing_res/
            run(train_root, test_root, skip_to_metrics, verbose)
