import re
import os
import glob

BATCH_SIZE = 4
NUM_IMGS = 10
EPOCHS = 2

curr_dir = os.getcwd()
prefix = "/training_set/"
train_root = "FLICKR/flickr_full_blur/train_flickr_full_blur/"
test_root = "FLICKR/flickr_full_blur/"

train_blur = train_root + "blur/"
train_sharp = train_root + "sharp/"

test_blur = test_root + "blur/"
test_sharp = test_root + "sharp/"

test_results = test_root

full_train_root_blur = curr_dir + prefix + train_root + "blur/"
full_train_root_sharp = curr_dir + prefix + train_root + "sharp/"


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


if __name__ == '__main__':

    # #################### CREATING DATALIST ####################
    print("[+]: Creating datalist")
    files = sorted(glob.glob(full_train_root_blur + "*.png"), key=natural_keys)
    if (len(files) == 0):
        files = sorted(glob.glob(full_train_root_blur + "*.jpg"), key=natural_keys)

    f = open("datalist_gopro.txt", "w")
    for i, file in enumerate(files):
        name = file.split("/")[-1]
        line = train_sharp + name + " " + train_blur + name + "\n"
        f.write(line)

        if i == NUM_IMGS:
            print("[+]: Breaking at iteration {}".format(i))
            break
    f.close()
    print("[+]: Datalist created")
    print("----------------------------------")

    # ###################### TRAINING MODEL ######################
    print("[+]: Training model")
    cmd_train = "python run_model.py --phase=train --batch=" + str(BATCH_SIZE) + " --lr=1e-4 --epoch=" + str(EPOCHS)
    print("[+]: "+cmd_train)
    print("----------------------------------")

    ret = os.popen(cmd_train).read()
    print(ret)

    if ("Resource exhausted" in ret) or ("loss = " not in ret):
        print("[-]: Resource Exhausted error, exiting")
        exit()

    print("----------------------------------")
    print("[+]: Model trained")

    # ###################### TESTING MODEL ######################
    print("[+]: Creating test images")
    cmd_test = "python run_model.py --input_path=./testing_set/" + test_blur + " --output_path=./testing_res/" + test_results
    print("[+]: " + cmd_test)
    print("----------------------------------")
    print(os.popen(cmd_test).read())
    print("[+]: Test images created!")
