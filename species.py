import numpy as np
import argparse
from tensorflow import lite as tflite


def loadMetaModel():

    global M_INTERPRETER
    global M_INPUT_LAYER_INDEX
    global M_OUTPUT_LAYER_INDEX
    global CLASSES

    # Load TFLite model and allocate tensors.
    M_INTERPRETER = tflite.Interpreter(model_path = './BirdNET_GLOBAL_6K_V2.4_MData_Model_FP16.tflite')
    M_INTERPRETER.allocate_tensors()
    # Get input and output tensors.
    input_details = M_INTERPRETER.get_input_details()
    output_details = M_INTERPRETER.get_output_details()
    # Get input tensor index
    M_INPUT_LAYER_INDEX = input_details[0]['index']
    M_OUTPUT_LAYER_INDEX = output_details[0]['index']
    # Load labels
    CLASSES = []
    labelspath = './labels.txt'
    with open(labelspath, 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))


def predictFilter(lat, lon, week):
    # Does interpreter exist?
    try:
        if M_INTERPRETER is None:
            loadMetaModel()
    except Exception:
        loadMetaModel()
    # Prepare mdata as sample
    sample = np.expand_dims(np.array([lat, lon, week], dtype='float32'), 0)
    # Run inference
    M_INTERPRETER.set_tensor(M_INPUT_LAYER_INDEX, sample)
    M_INTERPRETER.invoke()

    return M_INTERPRETER.get_tensor(M_OUTPUT_LAYER_INDEX)[0]


def explore(lat, lon, week, threshold):
    # Make filter prediction
    l_filter = predictFilter(lat, lon, week)
    # Apply threshold
    l_filter = np.where(l_filter >= threshold, l_filter, 0)
    # Zip with labels
    l_filter = list(zip(l_filter, CLASSES))
    # Sort by filter value
    l_filter = sorted(l_filter, key=lambda x: x[0], reverse=True)
    return l_filter


def getSpeciesList(lat, lon, threshold=0.05, sort=False):
    # Make species list
    slist = []
    for week in range(1, 53):
        # Extract species from model
        pred = explore(lat, lon, str(week), threshold)
        for p in pred:
            if p[0] >= threshold:
                slist.append(p[1])
    return sorted(list(set(slist)))


def outputTextFile(slist):
    file_path = "species_list.txt"
    with open(file_path, 'w') as file:
        for item in slist:
            file.write(item + '\n')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Get list of species for a given ' \
                                     'location with BirdNET. Sorted by occurrence ' \
                                     'frequency.')
    parser.add_argument('latitude', type=str, help='latitude of birdnet station')
    parser.add_argument('longitude', type=str, help='longitude of birdnet station')
    parser.add_argument('--loc_threshold', type=float, default=0.05, 
                        help='Occurrence frequency threshold. Defaults to 0.05.')
    args = parser.parse_args()

    # Get species list
    species_list = getSpeciesList(args.latitude, args.longitude, 
                                  args.loc_threshold, False)
    outputTextFile(species_list)
