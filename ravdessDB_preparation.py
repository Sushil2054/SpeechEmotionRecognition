import json
import os
import math
import librosa
import glob

DATASET_PATH = "C:\\Users\\PREDATOR\\Desktop\\project\\code\\ravdess dataset"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050

TRACK_DURATION = 0.03 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    
    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        #print(f"{filenames}\n")
        if dirpath is not dataset_path:
                
            # for filename in filenames:
            #     semantic_label = filename.split("-")[2]
            #     data["labels"].append(semantic_label)
  
# process all audio files in genre sub-dir
            for i, f in enumerate(filenames):
                

                labels = f.split("-")[2]
                # load audio file
                file_path = os.path.join(dirpath, f)
                #print(file_path)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                        # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        #print(labels)
                        #print("==========================================")
                        data["labels"].append(labels)
                        #print("{}, segment:{}".format(file_path, d+1))

          
    data["mapping"] = list(set(data["labels"]))
    data["mapping"].sort()
    print(data["labels"])
    #print(len(data["labels"]))
        #print(data["mapping"])
        #print(len(data["mapping"]))

    for i, label in enumerate(data["mapping"]):
        if label == '01':
            data["mapping"][i] = 'Neutral'
        if label == '02':
            data["mapping"][i] = 'Calm'
        if label == '03':
            data["mapping"][i] = 'Happy'
        if label == '04':
            data["mapping"][i] = 'Sad'
        if label == '05':
            data["mapping"][i] = 'Angry'
        if label == '06':
            data["mapping"][i] = 'Fearful'
        if label == '07':
            data["mapping"][i] = 'Disgust'
        if label == '08':
            data["mapping"][i] = 'Surprised'

    #print(data["mapping"])


    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
    print(len(data["labels"]))

        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)