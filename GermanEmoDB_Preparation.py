import json
import os
import math 
import librosa

DATASET_PATH = "..\\EmotionDB\GermanEmotinalDB"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 0.01
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc = 13, n_fft = 2048, hop_length = 512, num_segment = 5):
    data = {
        "mapping" : [],
        "labels" : [],
        "mfcc" : []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segment)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filesnames) in enumerate(os.walk(dataset_path)):
        # print(f"{filesnames}\n")
        for filename in filesnames:
            semantic_label = filename.split(".")[0][-2]
            data["labels"].append(semantic_label)


    #mappings of labels      
    data["mapping"] = list(set(data["labels"]))
    data["mapping"].sort()
    print(data["mapping"])

    for i, label in enumerate(data["mapping"]):
        if label == 'N':
            data["mapping"][i] = 'Neutral'
        if label == 'F':
            data["mapping"][i] = 'Happiness'
        if label == 'A':
            data["mapping"][i] = 'Anxiety/Fear'
        if label == 'L':
            data["mapping"][i] = 'Boredom'
        if label == 'E':
            data["mapping"][i] = 'Disgust'
        if label == 'T':
            data["mapping"][i] = 'Sadness'
        if label == 'W':
            data["mapping"][i] = 'Anger'

    print(data["labels"])
    #converting labels to integer
    for i,label in enumerate(data["labels"]):
        if label == 'A':
            data["labels"][i] = 0
        if label == 'E':
            data["labels"][i] = 1
        if label == 'F':
            data["labels"][i] = 2
        if label == 'L':
            data["labels"][i] = 3
        if label == 'N':
            data["labels"][i] = 4
        if label == 'T':
            data["labels"][i] = 5
        if label == 'W':
            data["labels"][i] = 6
    print(data["mapping"])
    print(data["labels"])

    labels = data["labels"].copy()
    data["labels"] = []
    for i,f in enumerate(filesnames):

        #load audio file
        file_path = os.path.join(dirpath, f)
        signal, sample_rate = librosa.load(file_path, sr= SAMPLE_RATE)

        #processing each segments of audio file
        for d in range(num_segment):

            #calculate the start and end of sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment

            #extract mfcc
            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate,n_mfcc=num_mfcc,n_fft=n_fft, hop_length = hop_length)
            mfcc = mfcc.T

            #store only if mfcc feature with expected number of vectors
            if len(mfcc) == num_mfcc_vectors_per_segment :
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(labels[i])
                # print(f"{file_path} \n segment : {d+1}")
    # print(data["mfcc"][1:5])

    #storing data to json file
    with open("GermanEmoDB.json","w") as db:
        json.dump(data, db, indent=4)
    print(len(data["labels"]))
    print(data["labels"][0:20])

if __name__ == "__main__":
    save_mfcc(DATASET_PATH,JSON_PATH, num_segment=10)