## Dataset 
We have extracted the mel spectrogram from audio files and converted them into a pytorch dataset using the prepare_dataset.py

 1) data_path = "dataset/dataset/"   ( directory containing audio files )
 2) gender_file = "dataset/voice.tsv" ( audio filename mapping to gender )

To prepare dataset , change the dataset path in the prepare_dataset.py and run it to generate a dataset.bin

## Run 
Run the train.ipynb to see the results , after generating the **dataset.bin** .  

## Model 
To Change the model , edit the AudioClassifier class in cnn_model.py 

**Note :** We have not included the dataset as , it larger than 5 GB . Instead , we have ipynb with results in it . You can further train on our data by the method mentioned in the Dataset Section . 