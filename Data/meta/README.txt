# ESC-US: Dataset for Environmental Sound Classification
# https://dx.doi.org/10.7910/DVN/YDEPUT

The ESC-US dataset is a supplementary unlabeled part of the ESC dataset consisting of
250 000 environmental recordings (5-second-long clips) suitable for unsupervised pre-training,
clustering and manifold learning experiments.

The dataset, although not hand-annotated, includes the labels (tags) submitted by the original
uploading users. This way the dataset could be also potentially used in weakly supervised
(noisy and/or missing labels) learning experiments.

## File naming scheme

ID-Freesound_file_ID-slice_start_time-slice_end_time.ogg

ID - ESC-US file identifier
Freesound_file_ID - ID of the original Freesound recording from which the clip has been extracted
slice_start_time / slice_end_time - start/end time for the extracted part

The dataset recordings are arranged in 25 directories, 10 000 clips each, available as
separate compressed files.

## File details

5-second-long recordings reconverted to a unified format:
- 44100 Hz,
- single channel (monophonic),
- Vorbis/Ogg compression @ 192 kbit/s. 

## License

The dataset is available under the terms of the Creative Commons
Attribution-NonCommercial license (http://creativecommons.org/licenses/by-nc/3.0/).

## Attribution / metadata

The ESC-US.csv file lists the details of the original Freesound recordings used to create the dataset.