"""HECKTOR 2022 preprocessor."""
import csv
from typing import Any, Dict, List, Optional, Tuple
from etils import epath
import numpy as np
import torchio as tio

import terminology as term

_MAX_HU = 1024
_MIN_HU = -1024


class MultiModalProcessor():
    """This class is used to preprocess multi-modal images.
    The data folder must follow the following structure:
    data_folder
    |-- *_in_train.csv: list of ids in train
    |-- *_in_valid.csv: list of ids in validation
    |-- *_in_test.csv: list of ids in test
    |-- images
        |-- patientID__CT.nii.gz (Note double _ here)
            ...
    |-- labels
        |-- patientID.nii.gz

    The files must have modality at the end of the filenames,
    for example: patient_01_PT.nii.gz
    """

    def __init__(self, data_folder: epath.Path,
                 phase: term.Phase,
                 modalities: List[term.Modality],
                 reference: term.Modality,
                 ) -> None:
        self.data_path = data_folder
        self.phase = phase
        self.modalities = modalities
        self.reference = reference

    def get_patient_lists(self) -> List[str]:
        """Find the patients in a specific csv file with corresponding phase."""
        csv_files = self.data_path.glob(f'*_in_{self.phase.value}.csv')
        csv_file = [x for x in csv_files]
        if len(csv_file) > 1:
            raise ValueError(f'More than 1 file is found: {csv_file}')
        if len(csv_file) == 0:
            raise ValueError(f'No file is found here {csv_files}')
        print(f'Find patient list file: {csv_file}')
        csv_file = csv_file[0]
        with open(csv_file, 'r') as f:
            patients = list(csv.reader(f, delimiter=','))
        patients = patients[0]
        patients = [patient for patient in patients]
        print(f'Find {len(patients)} patients for Phase: {self.phase.value}')
        return patients

    def create_subject(self, patient: str) -> tio.Subject:
        """Create subject for a patient."""
        subject_dict = {'ID': patient}
        for modality in self.modalities:
            img_file = self.data_path / 'images' / (patient + f'__{modality.value}.nii.gz')
            img = tio.ScalarImage(img_file)
            subject_dict[modality.value] = img
        label_file = self.data_path / 'labels' / (patient + '.nii.gz')
        label = tio.ScalarImage(label_file)
        subject_dict['LABEL'] = label
        subject = tio.Subject(subject_dict)
        return subject

    def create_subject_list(self) -> List[tio.Subject]:
        """Create subject list for all patients."""
        patients = self.get_patient_lists()
        subjects = []
        for patient in patients:
            subjects.append(self.create_subject(patient))
        return subjects

    def check_resolution(self, subjects: List[tio.Subject]) -> None:
        """Print out the data size and resolution for patients."""
        for subject in subjects:
            print(f'Patient: {subject.ID}')
            for modality in self.modalities:
                scalar_img = subject[modality.value]
                print(f'\t {modality.value}: Shape - {scalar_img.shape}; Spacing - {scalar_img.spacing}.')

    def resample_to_reference(self, subject: tio.Subject, xy_size: Tuple[int, int]) -> tio.Subject:
        """Assume all the volumes are already co-registered."""
        print(f'Resample {subject.ID}')
        ref_img = subject[self.reference.value]
        image_size = (xy_size[0], xy_size[1], ref_img.shape[-1])
        resize_for_ref = tio.transforms.Resize(image_size)
        ref_img = resize_for_ref(ref_img)
        resample_transform = tio.transforms.Resample(target=ref_img)
        return tio.Subject(resample_transform(subject))

    def train_histogram_standardization(
            self, modality: term.Modality) -> np.ndarray:
        """Train histogram standardization for normalization purpose.
        Only need to run once for each dataset.
        Args:
            modality: one modality
            output: path to save landmarks
        Return:
        """
        output_file = self.data_path / f'{modality.value}_landmarks.npy'
        if output_file.exists():  # no need to retrain
            return np.load(output_file)
        img_path = self.data_path / 'images'
        img_files = img_path.glob(f'*__{modality.value}.nii.gz')
        img_files = [f for f in img_files]
        landmarks = tio.HistogramStandardization.train(
            img_files, output_path=output_file)  # type: ignore
        return landmarks

    def create_landmark_dict(self) -> Dict[str, np.ndarray]:
        """Create landmarks for all modalities except for CT."""
        landmarks_dict = {}
        for modality in self.modalities:
            if modality != term.Modality.CT:
                landmarks = self.train_histogram_standardization(modality)
                landmarks_dict[modality.value] = landmarks
        return landmarks_dict

    def create_transformation(self, transform_keys: Dict[str, Dict[str, Any]]
                              ) -> tio.transforms.Transform:
        """Create transformation for data preprocessing and augmentation."""
        transform_collection = []
        # First image normalization.
        landmarks_dict = self.create_landmark_dict()
        for modality in self.modalities:
            if modality == term.Modality.CT:  # CT only
                transform_collection.append(
                    tio.transforms.Clamp(out_min=_MIN_HU, out_max=_MAX_HU,
                                         include=[term.Modality.CT.value]),
                )
                transform_collection.append(
                    tio.transforms.RescaleIntensity(out_min_max=(-1, 1),
                                                    include=[term.Modality.CT.value]),
                )
            else:
                transform_collection.append(tio.transforms.HistogramStandardization(
                    landmarks_dict, exclude=[term.Modality.CT.value, 'LABEL']))
                transform_collection.append(tio.transforms.ZNormalization(
                    masking_method=tio.ZNormalization.mean, exclude=[term.Modality.CT.value, 'LABEL']))

        # Second Augmentation
        for key, value in transform_keys.items():
            if 'affine' == key:
                degrees = value['degrees']
                translation = value['translation']
                transform_collection.append(
                    tio.transforms.RandomAffine(degrees=degrees, translation=translation))
            elif 'flip' == key:
                p = value['p']
                axes = value['axes']
                transform_collection.append(tio.transforms.RandomFlip(axes=axes, p=p))
            # elif 'blur' == key:
            #     transform_collection.append(tio.transforms.GaussianBlur((0.01, 0.5)))
            else:
                raise ValueError(f'Unsupported transformation: {key}')
        transform_comp = tio.transforms.Compose(transform_collection)
        return transform_comp


