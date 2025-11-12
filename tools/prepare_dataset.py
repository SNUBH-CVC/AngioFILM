import logging
import multiprocessing as mp
import traceback
from functools import partial
from pathlib import Path

import neurokit2 as nk
import numpy as np
from snubhcvc.io.dicom import BaseDicomHandler
from snubhcvc.transforms.normalize import min_max_normalization

# Define directories
num_processes = 10
data_dir = Path("/path/to/data/directory/")
output_dir = Path("/path/to/output/directory/")
output_dir.mkdir(parents=True, exist_ok=True)
error_file = output_dir.parent / "failed_files.txt"

# Configure logging to record exceptions
logging.basicConfig(
    filename=output_dir.parent / "errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Read the persistent error list (if available)
def load_failed_keys(error_file_path):
    failed = set()
    if error_file_path.exists():
        with open(error_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    failed.add(line)
    return failed


failed_keys_initial = load_failed_keys(error_file)

# Get list of all .dcm files (as strings for multiprocessing)
dcm_path_list = list(data_dir.glob("**/*.dcm"))
dcm_paths_str = [str(path) for path in dcm_path_list]


def process_dcm(dcm_path_str, failed_keys, error_lock):
    """
    Process a single DICOM file:
      - If the file has been processed successfully before (JSON exists) or it is in the failed list, it will be skipped.
      - Processes the ECG signal, image data, and writes output files.
      - If any error occurs, the file's key is recorded in the persistent error file.
    """
    try:
        dcm_path = Path(dcm_path_str)
        # Determine unique key from file structure
        try:
            rel_dcm_path = dcm_path.relative_to(data_dir)
            # Expecting a structure: patient_id/study_date/other/basename
            patient_id, study_date, _, basename = str(rel_dcm_path).split("/")
            series_no_str = basename.split(".")[0]
            key = f"{patient_id}_{study_date}_{series_no_str}"
        except Exception as e:
            # If key extraction fails, log and return error (no persistent key to record)
            error_msg = (
                f"Error extracting key from {dcm_path}: {e}\n{traceback.format_exc()}"
            )
            logging.error(error_msg)
            return f"Error (key extraction): {dcm_path}"

        # If key is already known to fail, skip processing
        if key in failed_keys:
            return f"Skipped (previous failure): {key}"

        # Set up output paths
        output_path = output_dir / f"{key}.npz"

        # Resume: if output JSON already exists, skip this file.
        if output_path.exists():
            return f"Skipped (already processed): {key}"

        # Load and process the DICOM file
        dcm_handler = BaseDicomHandler(dcm_path)
        fps = dcm_handler.cine_rate
        if fps != 15:
            raise ValueError(f"Unexpected frame rate: {fps}")

        ecg_signal, sampling_rate = dcm_handler.ecg_signal
        ecg_signal = np.array(ecg_signal)

        # Clean the ECG signal; if an error occurs, record the failure.
        try:
            signals, rpeaks = nk.ecg_process(ecg_signal[:, 1], sampling_rate)
        except Exception as e:
            raise Exception(f"ecg_process error: {e}")

        # Process pixel array
        pixel_array = dcm_handler.pixel_array
        if pixel_array.dtype != np.uint8:
            pixel_array = (min_max_normalization(pixel_array) * 255).astype(np.uint8)

        # Save the processed image array
        np.savez_compressed(
            output_path,
            pixel_array=pixel_array,
            ecg_signal=signals["ECG_Clean"].values,
            ecg_quality=signals["ECG_Quality"].values,
        )

        return f"Processed: {key}"

    except Exception as e:
        error_msg = f"Error processing {dcm_path_str} (key: {key}): {e}\n{traceback.format_exc()}"
        logging.error(error_msg)
        # Write the key to the persistent error file using the provided lock to avoid race conditions.
        with error_lock:
            with open(error_file, "a") as f:
                f.write(f"{key}\n")
            # Update the shared failed_keys dictionary
            failed_keys[key] = True
        return f"Error: {key}"


def main():
    # Create a multiprocessing manager for shared data and locks.
    manager = mp.Manager()
    shared_failed_keys = manager.dict({key: True for key in failed_keys_initial})
    error_lock = manager.Lock()

    # Prepare a partial function with shared resources.
    func = partial(process_dcm, failed_keys=shared_failed_keys, error_lock=error_lock)

    # Create a pool of workers; adjust the number of processes as needed.
    with mp.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(func, dcm_paths_str):
            print(result)


if __name__ == "__main__":
    main()
