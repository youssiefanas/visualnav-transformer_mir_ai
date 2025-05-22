import os
import cv2
import csv
import argparse
import numpy as np
import pandas as pd
import pickle
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R

def extract_single_images(bag_path, image_topic, output_dir, csv_file, target_size):
    rclpy.init()
    bridge = CvBridge()
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_filename', 'timestamp'])

        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        topics = reader.get_all_topics_and_types()
        topic_types = {t.name: t.type for t in topics}
        if image_topic not in topic_types:
            return

        filter = StorageFilter(topics=[image_topic])
        reader.set_filter(filter)

        count = 0
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == image_topic:
                msg = deserialize_message(data, Image)
                try:
                    timestamp_sec = msg.header.stamp.sec
                    timestamp_nsec = msg.header.stamp.nanosec
                    timestamp_str = f"{timestamp_sec}.{timestamp_nsec}"

                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    resized = cv2.resize(cv_image, target_size)

                    image_filename = os.path.join(output_dir, f"{count}.jpg")
                    cv2.imwrite(image_filename, resized)

                    writer.writerow([f"{count}.jpg", timestamp_str])
                    count += 1
                except:
                    pass

def extract_stacked_images(bag_path, output_dir, csv_file, image_topics, target_size, tolerance):
    rclpy.init()
    bridge = CvBridge()
    os.makedirs(output_dir, exist_ok=True)

    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics}
    msg_types = {topic: get_message(type_map[topic]) for topic in image_topics}

    topic_msgs = {topic: [] for topic in image_topics}
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in image_topics:
            msg = deserialize_message(data, msg_types[topic])
            topic_msgs[topic].append((t / 1e9, msg))

    ref_times = [t for t, _ in topic_msgs[image_topics[0]]]
    cam0_msgs = topic_msgs[image_topics[0]]
    cam3_msgs = topic_msgs[image_topics[1]]
    cam4_msgs = topic_msgs[image_topics[2]]

    def find_closest(target_time, msg_list):
        return min(msg_list, key=lambda x: abs(x[0] - target_time))

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "timestamp"])

        for idx, (t0, msg0) in enumerate(cam0_msgs):
            t3, msg3 = find_closest(t0, cam3_msgs)
            t4, msg4 = find_closest(t0, cam4_msgs)

            if abs(t3 - t0) > tolerance or abs(t4 - t0) > tolerance:
                continue

            try:
                img0 = bridge.imgmsg_to_cv2(msg0, desired_encoding='passthrough')
                img3 = bridge.imgmsg_to_cv2(msg3, desired_encoding='passthrough')
                img4 = bridge.imgmsg_to_cv2(msg4, desired_encoding='passthrough')

                img0 = cv2.resize(img0, target_size)
                img3 = cv2.resize(img3, target_size)
                img4 = cv2.resize(img4, target_size)

                if len(img0.shape) > 2: img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                if len(img3.shape) > 2: img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                if len(img4.shape) > 2: img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

                stacked_img = np.stack([img0, img3, img4], axis=-1)
                out_path = os.path.join(output_dir, f"{idx}.jpg")
                cv2.imwrite(out_path, stacked_img)
                writer.writerow([idx, f"{t0:.9f}"])
            except:
                pass

def convert_tum_to_csv(tum_file, output_csv):
    with open(tum_file, 'r') as tum:
        lines = [line.strip() for line in tum if line.strip() and not line.startswith('#')]

    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        for line in lines:
            parts = line.split()
            if len(parts) == 8:
                writer.writerow(parts)

def match_poses(image_csv, pose_csv, output_pkl):
    pose_df = pd.read_csv(pose_csv)
    image_df = pd.read_csv(image_csv)

    positions = []
    yaws = []

    for _, image_row in image_df.iterrows():
        image_timestamp = image_row['timestamp']
        pose_df['time_diff'] = abs(pose_df['timestamp'] - image_timestamp)
        closest_pose_row = pose_df.loc[pose_df['time_diff'].idxmin()]

        x, y = closest_pose_row['tx'], closest_pose_row['ty']
        positions.append([x, y])

        quat = [closest_pose_row['qx'], closest_pose_row['qy'], closest_pose_row['qz'], closest_pose_row['qw']]
        yaw = R.from_quat(quat).as_euler('zyx', degrees=False)[0]
        yaws.append(np.array([yaw]))

    data = {'position': np.array(positions, dtype=object), 'yaw': np.array(yaws, dtype=object)}
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_path", type=str, required=True)
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--tum", type=str, required=True)
    args = parser.parse_args()

    parent_path = args.parent_path
    bag_path = args.bag
    tum_file = args.tum

    image_single_dir = os.path.join(parent_path, "images")
    image_stacked_dir = os.path.join(parent_path, "images_stacked")
    csv_single = os.path.join(image_single_dir, "timestamps.csv")
    csv_stacked = os.path.join(image_stacked_dir, "timestamps.csv")
    trajectory_csv = os.path.join(parent_path, "trajectory.csv")
    pkl_output = os.path.join(parent_path, "traj_data.pkl")

    extract_single_images(bag_path, "/alphasense_driver_ros/cam0", image_single_dir, csv_single, (96, 96))
    extract_stacked_images(bag_path, image_stacked_dir, csv_stacked, [
        "/alphasense_driver_ros/cam0",
        "/alphasense_driver_ros/cam3",
        "/alphasense_driver_ros/cam4"
    ], (96, 96), 0.03)
    convert_tum_to_csv(tum_file, trajectory_csv)
    match_poses(csv_stacked, trajectory_csv, pkl_output)

if __name__ == "__main__":
    main()