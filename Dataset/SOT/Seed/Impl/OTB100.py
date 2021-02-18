import os
from Dataset.Type.data_split import DataSplit
import re
from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor
from Miscellaneous.platform_style_path import get_platform_style_path


def _get_sequence_info_list(split: str):
    if split == 'otb100':
        sequence_names = ["Basketball", "Biker", "Bird1", "Bird2", "BlurBody", "BlurCar1", "BlurCar2", "BlurCar3",
                          "BlurCar4", "BlurFace", "BlurOwl", "Board", "Bolt", "Bolt2", "Box", "Boy", "Car1", "Car2",
                          "Car24", "Car4", "CarDark", "CarScale", "ClifBar", "Coke", "Couple", "Coupon", "Crossing",
                          "Crowds", "Dancer", "Dancer2", "David", "David2", "David3", "Deer", "Diving", "Dog", "Dog1",
                          "Doll", "DragonBaby", "Dudek", "FaceOcc1", "FaceOcc2", "Fish", "FleetFace", "Football",
                          "Football1", "Freeman1", "Freeman3", "Freeman4", "Girl", "Girl2", "Gym", "Human2", "Human3",
                          "Human4_2", "Human5", "Human6", "Human7", "Human8", "Human9", "Ironman", "Jogging_1",
                          "Jogging_2", "Jump", "Jumping", "KiteSurf", "Lemming", "Liquor", "Man", "Matrix", "Mhyang",
                          "MotorRolling", "MountainBike", "Panda", "RedTeam", "Rubik", "Shaking", "Singer1", "Singer2",
                          "Skater", "Skater2", "Skating1", "Skating2_1", "Skating2_2", "Skiing", "Soccer", "Subway",
                          "Surfer", "Suv", "Sylvester", "Tiger1", "Tiger2", "Toy", "Trans", "Trellis", "Twinnings",
                          "Vase", "Walking", "Walking2", "Woman"]
    elif split == 'otb2013':
        # from http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html tracker_benchmark_v1.0.zip/util/configSeqs.m
        sequence_names = ["Basketball", "Bolt", "Boy", "Car4", "CarDark", "CarScale", "Coke", "Couple", "Crossing",
                          "David", "David2", "David3", "Deer", "Dog1", "Doll", "Dudek", "FaceOcc1", "FaceOcc2", "Fish",
                          "FleetFace", "Football", "Football1", "Freeman1", "Freeman3", "Freeman4", "Girl", "Ironman",
                          "Jogging_1", "Jogging_2", "Jumping", "Lemming", "Liquor", "Matrix", "Mhyang", "MotorRolling",
                          "MountainBike", "Shaking", "Singer1", "Singer2", "Skating1", "Skiing", "Soccer", "Subway",
                          "Suv", "Sylvester", "Tiger1", "Tiger2", "Trellis", "Walking", "Walking2", "Woman"]
    else:
        raise Exception('unknown split')

    sequence_info_list = {
        "Basketball": {"path": "Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4, "ext": "jpg",
                       "anno_path": "Basketball/groundtruth_rect.txt",
                       "object_class": "person"},
        "Biker": {"path": "Biker/img", "startFrame": 1, "endFrame": 142, "nz": 4, "ext": "jpg",
                  "anno_path": "Biker/groundtruth_rect.txt",
                  "object_class": "person head"},
        "Bird1": {"path": "Bird1/img", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg",
                  "anno_path": "Bird1/groundtruth_rect.txt",
                  "object_class": "bird"},
        "Bird2": {"path": "Bird2/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg",
                  "anno_path": "Bird2/groundtruth_rect.txt",
                  "object_class": "bird"},
        "BlurBody": {"path": "BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg",
                     "anno_path": "BlurBody/groundtruth_rect.txt",
                     "object_class": "person"},
        "BlurCar1": {"path": "BlurCar1/img", "startFrame": 247, "endFrame": 988, "nz": 4, "ext": "jpg",
                     "anno_path": "BlurCar1/groundtruth_rect.txt",
                     "object_class": "car"},
        "BlurCar2": {"path": "BlurCar2/img", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg",
                     "anno_path": "BlurCar2/groundtruth_rect.txt",
                     "object_class": "car"},
        "BlurCar3": {"path": "BlurCar3/img", "startFrame": 3, "endFrame": 359, "nz": 4, "ext": "jpg",
                     "anno_path": "BlurCar3/groundtruth_rect.txt",
                     "object_class": "car"},
        "BlurCar4": {"path": "BlurCar4/img", "startFrame": 18, "endFrame": 397, "nz": 4, "ext": "jpg",
                     "anno_path": "BlurCar4/groundtruth_rect.txt",
                     "object_class": "car"},
        "BlurFace": {"path": "BlurFace/img", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg",
                     "anno_path": "BlurFace/groundtruth_rect.txt",
                     "object_class": "face"},
        "BlurOwl": {"path": "BlurOwl/img", "startFrame": 1, "endFrame": 631, "nz": 4, "ext": "jpg",
                    "anno_path": "BlurOwl/groundtruth_rect.txt",
                    "object_class": "other"},
        "Board": {"path": "Board/img", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg",
                  "anno_path": "Board/groundtruth_rect.txt",
                  "object_class": "other"},
        "Bolt": {"path": "Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
                 "anno_path": "Bolt/groundtruth_rect.txt",
                 "object_class": "person"},
        "Bolt2": {"path": "Bolt2/img", "startFrame": 1, "endFrame": 293, "nz": 4, "ext": "jpg",
                  "anno_path": "Bolt2/groundtruth_rect.txt",
                  "object_class": "person"},
        "Box": {"path": "Box/img", "startFrame": 1, "endFrame": 1161, "nz": 4, "ext": "jpg",
                "anno_path": "Box/groundtruth_rect.txt",
                "object_class": "other"},
        "Boy": {"path": "Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg",
                "anno_path": "Boy/groundtruth_rect.txt",
                "object_class": "face"},
        "Car1": {"path": "Car1/img", "startFrame": 1, "endFrame": 1020, "nz": 4, "ext": "jpg",
                 "anno_path": "Car1/groundtruth_rect.txt",
                 "object_class": "car"},
        "Car2": {"path": "Car2/img", "startFrame": 1, "endFrame": 913, "nz": 4, "ext": "jpg",
                 "anno_path": "Car2/groundtruth_rect.txt",
                 "object_class": "car"},
        "Car24": {"path": "Car24/img", "startFrame": 1, "endFrame": 3059, "nz": 4, "ext": "jpg",
                  "anno_path": "Car24/groundtruth_rect.txt",
                  "object_class": "car"},
        "Car4": {"path": "Car4/img", "startFrame": 1, "endFrame": 659, "nz": 4, "ext": "jpg",
                 "anno_path": "Car4/groundtruth_rect.txt",
                 "object_class": "car"},
        "CarDark": {"path": "CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg",
                    "anno_path": "CarDark/groundtruth_rect.txt",
                    "object_class": "car"},
        "CarScale": {"path": "CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
                     "anno_path": "CarScale/groundtruth_rect.txt",
                     "object_class": "car"},
        "ClifBar": {"path": "ClifBar/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg",
                    "anno_path": "ClifBar/groundtruth_rect.txt",
                    "object_class": "other"},
        "Coke": {"path": "Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg",
                 "anno_path": "Coke/groundtruth_rect.txt",
                 "object_class": "other"},
        "Couple": {"path": "Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg",
                   "anno_path": "Couple/groundtruth_rect.txt",
                   "object_class": "person"},
        "Coupon": {"path": "Coupon/img", "startFrame": 1, "endFrame": 327, "nz": 4, "ext": "jpg",
                   "anno_path": "Coupon/groundtruth_rect.txt",
                   "object_class": "other"},
        "Crossing": {"path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4, "ext": "jpg",
                     "anno_path": "Crossing/groundtruth_rect.txt",
                     "object_class": "person"},
        "Crowds": {"path": "Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg",
                   "anno_path": "Crowds/groundtruth_rect.txt",
                   "object_class": "person"},
        "Dancer": {"path": "Dancer/img", "startFrame": 1, "endFrame": 225, "nz": 4, "ext": "jpg",
                   "anno_path": "Dancer/groundtruth_rect.txt",
                   "object_class": "person"},
        "Dancer2": {"path": "Dancer2/img", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg",
                    "anno_path": "Dancer2/groundtruth_rect.txt",
                    "object_class": "person"},
        "David": {"path": "David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg",
                  "anno_path": "David/groundtruth_rect.txt",
                  "object_class": "face"},
        "David2": {"path": "David2/img", "startFrame": 1, "endFrame": 537, "nz": 4, "ext": "jpg",
                   "anno_path": "David2/groundtruth_rect.txt",
                   "object_class": "face"},
        "David3": {"path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
                   "anno_path": "David3/groundtruth_rect.txt",
                   "object_class": "person"},
        "Deer": {"path": "Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg",
                 "anno_path": "Deer/groundtruth_rect.txt",
                 "object_class": "mammal"},
        "Diving": {"path": "Diving/img", "startFrame": 1, "endFrame": 215, "nz": 4, "ext": "jpg",
                   "anno_path": "Diving/groundtruth_rect.txt",
                   "object_class": "person"},
        "Dog": {"path": "Dog/img", "startFrame": 1, "endFrame": 127, "nz": 4, "ext": "jpg",
                "anno_path": "Dog/groundtruth_rect.txt",
                "object_class": "dog"},
        "Dog1": {"path": "Dog1/img", "startFrame": 1, "endFrame": 1350, "nz": 4, "ext": "jpg",
                 "anno_path": "Dog1/groundtruth_rect.txt",
                 "object_class": "dog"},
        "Doll": {"path": "Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg",
                 "anno_path": "Doll/groundtruth_rect.txt",
                 "object_class": "other"},
        "DragonBaby": {"path": "DragonBaby/img", "startFrame": 1, "endFrame": 113, "nz": 4, "ext": "jpg",
                       "anno_path": "DragonBaby/groundtruth_rect.txt",
                       "object_class": "face"},
        "Dudek": {"path": "Dudek/img", "startFrame": 1, "endFrame": 1145, "nz": 4, "ext": "jpg",
                  "anno_path": "Dudek/groundtruth_rect.txt",
                  "object_class": "face"},
        "FaceOcc1": {"path": "FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4, "ext": "jpg",
                     "anno_path": "FaceOcc1/groundtruth_rect.txt",
                     "object_class": "face"},
        "FaceOcc2": {"path": "FaceOcc2/img", "startFrame": 1, "endFrame": 812, "nz": 4, "ext": "jpg",
                     "anno_path": "FaceOcc2/groundtruth_rect.txt",
                     "object_class": "face"},
        "Fish": {"path": "Fish/img", "startFrame": 1, "endFrame": 476, "nz": 4, "ext": "jpg",
                 "anno_path": "Fish/groundtruth_rect.txt",
                 "object_class": "other"},
        "FleetFace": {"path": "FleetFace/img", "startFrame": 1, "endFrame": 707, "nz": 4, "ext": "jpg",
                      "anno_path": "FleetFace/groundtruth_rect.txt",
                      "object_class": "face"},
        "Football": {"path": "Football/img", "startFrame": 1, "endFrame": 362, "nz": 4, "ext": "jpg",
                     "anno_path": "Football/groundtruth_rect.txt",
                     "object_class": "person head"},
        "Football1": {"path": "Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4, "ext": "jpg",
                      "anno_path": "Football1/groundtruth_rect.txt",
                      "object_class": "face"},
        "Freeman1": {"path": "Freeman1/img", "startFrame": 1, "endFrame": 326, "nz": 4, "ext": "jpg",
                     "anno_path": "Freeman1/groundtruth_rect.txt",
                     "object_class": "face"},
        "Freeman3": {"path": "Freeman3/img", "startFrame": 1, "endFrame": 460, "nz": 4, "ext": "jpg",
                     "anno_path": "Freeman3/groundtruth_rect.txt",
                     "object_class": "face"},
        "Freeman4": {"path": "Freeman4/img", "startFrame": 1, "endFrame": 283, "nz": 4, "ext": "jpg",
                     "anno_path": "Freeman4/groundtruth_rect.txt",
                     "object_class": "face"},
        "Girl": {"path": "Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
                 "anno_path": "Girl/groundtruth_rect.txt",
                 "object_class": "face"},
        "Girl2": {"path": "Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg",
                  "anno_path": "Girl2/groundtruth_rect.txt",
                  "object_class": "person"},
        "Gym": {"path": "Gym/img", "startFrame": 1, "endFrame": 767, "nz": 4, "ext": "jpg",
                "anno_path": "Gym/groundtruth_rect.txt",
                "object_class": "person"},
        "Human2": {"path": "Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg",
                   "anno_path": "Human2/groundtruth_rect.txt",
                   "object_class": "person"},
        "Human3": {"path": "Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg",
                   "anno_path": "Human3/groundtruth_rect.txt",
                   "object_class": "person"},
        "Human4_2": {"path": "Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg",
                     "anno_path": "Human4/groundtruth_rect.2.txt",
                     "object_class": "person"},
        "Human5": {"path": "Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg",
                   "anno_path": "Human5/groundtruth_rect.txt",
                   "object_class": "person"},
        "Human6": {"path": "Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg",
                   "anno_path": "Human6/groundtruth_rect.txt",
                   "object_class": "person"},
        "Human7": {"path": "Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg",
                   "anno_path": "Human7/groundtruth_rect.txt",
                   "object_class": "person"},
        "Human8": {"path": "Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg",
                   "anno_path": "Human8/groundtruth_rect.txt",
                   "object_class": "person"},
        "Human9": {"path": "Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg",
                   "anno_path": "Human9/groundtruth_rect.txt",
                   "object_class": "person"},
        "Ironman": {"path": "Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg",
                    "anno_path": "Ironman/groundtruth_rect.txt",
                    "object_class": "person head"},
        "Jogging_1": {"path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg",
                      "anno_path": "Jogging/groundtruth_rect.1.txt",
                      "object_class": "person"},
        "Jogging_2": {"path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg",
                      "anno_path": "Jogging/groundtruth_rect.2.txt",
                      "object_class": "person"},
        "Jump": {"path": "Jump/img", "startFrame": 1, "endFrame": 122, "nz": 4, "ext": "jpg",
                 "anno_path": "Jump/groundtruth_rect.txt",
                 "object_class": "person"},
        "Jumping": {"path": "Jumping/img", "startFrame": 1, "endFrame": 313, "nz": 4, "ext": "jpg",
                    "anno_path": "Jumping/groundtruth_rect.txt",
                    "object_class": "face"},
        "KiteSurf": {"path": "KiteSurf/img", "startFrame": 1, "endFrame": 84, "nz": 4, "ext": "jpg",
                     "anno_path": "KiteSurf/groundtruth_rect.txt",
                     "object_class": "face"},
        "Lemming": {"path": "Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg",
                    "anno_path": "Lemming/groundtruth_rect.txt",
                    "object_class": "other"},
        "Liquor": {"path": "Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg",
                   "anno_path": "Liquor/groundtruth_rect.txt",
                   "object_class": "other"},
        "Man": {"path": "Man/img", "startFrame": 1, "endFrame": 134, "nz": 4, "ext": "jpg",
                "anno_path": "Man/groundtruth_rect.txt",
                "object_class": "face"},
        "Matrix": {"path": "Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg",
                   "anno_path": "Matrix/groundtruth_rect.txt",
                   "object_class": "person head"},
        "Mhyang": {"path": "Mhyang/img", "startFrame": 1, "endFrame": 1490, "nz": 4, "ext": "jpg",
                   "anno_path": "Mhyang/groundtruth_rect.txt",
                   "object_class": "face"},
        "MotorRolling": {"path": "MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4, "ext": "jpg",
                         "anno_path": "MotorRolling/groundtruth_rect.txt",
                         "object_class": "vehicle"},
        "MountainBike": {"path": "MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4, "ext": "jpg",
                         "anno_path": "MountainBike/groundtruth_rect.txt",
                         "object_class": "bicycle"},
        "Panda": {"path": "Panda/img", "startFrame": 1, "endFrame": 1000, "nz": 4, "ext": "jpg",
                  "anno_path": "Panda/groundtruth_rect.txt",
                  "object_class": "mammal"},
        "RedTeam": {"path": "RedTeam/img", "startFrame": 1, "endFrame": 1918, "nz": 4, "ext": "jpg",
                    "anno_path": "RedTeam/groundtruth_rect.txt",
                    "object_class": "vehicle"},
        "Rubik": {"path": "Rubik/img", "startFrame": 1, "endFrame": 1997, "nz": 4, "ext": "jpg",
                  "anno_path": "Rubik/groundtruth_rect.txt",
                  "object_class": "other"},
        "Shaking": {"path": "Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
                    "anno_path": "Shaking/groundtruth_rect.txt",
                    "object_class": "face"},
        "Singer1": {"path": "Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg",
                    "anno_path": "Singer1/groundtruth_rect.txt",
                    "object_class": "person"},
        "Singer2": {"path": "Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg",
                    "anno_path": "Singer2/groundtruth_rect.txt",
                    "object_class": "person"},
        "Skater": {"path": "Skater/img", "startFrame": 1, "endFrame": 160, "nz": 4, "ext": "jpg",
                   "anno_path": "Skater/groundtruth_rect.txt",
                   "object_class": "person"},
        "Skater2": {"path": "Skater2/img", "startFrame": 1, "endFrame": 435, "nz": 4, "ext": "jpg",
                    "anno_path": "Skater2/groundtruth_rect.txt",
                    "object_class": "person"},
        "Skating1": {"path": "Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4, "ext": "jpg",
                     "anno_path": "Skating1/groundtruth_rect.txt",
                     "object_class": "person"},
        "Skating2_1": {"path": "Skating2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg",
                       "anno_path": "Skating2/groundtruth_rect.1.txt",
                       "object_class": "person"},
        "Skating2_2": {"path": "Skating2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg",
                       "anno_path": "Skating2/groundtruth_rect.2.txt",
                       "object_class": "person"},
        "Skiing": {"path": "Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg",
                   "anno_path": "Skiing/groundtruth_rect.txt",
                   "object_class": "person"},
        "Soccer": {"path": "Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg",
                   "anno_path": "Soccer/groundtruth_rect.txt",
                   "object_class": "face"},
        "Subway": {"path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg",
                   "anno_path": "Subway/groundtruth_rect.txt",
                   "object_class": "person"},
        "Surfer": {"path": "Surfer/img", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "jpg",
                   "anno_path": "Surfer/groundtruth_rect.txt",
                   "object_class": "person head"},
        "Suv": {"path": "Suv/img", "startFrame": 1, "endFrame": 945, "nz": 4, "ext": "jpg",
                "anno_path": "Suv/groundtruth_rect.txt",
                "object_class": "car"},
        "Sylvester": {"path": "Sylvester/img", "startFrame": 1, "endFrame": 1345, "nz": 4, "ext": "jpg",
                      "anno_path": "Sylvester/groundtruth_rect.txt",
                      "object_class": "other"},
        "Tiger1": {"path": "Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg",
                   "anno_path": "Tiger1/groundtruth_rect.txt", "initOmit": 5,
                   "object_class": "other"},
        "Tiger2": {"path": "Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
                   "anno_path": "Tiger2/groundtruth_rect.txt",
                   "object_class": "other"},
        "Toy": {"path": "Toy/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
                "anno_path": "Toy/groundtruth_rect.txt",
                "object_class": "other"},
        "Trans": {"path": "Trans/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg",
                  "anno_path": "Trans/groundtruth_rect.txt",
                  "object_class": "other"},
        "Trellis": {"path": "Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg",
                    "anno_path": "Trellis/groundtruth_rect.txt",
                    "object_class": "face"},
        "Twinnings": {"path": "Twinnings/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg",
                      "anno_path": "Twinnings/groundtruth_rect.txt",
                      "object_class": "other"},
        "Vase": {"path": "Vase/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
                 "anno_path": "Vase/groundtruth_rect.txt",
                 "object_class": "other"},
        "Walking": {"path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg",
                    "anno_path": "Walking/groundtruth_rect.txt",
                    "object_class": "person"},
        "Walking2": {"path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
                     "anno_path": "Walking2/groundtruth_rect.txt",
                     "object_class": "person"},
        "Woman": {"path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg",
                  "anno_path": "Woman/groundtruth_rect.txt",
                  "object_class": "person"}
    }

    return dict((k, sequence_info_list[k]) for k in sequence_names)


def construct_OTB100(constructor: SingleObjectTrackingDatasetConstructor, seed, split):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    sequence_info_list = _get_sequence_info_list(split)

    spacer = re.compile(r'[\s,]')
    constructor.set_total_number_of_sequences(len(sequence_info_list))
    for sequence_name, sequence_info in sequence_info_list.items():
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        sequence_path = get_platform_style_path(os.path.join(root_path, sequence_path))

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            init_omit = 0
            if 'initOmit' in sequence_info:
                init_omit = sequence_info['initOmit']

            anno_path = get_platform_style_path('{}/{}'.format(root_path, sequence_info['anno_path']))

            images = os.listdir(sequence_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()

            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_path, image))
            actual_start_index = images[0]
            actual_start_index = int(actual_start_index[0: actual_start_index.find('.')])
            offset = start_frame - actual_start_index

            for index_line, line in enumerate(open(os.path.join(anno_path), 'r')):
                if index_line < init_omit:
                    continue

                index = offset + index_line
                line = line.strip()
                if len(line) == 0:
                    continue
                bounding_box = [int(value) for value in spacer.split(line) if value]

                with sequence_constructor.open_frame(index) as frame_constructor:
                    frame_constructor.set_bounding_box(bounding_box)
            assert end_frame - start_frame == index_line
