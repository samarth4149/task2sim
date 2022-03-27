import json
import os
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import time
from typing import List, Dict
from typing import Union

import numpy as np
from tdw.controller import Controller
from tdw.librarian import ModelLibrarian, MaterialLibrarian, HDRISkyboxLibrarian, SceneLibrarian, ModelRecord, \
    HDRISkyboxRecord
from tdw.output_data import Images, ImageSensors, Environments, OutputData, Transforms, Bounds
from tdw.tdw_utils import TDWUtils
from tqdm import tqdm

from generator.generator_params import GenParams

RNG = np.random.RandomState(0)


class ImagePosition:
    """
    Data to stage an image.
    """

    def __init__(self, avatar_position: dict,
                 camera_rotation: dict,
                 object_position: dict,
                 object_rotation: dict):
        """
        :param avatar_position: The position of the avatar as a Vector3.
        :param camera_rotation: The rotation of the avatar as a Quaternion.
        :param object_position: The position of the object as a Vector3.
        :param object_rotation: The rotation of the object as a Quaternion.
        """

        self.avatar_position = avatar_position
        self.camera_rotation = camera_rotation
        self.object_position = object_position
        self.object_rotation = object_rotation

class Environment:
    """
    Environment data for a single environment.
    """

    def __init__(self, envs: Environments = None, e: int = None):
        """
        :param envs: The environments data.
        :param e: The index of this environment.
        """

        if envs:
            self.x, self.y, self.z = envs.get_center(e)
            self.w, self.h, self.l = envs.get_bounds(e)
        else:
            # Some default values for proc gen scene
            self.x, self.y, self.z = 0., 0., 0.
            self.w, self.h, self.l = 10., 4.5, 10.

class Generator(Controller):
    def __init__(self,
                 port=1071,
                 launch_build=False,
                 screen_size=256,
                 output_size=256,
                 library="models_full.json",
                 temp_urls: bool = True,
                 load_path: Union[str, Path] = Path('/mnt/media'),
                 simple_objs: bool = True,
                 skybox_preload: bool = True,
                 dataset_dir: Union[str, Path] = None):
        """
        :param port: The port used to connect to the build.
        :param launch_build: If True, automatically launch the build. Always set this to False on a Linux server.
        :param visual_material_swapping: If true, set random visual materials per frame.
        :param new: If true, clear the list of models that have already been used.
        :param screen_size: The screen size of the build.
        :param output_size: The size of the output images.
        :param hdri: If true, use a random HDRI skybox per frame.
        :param show_objects: If true, show objects.
        :param max_rot_angle: If not None, clamp the rotation to +/- x (max 180) degrees around each axis.
        :param min_dist: minimum distance of object to camera (when fixed_cam = True)
        :param max_dist: maximum distance of object to camera (when fixed_cam = True)
        :param max_height: The percentage of the environment height that is the ceiling for the avatar and object. Must be between 0 and 1.
        :param grayscale_threshold: The grayscale threshold. Higher value = slower FPS, better composition. Must be between 0 and 1.
        :param less_dark: If true, there will be more daylight exterior skyboxes (requires hdri == True)
        :param id_pass: If true, send the _id pass.
        :param no_overwrite: If true, don't overwrite images.
        :param do_zip: If true, zip the directory at the end.
        :param train: Number of train images.
        :param val: Number of val images.
        :param library: The path to the library records file.
        """

        self.screen_size = screen_size
        self.output_size = output_size


        self.substructures: Dict[str, List[dict]] = {}

        self.load_path = load_path
        self.simple_objs = simple_objs

        # Other constants :
        self.max_rot_angle = 180
        self.min_dist = 0.4
        self.max_dist = 1.5
        self.max_height = 1.
        self.max_light_intensity_var = 0.6

        super(Generator, self).__init__(port=port, launch_build=launch_build)
        self.model_librarian = ModelLibrarian(library=library)

        # Get material records
        with open('configs/material_subset.txt', 'r') as f:
            material_names = f.read().splitlines()
        self.material_librarian = MaterialLibrarian("materials_low.json")
        self.materials = [
            record for record in self.material_librarian.records if record.name in material_names]

        self.hdri_skybox_librarian = HDRISkyboxLibrarian()
        self.skyboxes = [
            record for record in self.hdri_skybox_librarian.records
            if record.location != 'interior' and record.sun_elevation >= 120
        ]

        if simple_objs:
            with open('configs/tdw_simple_objs_100.json', 'r') as f: # TODO : add this to configs
                self.objs = json.load(f)
        else:
            # Fetch the WordNet IDs.
            wnids = self.model_librarian.get_model_wnids()
            # Remove any wnids that don't have valid models.
            self.wnids = [w for w in wnids if len(
                [r for r in self.model_librarian.get_all_models_in_wnid(w) if not r.do_not_use]) > 0]

        # Download from pre-signed URLs.
        if temp_urls:
            self.communicate({"$type": "use_pre_signed_urls",
                              "value": True})

        self.envs = self.initialize_scene({'$type':'load_scene', 'scene_name':'ProcGenScene'})
        if skybox_preload:
            print('Preloading skyboxes')
            for record in self.skyboxes:
                self.communicate(self.get_add_hdri_skybox(record.name))

        if len(self.envs) == 0:
            self.envs = [Environment()]
        self.avatar_id = 'a'

        # Create a directory with the current datasets
        self.root_dir = Path(dataset_dir)

        if os.path.exists(self.root_dir):
            print('Warning : Dataset Directory exists. Might be using some datasets previously generated')
        else:
            os.makedirs(self.root_dir)

        self.model_times = OrderedDict()

    def get_existing_params(self):
        gp_list = []
        for fld in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, fld)):
                params = fld.split('_')[1:]
                gp_list.append(GenParams.from_tuple(tuple(int(p) for p in params)))

        return gp_list

    def initialize_scene(self, scene_command, a="a") -> list:
        """
        Initialize the scene.

        :param scene_command: The command to load the scene.
        :param a: The avatar ID.
        :return: The Environments data of the scene.
        """

        # Initialize the scene.
        # Add the avatar.
        commands = [scene_command,
                    {"$type": "create_avatar",
                     "type": "A_Img_Caps_Kinematic",
                     "id": a,
                     "envs": [0]}]
        # Disable physics.
        # Enable jpgs.
        # Set FOV.
        # Set clipping planes.
        # Set AA.
        # Set aperture.
        # Disable vignette.
        commands.extend([{"$type": "simulate_physics",
                          "value": False},
                         {"$type": "set_img_pass_encoding",
                          "value": False},
                         {'$type': 'set_field_of_view',
                          'avatar_id': 'a',
                          'field_of_view': 60},
                         {'$type': 'set_camera_clipping_planes',
                          'avatar_id': 'a',
                          'far': 160,
                          'near': 0.01},
                         {"$type": "set_anti_aliasing",
                          "avatar_id": "a",
                          "mode": "subpixel"},
                         {"$type": "set_aperture",
                          "aperture": 70},
                         {'$type': 'set_vignette',
                          'enabled': False}])

        # If we're using HDRI skyboxes, send additional favorable post-process commands.
        if self.skyboxes is not None:
            commands.extend([{"$type": "set_post_exposure",
                              "post_exposure": 0.6},
                             {"$type": "set_contrast",
                              "contrast": -20},
                             {"$type": "set_saturation",
                              "saturation": 10},
                             {"$type": "set_screen_space_reflections",
                              "enabled": False},
                             {"$type": "set_shadow_strength",
                              "strength": 1.0}])

        # Send the commands.
        self.communicate(commands)

        # Get the environments data.
        env_data = Environments(self.communicate({"$type": "send_environments",
                                                  "frequency": "once"})[0])
        envs = []
        for i in range(env_data.get_num()):
            envs.append(Environment(env_data, i))
        return envs

    def get_add_scene(self, scene_name: str, library: str = "") -> dict:
        """
        Returns a valid add_scene command.

        :param scene_name: The name of the scene.
        :param library: The path to the records file. If left empty, the default library will be selected. See `SceneLibrarian.get_library_filenames()` and `SceneLibrarian.get_default_library()`.

        :return An add_scene command that the controller can then send.
        """

        if self.scene_librarian is None:
            self.scene_librarian = SceneLibrarian(library=library)

        record = self.scene_librarian.get_record(scene_name)
        if self.load_from_disk:
            scene_url = 'file:///mnt/media/tdw_asset_bundles/scenes/{}'.format(scene_name)
        else:
            scene_url = record.get_url()

        return {"$type": "add_scene",
                "name": scene_name,
                "url": scene_url}

    def _get_default_avatar_position(self):
        return np.array([1., 1.5, 1.8])

    @staticmethod
    def sample_spherical(npoints=1, ndim=3) -> np.array:
        vec = RNG.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return np.array([vec[0][0], vec[1][0], vec[2][0]])

    def generate_metadata(self, datadir: Path, gen_params:GenParams) -> None:
        """
        Generate a metadata file for this dataset.

        :param dataset_dir: The dataset directory for images.
        :param scene_name: The scene name.
        """

        data = {"datadir": str(datadir),
                "screen_size": self.screen_size,
                "output_size": self.output_size,
                "max_rot_angle": self.max_rot_angle,
                "min_dist": self.min_dist,
                "max_dist": self.max_dist,
                "max_height": self.max_height,
                "simple_objs": self.simple_objs,
                "gen_params": gen_params,
                "max_lighting_intensity_var" : self.max_light_intensity_var,
                "start": datetime.now().strftime("%H:%M %d.%m.%y")
                }
        with open(os.path.join(datadir, "metadata.txt"), "wt") as f:
            json.dump(data, f, sort_keys=True, indent=4)

    def clamp_max_scale(self, o_id, env):
        resp = self.communicate([{'$type': 'send_bounds',
                                  'ids': [o_id],
                                  'frequency': 'once'}])
        bds = Bounds(resp[-2])
        top = np.array(bds.get_top(0))
        bottom = np.array(bds.get_bottom(0))
        front = np.array(bds.get_front(0))
        back = np.array(bds.get_back(0))
        left = np.array(bds.get_left(0))
        right = np.array(bds.get_right(0))

        # See if the bounds are bigger than half the environment
        x_ratio = 2 * max(np.abs(front - back))/env.w
        y_ratio = max(np.abs(top - bottom))/(0.9 * env.h)
        z_ratio = 2 * max(np.abs(left - right))/env.l

        max_ratio = max(x_ratio, y_ratio, z_ratio)
        if max_ratio > 1:
            self.communicate({"$type": "scale_object",
                              "id": o_id,
                              "scale_factor": {"x": 1/max_ratio, "y": 1/max_ratio, "z": 1/max_ratio}})
            with open(self.misc_log, "at") as f:
                f.write(f'Object too large. Scaling down by a factor {1/max_ratio} \n')

    def clamp_min_scale(self, o_id, env):
        resp = self.communicate([{'$type': 'send_bounds',
                                  'ids': [o_id],
                                  'frequency': 'once'}])
        bds = Bounds(resp[-2])
        top = np.array(bds.get_top(0))
        bottom = np.array(bds.get_bottom(0))
        front = np.array(bds.get_front(0))
        back = np.array(bds.get_back(0))
        left = np.array(bds.get_left(0))
        right = np.array(bds.get_right(0))

        # See if the bounds are smaller than 1/20 the environment
        x_ratio = 10 * max(np.abs(front - back)) / env.w
        y_ratio = 5 * max(np.abs(top - bottom)) / (0.9 * env.h)
        z_ratio = 10 * max(np.abs(left - right)) / env.l

        min_ratio = max(x_ratio, y_ratio, z_ratio) # at least 1 edge of the object should be of decent size in the env
        if min_ratio < 1:
            self.communicate({"$type": "scale_object",
                              "id": o_id,
                              "scale_factor": {"x": 1 / min_ratio, "y": 1 / min_ratio, "z": 1 / min_ratio}})
            with open(self.misc_log, "at") as f:
                f.write(f'Object too small. Scaling up by a factor {1 / min_ratio} \n')

    def re_center(self, o_id, o_p):
        # Note that this modifies o_p
        # This assumed that send_images is not 'always'
        resp = self.communicate([{'$type': 'send_bounds',
                                  'ids': [o_id],
                                  'frequency': 'once'}])
        bds = Bounds(resp[-2])
        center = bds.get_center(0)
        bottom = bds.get_bottom(0)

        # get center to height 1, or if that isn't high enough, get object bottom to the ground, as long as center is below height 2.5
        o_p['y'] = min(max(o_p['y'] + 1 - center[1], o_p['y'] - bottom[1]), o_p['y'] + 2.5 - center[1])

        self.communicate([{"$type": "teleport_object",
                           "id": o_id,
                           "position": o_p},
                          {"$type": "look_at",
                           "avatar_id": 'a',
                           "object_id": o_id,
                           "use_centroid": True},
                          ])
        with open(self.misc_log, "at") as f:
            f.write(f'Recentering new o_p : {o_p}\n')

    def get_obj_pct_in_frame(self):
        # NOTE : This assumes that pass_masks have already been set to _mask
        resp = self.communicate({'$type': 'send_images',
                                 'frequency': 'once'})
        imgs = Images(resp[0])
        mask = np.array(TDWUtils.get_pil_image(imgs, 0).convert('L')) > 0
        obj_pct_in_frame = (mask.sum()) / float(mask.shape[0] * mask.shape[1])

        return obj_pct_in_frame


    def set_object_scale(self, o_id, env, a_id):
        a_p = self._get_default_avatar_position()
        o_p = np.array([0, 0, 0])

        a_p = TDWUtils.array_to_vector3(a_p)
        o_p = TDWUtils.array_to_vector3(o_p)

        self.communicate([
            {"$type": "send_images",
             "frequency": "never"},
            {"$type": "rotate_object_to",
             "id": o_id,
             "rotation": {"x": 0, "y": 0, "z": 0, "w": 0}},
            {"$type": "teleport_object",
             "id": o_id,
             "position": o_p},
            {"$type": "teleport_avatar_to",
             "avatar_id": a_id,
             "position": a_p},
            {"$type": "look_at",
             "avatar_id": a_id,
             "object_id": o_id,
             "use_centroid": True},
        ])

        self.clamp_max_scale(o_id, env)
        self.clamp_min_scale(o_id, env)
        self.re_center(o_id, o_p)

        resp = self.communicate([{"$type": "set_pass_masks",
                                  "avatar_id": 'a',
                                  "pass_masks": ["_mask"]}])

        obj_pct_in_frame = self.get_obj_pct_in_frame()
        upscale_times = 0
        while obj_pct_in_frame < 0.002:
            self.communicate([{"$type": "scale_object",
                               "id": o_id,
                               "scale_factor": {"x": 2, "y": 2, "z": 2}},
                              {"$type": "look_at",
                               "avatar_id": a_id,
                               "object_id": o_id,
                               "use_centroid": True}])
            self.re_center(o_id, o_p)
            upscale_times += 1

            obj_pct_in_frame = self.get_obj_pct_in_frame()

            with open(self.misc_log, "at") as f:
                f.write('Scaled up by a factor of {}, '.format(2 ** upscale_times))
                f.write('Current obj_pct_in_frame : {}\n'.format(obj_pct_in_frame))

        if obj_pct_in_frame < 0.005:
            # If the object is still small, move it closer to the camera
            a_p = TDWUtils.vector3_to_array(a_p)
            o_p = TDWUtils.vector3_to_array(o_p)

            o_p = a_p + 0.5 * (o_p - a_p)

            a_p = TDWUtils.array_to_vector3(a_p)
            o_p = TDWUtils.array_to_vector3(o_p)

            self.communicate([{"$type": "teleport_object",
                               "id": o_id,
                               "position": o_p},
                              {'$type': 'look_at',
                               'avatar_id': a_id,
                               'object_id' : o_id,
                               'use_centroid': True}])

        resp = self.communicate({"$type": "send_transforms",
                                 "frequency": "once",
                                 "ids": [o_id]})
        t = Transforms(resp[-2])
        o_rot = t.get_rotation(0)
        o_rot = {"x": o_rot[0],
                 "y": o_rot[1],
                 "z": o_rot[2],
                 "w": o_rot[3], }

        o_p = t.get_position(0)
        o_p = {
            'x' : o_p[0],
            'y' : o_p[1],
            'z' : o_p[2]
        }
        return o_p, o_rot

    def get_img_pos(self, gp:GenParams, o_id: int, o_init_p: dict, o_init_rot: dict, a_id: str, e: Environment) -> (float, float, dict, dict, dict, dict):
        """
        Get the "real" grayscale value of an image we hope to capture.

        :param o_id: The ID of the object.
        :param a_id: The ID of the avatar.
        :param e: The environment.

        :return: (grayscale, distance, avatar_position, object_position, object_rotation, avatar_rotation)
        """

        # Move to initial avatar and object locations
        a_p = TDWUtils.array_to_vector3(self._get_default_avatar_position())

        commands = [{'$type': 'teleport_avatar_to',
                     'position': a_p,
                     'id': a_id},
                    {'$type': 'teleport_object',
                     'id': o_id,
                     'position': o_init_p,},
                    {"$type": "rotate_object_to",
                     "id": o_id,
                     "rotation": o_init_rot},]

        if gp.pose_rot == 1:
            commands.extend([
                {"$type": "rotate_object_by",
                 "id": o_id,
                 "angle": RNG.uniform(-self.max_rot_angle, self.max_rot_angle),
                 "axis": "pitch"},
                {"$type": "rotate_object_by",
                 "id": o_id,
                 "angle": RNG.uniform(-self.max_rot_angle, self.max_rot_angle),
                 "axis": "yaw"},
                {"$type": "rotate_object_by",
                 "id": o_id,
                 "angle": RNG.uniform(-self.max_rot_angle, self.max_rot_angle),
                 "axis": "roll"}
            ])

        if gp.pose_scale == 1:
            a_p = self._get_default_avatar_position()
            d = RNG.uniform(self.min_dist, self.max_dist)  # note that d here is a scaled distance
            o_init_p = TDWUtils.vector3_to_array(o_init_p)
            o_p = a_p + (o_init_p - a_p) * d
            o_p = TDWUtils.array_to_vector3(o_p)
            a_p = TDWUtils.array_to_vector3(a_p)
            commands.extend([{"$type": "teleport_object",
                              "id": o_id,
                              "position": o_p},
                             {'$type': 'look_at',
                              'avatar_id': a_id,
                              'object_id' : o_id,
                              'use_centroid': True}])


        if gp.backgr == 1:
            # Get a random position for the avatar.
            a_p = self._get_default_avatar_position()
            look_pos = TDWUtils.array_to_vector3(a_p + Generator.sample_spherical())
            a_p = TDWUtils.array_to_vector3(a_p)

            commands.extend([
                {"$type": "parent_object_to_avatar",
                 "id": o_id,
                 "avatar_id": "a",
                 "sensor": True},
                {'$type': 'look_at_position',
                 'position': look_pos,
                 'id': a_id},
                {"$type": "unparent_object",
                 "id": o_id},
            ])

        if gp.pose_cam == 1:
            yaw = RNG.uniform(-15, 15)
            pitch = RNG.uniform(-15, 15)
            commands.extend([{"$type": "rotate_sensor_container_by",
                              "angle": pitch,
                              "avatar_id": "a",
                              "axis": "pitch"},
                             {"$type": "rotate_sensor_container_by",
                              "angle": yaw,
                              "avatar_id": "a",
                              "axis": "yaw"}, ])

        commands.extend([{"$type": "send_image_sensors",
                          "frequency": "once"}])

        commands.append({"$type": "send_transforms",
                         "frequency": "once",
                         "ids": [o_id]})
        resp = self.communicate(commands)

        # Parse the output data:
        # The camera rotation and the object position and rotation
        cam_rot = None
        o_p = None
        o_rot = None
        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "imse":
                cam_rot = ImageSensors(r).get_sensor_rotation(0)
                cam_rot = {"x": cam_rot[0], "y": cam_rot[1], "z": cam_rot[2], "w": cam_rot[3]}
            elif r_id == 'tran':
                t = Transforms(r)
                o_rot = t.get_rotation(0)
                o_rot = {"x": o_rot[0],
                         "y": o_rot[1],
                         "z": o_rot[2],
                         "w": o_rot[3], }
                o_p = t.get_position(0)
                o_p = {
                    "x": o_p[0],
                    "y": o_p[1],
                    "z": o_p[2]
                }

        return a_p, o_p, o_rot, cam_rot

    def set_skybox(self, records: List[HDRISkyboxRecord], hdri_order: np.ndarray, its_per_skybox: int, hdri_index: int, skybox_count: int) -> (
    int, int, dict):
        """
        If it's time, set a new skybox.

        :param records: All HDRI records.
        :param its_per_skybox: Iterations per skybox.
        :param hdri_index: The index in the records list.
        :param skybox_count: The number of images of this model with this skybox.
        :return: (hdri_index, skybox_count, command used to set the skybox)
        """

        # Set a new skybox.
        if skybox_count == 0:
            command = self.get_add_hdri_skybox(records[hdri_order[hdri_index]].name)
        # It's not time yet to set a new skybox. Don't send a command.
        else:
            command = None
        skybox_count += 1

        if skybox_count >= its_per_skybox:
            hdri_index += 1
            if hdri_index >= len(records):
                hdri_index = 0
            skybox_count = 0

        return hdri_index, skybox_count, command

    def get_lighting_opts(self, gen_params: GenParams, prev_light_pert: float=0):
        commands = []
        light_pert = 0
        if gen_params.lighting_dir == 1:
            commands.extend([{'$type': 'rotate_directional_light_by',
                              'angle': RNG.uniform(0, 360),
                              'axis': 'yaw'},
                             {'$type': 'rotate_directional_light_by',
                              'angle': RNG.uniform(0, 360),  # allowing all angles
                              'axis': 'pitch'}
                             ])

        if gen_params.lighting_color == 1:  # changes both
            commands.extend([{'$type': 'set_directionial_light_color',
                              "color": {
                                  "r": RNG.uniform(0.05, 1),
                                  "g": RNG.uniform(0.05, 1),
                                  "b": RNG.uniform(0.05, 1),
                                  "a": 1.0}
                              }])

        if gen_params.lighting_intensity == 1:
            light_pert = RNG.uniform(
                -self.max_light_intensity_var, self.max_light_intensity_var)
            commands.extend([{'$type': 'adjust_directional_light_intensity_by',
                              'intensity': -prev_light_pert + light_pert}])

        return commands, light_pert

    def get_blur_opts(self, gen_params: GenParams, p: ImagePosition):
        commands = []
        if gen_params.blur == 1:
            d = TDWUtils.get_distance(p.avatar_position, p.object_position)
            if RNG.rand() < 0.75:
                commands.extend([{'$type': 'set_focus_distance',
                                  'focus_distance': d + RNG.uniform(-1, 1)},
                                 {"$type": "set_aperture",
                                  "aperture": RNG.uniform(0.5, 1.2)},
                                 ])
            else:  # sometimes everything is in focus
                commands.extend([{"$type": "set_aperture",
                                  "aperture": 70}])

        return commands

    def save_image(self, resp, record: ModelRecord, image_count: int, datadir: Path, wnid: str) -> None:
        """
        Save an image.

        :param resp: The raw response data.
        :param record: The model record.
        :param image_count: The image count.
        :param root_dir: The root directory.
        :param wnid: The wnid.
        :param train: Number of train images so far.
        :param train_count: Total number of train images to generate.
        """

        # Get the directory.
        directory = datadir.joinpath(wnid).resolve()
        if not os.path.exists(directory):
            # Try to make the directories. Due to threading, they might already be made.
            try:
                os.makedirs(directory)
            except:
                pass

        # Save the image.
        filename = f"{record.name}_{image_count:04d}"

        # Save the image without resizing.
        if self.screen_size == self.output_size:
            TDWUtils.save_images(Images(resp[0]), filename,
                                 output_directory=directory)
        # Resize the image and save it.
        else:
            TDWUtils.save_images(Images(resp[0]), filename,
                                 output_directory=directory,
                                 resize_to=(self.output_size, self.output_size))

    def process_model(
            self, gen_params:GenParams, record:ModelRecord, avatar_id:str,
            envs:list, num_imgs_per_obj:int, wnid:str, datadir:Path) -> float:

        self.model_times[record.name] = {}
        image_count = 0
        image_positions = []
        o_id = self.get_unique_id()
        s = TDWUtils.get_unit_scale(record)
        hdri_order = RNG.permutation(len(self.skyboxes))

        if self.load_path:
            obj_url = (Path(self.load_path) / 'tdw_asset_bundles/models/{}'.format(record.name)).as_uri()
        else:
            obj_url = record.get_url()

        # Add the object.
        # Set the screen size to 32x32 (to make the build run faster; we only need rough occlusion values).
        # Toggle off pass masks.
        # Set render quality to minimal.
        # Scale the object to "unit size".
        start = time()
        self.communicate([{"$type": "add_object",
                           "name": record.name,
                           "url": obj_url,
                           "scale_factor": record.scale_factor,
                           "category": record.wcategory,
                           "id": o_id},
                          {"$type": "set_screen_size",
                           "height": 32,
                           "width": 32},
                          {"$type": "set_pass_masks",
                           "avatar_id": avatar_id,
                           "pass_masks": []},
                          {"$type": "set_render_quality",
                           "render_quality": 0},
                          {"$type": "scale_object",
                           "id": o_id,
                           "scale_factor": {"x": s, "y": s, "z": s}}])
        self.model_times[record.name]['load_time'] = time() - start


        with open(self.misc_log, 'at') as f:
            f.write(f'Setting scale for object {record.name}\n')
        start = time()
        o_init_p, o_init_rot = self.set_object_scale(o_id, envs[0], avatar_id)
        self.model_times[record.name]['set_scale_time'] = time() - start

        # The index in the HDRI records array.
        hdri_index = 0
        # The number of iterations on this skybox so far.
        skybox_count = 0
        # The number of iterations per skybox for this model.
        its_per_skybox = max(2, round(num_imgs_per_obj / len(self.skyboxes)))

        start = time()
        if gen_params.backgr == 1:
            # Set the first skybox
            hdri_index, skybox_count, command = self.set_skybox(
                self.skyboxes, hdri_order, its_per_skybox, hdri_index, skybox_count)
            self.model_times[record.name]['skybox_times'] = [time() - start]
        else:
            command = self.get_add_hdri_skybox('fish_hoek_beach_4k')
            self.model_times[record.name]['skybox_time'] = time() - start
        self.communicate(command)
        # Add changing skybox time as well


        self.model_times[record.name]['img_pos_times'] = []
        while len(image_positions) < num_imgs_per_obj:
            e = RNG.choice(envs)  # This should be a single element, making the random choice moot

            start = time()
            # Get the real grayscale, in case of occlusion, it returns occlusion as g_r
            a_p, o_p, o_rot, cam_rot = self.get_img_pos(
                gen_params, o_id, o_init_p, o_init_rot, avatar_id, e)

            self.model_times[record.name]['img_pos_times'].append(time() - start)
            image_positions.append(ImagePosition(a_p, cam_rot, o_p, o_rot))

        # Send images.
        # Set the screen size.
        # Set render quality to maximum.
        commands = [{"$type": "send_images",
                     "frequency": "never"},
                    {"$type": "set_pass_masks",
                     "avatar_id": avatar_id,
                     "pass_masks": ["_img"]},
                    {"$type": "set_screen_size",
                     "height": self.screen_size,
                     "width": self.screen_size},
                    {"$type": "set_render_quality",
                     "render_quality": 5}]

        start = time()
        if gen_params.materials == 0 or gen_params.materials == 2:
            if record.name not in self.substructures:
                self.substructures.update({record.name: record.substructure})
            for sub_object in self.substructures[record.name]:
                for i in range(len(sub_object["materials"])):
                    if gen_params.materials == 0:
                        material_name = 'cotton_natural_rough'
                    elif gen_params.materials == 2:
                        material_name = self.materials[RNG.randint(0, len(self.materials))].name
                    commands.extend([self.get_add_material(material_name),
                                     {"$type": "set_visual_material",
                                      "id": o_id,
                                      "material_name": material_name,
                                      "object_name": sub_object["name"],
                                      "material_index": i}])
        self.communicate(commands)
        self.model_times[record.name]['material_time'] = time() - start

        self.model_times[record.name]['move_times'] = []
        self.model_times[record.name]['light_times'] = []
        self.model_times[record.name]['blur_times'] = []
        self.model_times[record.name]['material_times'] = []
        self.model_times[record.name]['image_times'] = []

        t0 = time()

        # Generate images from the cached spatial data.
        train = 0
        prev_light_pert = 0 # For light perturbations
        for p in image_positions:
            # Teleport the avatar.
            # Rotate the avatar's camera.
            # Teleport the object.
            # Rotate the object.
            # Get the response.
            start = time()
            commands = [{"$type": "teleport_avatar_to",
                         "avatar_id": avatar_id,
                         "position": p.avatar_position},
                        {"$type": "rotate_sensor_container_to",
                         "avatar_id": avatar_id,
                         "rotation": p.camera_rotation},
                        {"$type": "teleport_object",
                         "id": o_id,
                         "position": p.object_position},
                        {"$type": "rotate_object_to",
                         "id": o_id,
                         "rotation": p.object_rotation}]

            self.communicate(commands)
            self.model_times[record.name]['move_times'].append(time() - start)

            commands = []
            start = time()
            # Maybe set a new skybox.
            if gen_params.backgr == 1:
                hdri_index, skybox_count, command = self.set_skybox(
                    self.skyboxes, hdri_order, its_per_skybox, hdri_index, skybox_count)
                if command:
                    prev_light_pert = 0 # reset lighting intensity perturbation
                    commands.append(command)
                if len(commands) > 0:
                    self.communicate(commands)
                    self.model_times[record.name]['skybox_times'].append(time() - start)

            start = time()
            commands = []
            light_commands, prev_light_pert = self.get_lighting_opts(gen_params, prev_light_pert)
            commands.extend(light_commands)
            self.communicate(commands)
            self.model_times[record.name]['light_times'].append(time() - start)

            start = time()
            commands = []
            blur_commands = self.get_blur_opts(gen_params, p)
            commands.extend(blur_commands)
            self.communicate(commands)
            self.model_times[record.name]['blur_times'].append(time() - start)


            # Set the visual materials.
            if gen_params.materials == 3:
                start = time()
                commands = []
                if record.name not in self.substructures:
                    self.substructures.update({record.name: record.substructure})
                for sub_object in self.substructures[record.name]:
                    for i in range(len(sub_object["materials"])):
                        material_name = self.materials[RNG.randint(0, len(self.materials))].name
                        commands.extend([self.get_add_material(material_name),
                                         {"$type": "set_visual_material",
                                          "id": o_id,
                                          "material_name": material_name,
                                          "object_name": sub_object["name"],
                                          "material_index": i}])
                self.communicate(commands)
                self.model_times[record.name]['material_times'].append(time() - start)

            start = time()
            resp = self.communicate({'$type': 'send_images',
                                     'frequency': 'once'})
            self.model_times[record.name]['image_times'].append(time() - start)

            # Create a thread to save the image.
            t = Thread(target=self.save_image, args=(resp, record, image_count, datadir, wnid))
            t.daemon = True
            t.start()
            train += 1
            image_count += 1
        t1 = time()

        # Stop sending images.
        # Destroy the object.
        # Unload asset bundles.
        self.communicate([{"$type": "send_images",
                           "frequency": "never"},
                          {"$type": "destroy_object",
                           "id": o_id},
                          {"$type": "unload_asset_bundles"}])
        os.makedirs(datadir, exist_ok=True)
        with open(os.path.join(datadir, 'model_times.json'), 'w') as f:
            json.dump(self.model_times, f)

        return t1 - t0


    def check_existing(self, gen_params:GenParams):
        datadir = self.root_dir / 'dataset_{}'.format(gen_params.get_tuple_str())
        if os.path.exists(datadir):
            # This means this dataset could've been only half-generated
            if os.path.exists(datadir / 'metadata.txt'):
                with open(datadir / 'metadata.txt') as f:
                    run_metadata = json.load(f)
            else:
                return False

            # NOTE: Assuming that if the directory exists, metadata file would as well
            if 'end' in run_metadata:
                # Meaning dataset is complete
                return True
            else:
                return False
        else:
            return False

    def gen_data(self, gen_params: GenParams, num_imgs: int) -> Path:

        print('Generating {} imgs for GenParams : {}'.format(num_imgs, gen_params.get_tuple_str()))
        # Set the number of train and val images per wnid.
        if self.simple_objs:
            num_imgs_per_wnid = num_imgs // len(self.objs)
        else:
            num_imgs_per_wnid = num_imgs // len(self.wnids)

        start_time = time()

        # Create the progress bar.
        datadir = self.root_dir / 'dataset_{}'.format(gen_params.get_tuple_str())
        imgdir = datadir / 'images'
        done_models_filename = os.path.join(datadir, 'processed_records.txt')
        self.occ_fail_file = os.path.join(datadir, 'id_fail.txt')
        self.misc_log = os.path.join(datadir, 'misc_output.txt')

        if os.path.exists(datadir):
            # This means this dataset could've been only half-generated
            if os.path.exists(datadir / 'metadata.txt'):
                with open(datadir / 'metadata.txt') as f:
                    run_metadata = json.load(f)
                    # NOTE: Assuming that if the directory exists, metadata file would as well
                    if 'end' in run_metadata:
                        # Meaning dataset is complete
                        return imgdir
                    else:
                        if os.path.exists(done_models_filename):
                            with open(done_models_filename, "rt") as f:
                                processed_model_names = f.read().splitlines()
                            print('Warning : Completing part-finished dataset in {}'.format(datadir))
                        else:
                            # Remove
                            for sub_path in os.listdir(datadir):
                                if os.path.isdir(datadir / sub_path):
                                    shutil.rmtree(datadir / sub_path)
                                else:
                                    os.remove(datadir / sub_path)
                            processed_model_names = []
            else:
                for sub_path in os.listdir(datadir):
                    if os.path.isdir(datadir / sub_path):
                        shutil.rmtree(datadir / sub_path)
                    else:
                        os.remove(datadir / sub_path)
                processed_model_names = []
        else:
            os.makedirs(datadir)
            processed_model_names = []

        if self.simple_objs:
            wnid_iter = self.objs
        else:
            wnid_iter = self.wnids

        pbar = tqdm(total=len(wnid_iter))

        # write metadata : overwrites old metadata
        self.generate_metadata(datadir, gen_params)

        # Iterate through each wnid.
        for q, w in enumerate(wnid_iter):
            # Update the progress bar.
            pbar.set_description(w)

            # Get all valid models in the wnid.
            if self.simple_objs:
                records = [self.model_librarian.get_record(name) for name in self.objs[w]]
            else:
                records = self.model_librarian.get_all_models_in_wnid(w)
                records = [r for r in records if not r.do_not_use]

            # Get the train and val counts.
            num_imgs_per_obj = (num_imgs_per_wnid//len(records))

            # Process each record.
            fps = "nan"
            for record, i in zip(records, range(len(records))):
                # Set the progress bar description to the wnid and FPS.
                pbar.set_description(f"record {i + 1}/{len(records)}, FPS {fps}")

                # Skip models that have already been processed.
                if record.name in processed_model_names:
                    continue

                # Create all of the images for this model.
                dt = self.process_model(gen_params, record, self.avatar_id, self.envs, num_imgs_per_obj, w, imgdir)
                if dt > 0.: # just adding bc this created a divbyzero error once
                    fps = round(num_imgs_per_obj / dt)
                else:
                    fps = "inf"

                # Mark this record as processed.
                with open(done_models_filename, "at") as f:
                    f.write(f"\n{record.name}")

            pbar.update(1)
        pbar.close()

        total_time = time() - start_time

        print('Total time taken for generation : {:.2f} s'.format(total_time))
        # Add the end time to the metadata file.
        with open(os.path.join(datadir, "metadata.txt"), "rt") as f:
            data = json.load(f)
            end_time = datetime.now().strftime("%H:%M %d.%m.%y")
            if "end" in data:
                data["end"] = end_time
            else:
                data.update({"end": end_time})
            data['total_time'] = total_time
        with open(os.path.join(datadir, "metadata.txt"), "wt") as f:
            json.dump(data, f, sort_keys=True, indent=4)

        return imgdir