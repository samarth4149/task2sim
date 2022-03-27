from yacs.config import CfgNode as CN


class GenParams(CN):
    def __init__(self, config_dict=None):
        super(GenParams, self).__init__()

        # Each option has to be an integer from 0 to a certain max val

        # Lighting levels
        # 0-1 for each
        self.lighting_intensity = 0
        self.lighting_color = 0
        self.lighting_dir = 0

        # Blurring
        # Currently : 0-1 (blur or no blur)
        # Can add levels as to how often blurring is to take place,
        # OR 0 : no blur, 1: only background blur, 2: everything can be blurred
        self.blur = 0

        # Pose
        # 0-1 for each
        self.pose_rot = 0
        self.pose_scale = 0
        self.pose_cam = 0
        # Another Possibility
        # # 0 : No pose var
        # # 1 : Rotation
        # # 2 : Scale Variation (distance from cam)
        # # 3 : Rotation and Scale
        # # 4 : Rotation Scale and Camera angle changes (for off centering)

        # Background
        # 0 : Fixed background
        # 1 : Varying background (using both cam location and hdri)
        self.backgr = 0

        # Materials :
        # 0 : Single material
        # 1 : Natural materials
        # 2 : Random but fixed materials
        # 3 : Random materials in each image
        self.materials = 1

        # NOTE that if the config dict doesn't have any item, it gets set to the default
        if config_dict is not None:
            self.update(config_dict)

        # TODO : add these options
        # self.num_imgs
        # self.num_objs_per_class = None

    def get_tuple(self, params=None):
        if params is not None:
            return tuple([self[p] for p in params])
        else:
            return tuple([v for _, v in sorted(self.items())])

    def get_tuple_str(self):
        return '_'.join([str(v) for _, v in sorted(self.items())])

    @classmethod
    def get_ranges(cls):
        """
        returns a GenParams of max values for each option
        """
        ret = cls()
        ret.lighting_intensity = 2
        ret.lighting_color = 2
        ret.lighting_dir = 2
        ret.blur = 2
        ret.pose_rot = 2
        ret.pose_scale = 2
        ret.pose_cam = 2
        ret.backgr = 2
        ret.materials = 4
        return ret

    @classmethod
    def from_tuple(cls, params: tuple):
        ret = cls()
        if len(params) != len(ret.keys()):
            raise Exception('Params has incorrect length {}. Expected {}'.format(
                len(params), len(ret)))
        for p, k in zip(params, sorted(ret.keys())):
            ret[k] = p
        return ret

    @classmethod
    def from_tuple_str(cls, params : str):
        ret = cls()
        params = params.split('_')
        if len(params) != len(ret.keys()):
            raise Exception('Params has incorrect length {}. Expected {}'.format(
                len(params), len(ret)))
        for p, k in zip(params, sorted(ret.keys())):
            ret[k] = int(p)

        return ret

    @classmethod
    def get_keys(cls):
        return cls().keys()