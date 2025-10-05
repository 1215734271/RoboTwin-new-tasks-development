from ._base_task import Base_Task
from .utils import *
import sapien
from copy import deepcopy
import numpy as np


class put_bottles_dustbin_new(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwargs)

    def load_actors(self):
        pose_lst = []

        # 定义瓶子 ID（这里先写死三个 ID，和你原来一致）
        self.bottle_id = [0, 1, 2]
        self.bottle_num = 3

        # 随机打乱 ID 确保不重复
        bottle_ids = np.random.permutation(self.bottle_id).tolist()

        def create_bottle(model_id, xlim):
            bottle_pose = rand_pose(
                xlim=xlim,
                ylim=[0.03, 0.23],
                rotate_rand=False,
                rotate_lim=[0, 1, 0],
                qpos=[0.707, 0.707, 0, 0],
            )

            tag = True
            gen_lim = 100
            i = 1
            while tag and i < gen_lim:
                tag = False
                # 防止瓶子太靠近中心
                if np.abs(bottle_pose.p[0]) < 0.05:
                    tag = True
                # 防止瓶子重叠
                for pose in pose_lst:
                    if np.sum(np.power(np.array(pose[:2]) - np.array(bottle_pose.p[:2]), 2)) < 0.0169:
                        tag = True
                        break
                if tag:
                    i += 1
                    bottle_pose = rand_pose(
                        xlim=xlim,
                        ylim=[0.03, 0.23],
                        rotate_rand=False,
                        rotate_lim=[0, 1, 0],
                        qpos=[0.707, 0.707, 0, 0],
                    )

            pose_lst.append(bottle_pose.p[:2])

            bottle = create_actor(
                self,
                bottle_pose,
                modelname="123_bottle2",
                convex=True,
                model_id=model_id,
            )
            return bottle

        # 初始化列表
        self.bottles = []
        self.bottles_data = []

        # 左边两个瓶子
        for i in range(2):
            bottle = create_bottle(bottle_ids[i], xlim=[-0.25, 0])
            self.bottles.append(bottle)
            self.add_prohibit_area(bottle, padding=0.1)

        # 右边一个瓶子
        bottle = create_bottle(bottle_ids[2], xlim=[0, 0.3])
        self.bottles.append(bottle)
        self.add_prohibit_area(bottle, padding=0.1)

        # 垃圾桶
        self.dustbin = create_actor(
            self.scene,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )

        self.delay(2)

        # 中间辅助块
        self.middle_block = create_box(
            scene=self,
            pose=sapien.Pose([self.bottles[2].get_pose().p[0]-0.25, self.bottles[2].get_pose().p[1], 0.74], [1, 0, 0, 0]),
            half_size=(0.0001, 0.0001, 0.0001),
            color=(0, 0, 0),
            is_static=True,
            name="middle_box",
        )

        self.middle_block.config["functional_matrix"] = [
            [
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]

    def play_once(self):
        # Sort bottles based on x and y coordinates
        bottle_lst = sorted(self.bottles, key=lambda x: [x.get_pose().p[0] > 0, x.get_pose().p[1]])

        left_end_action = Action("left", "move", [-0.35, -0.1, 0.93, 0.65, -0.25, 0.25, 0.65])
        delta_dis = 0.04

        for i in range(self.bottle_num):
            bottle = bottle_lst[i]
            arm_tag = ArmTag("left" if bottle.get_pose().p[0] < 0 else "right")

            if arm_tag == "left":
                # ----- LEFT ARM -----
                # Grasp the bottle
                self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
                # Move arm up
                self.move(self.move_by_displacement(arm_tag, z=0.1))
                # Move to end position
                self.move((ArmTag("left"), [left_end_action]))
                # Open gripper after reaching end position
                self.move(self.open_gripper("left"))

            else:
                # ----- RIGHT ARM -----
                # Grasp the bottle
                self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
                # Move up
                self.move(self.move_by_displacement(arm_tag, z=0.1))
                # Place the bottle at middle block
                target_pose = self.middle_block.get_functional_point(0)
                self.move(
                    self.place_actor(
                        bottle,
                        arm_tag=arm_tag,
                        target_pose=target_pose,
                        functional_point_id=0,
                        pre_dis=0.1,
                        constrain="free",
                    )
                )
                # Open right gripper
                self.move(self.open_gripper("right"))
                # Move right arm back to origin
                self.move(self.back_to_origin("right"))

                # ----- LEFT ARM PICK AFTER RIGHT -----
                # Grasp with left arm
                self.move(self.grasp_actor(bottle, arm_tag="left", pre_grasp_dis=0.1))
                # Adjust height
                self.move(self.move_by_displacement("left", z=-delta_dis))
                # Move to end position
                self.move((ArmTag("left"), [left_end_action]))
                # Open left gripper after reaching end
                self.move(self.open_gripper("left"))

        self.info["info"] = {
            "{A}": f"123_bottle2/base{self.bottle_id[0]}",
            "{B}": f"123_bottle2/base{self.bottle_id[1]}",
            "{C}": f"123_bottle2/base{self.bottle_id[2]}",
            "{D}": f"011_dustbin/base0",
        }

        return self.info

    def check_success(self):
        target_pose = [-0.45, 0]
        eps = np.array([0.221, 0.325])

        for i in range(self.bottle_num):
            bottle_pose = self.bottles[i].get_pose().p
            if not (np.all(np.abs(bottle_pose[:2] - target_pose) < eps) and 0.2 < bottle_pose[2] < 0.7):
                return False
        return True
