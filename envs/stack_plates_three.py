from ._base_task import Base_Task
from .utils import *
import sapien
import math


class stack_plates_three(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        plate_pose_lst = []
        for i in range(3):
            plate_pose = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[-0.15, 0.15],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )

            def check_plate_pose(plate_pose):
                for j in range(len(plate_pose_lst)):
                    if (np.sum(pow(plate_pose.p[:2] - plate_pose_lst[j].p[:2], 2)) < 0.03):
                        return False
                return True

            while (abs(plate_pose.p[0]) < 0.09 or np.sum(pow(plate_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.03
                   or not check_plate_pose(plate_pose)):
                plate_pose = rand_pose(
                    xlim=[-0.3, 0.3],
                    ylim=[-0.15, 0.15],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    ylim_prop=True,
                    rotate_rand=False,
                )
            plate_pose_lst.append(deepcopy(plate_pose))

            plate_pose_lst = sorted(plate_pose_lst, key=lambda x: x.p[1])

            def create_plate(plate_pose):
                return create_actor(self, pose=plate_pose, modelname="003_plate", model_id=0, convex=True)

        self.plate1 = create_plate(plate_pose_lst[0])
        self.plate2 = create_plate(plate_pose_lst[1])
        self.plate3 = create_plate(plate_pose_lst[2])

        self.add_prohibit_area(self.plate1, padding=0.07)
        self.add_prohibit_area(self.plate2, padding=0.07)
        self.add_prohibit_area(self.plate3, padding=0.07)
        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)
        self.plate1_target_pose = np.array([0, -0.1, 0.755])
        self.quat_of_target_pose =  [0, 0.707, 0.707, 0]

    def move_plate(self, actor, target_pose):
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        if self.las_arm is None or arm_tag == self.las_arm:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    contact_point_id=0,
                    pre_grasp_dis=0.1,
                ))
        else:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    contact_point_id=0,
                    pre_grasp_dis=0.1,
                ),  # arm_tag
                self.back_to_origin(arm_tag=arm_tag.opposite),  # arm_tag.opposite
            )
        self.move(self.move_by_displacement(arm_tag, z=0.1))
        self.move(
            self.place_actor(
                actor,
                target_pose=target_pose.tolist() + self.quat_of_target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.02,
                dis=0,
                constrain="snap",
            ))
        self.move(self.move_by_displacement(arm_tag, z=-0.02))
        self.open_gripper(arm_tag)
        self.move(self.move_by_displacement(arm_tag, x=-0.05))
        self.move(self.move_by_displacement(arm_tag, z=0.05))
        self.las_arm = arm_tag
        return arm_tag

    def play_once(self):
        # Initialize last arm used to None
        self.las_arm = None

        # Move plate1 to position [0, -0.1, 0.76]
        self.move_plate(self.plate1, self.plate1_target_pose)
        # Move plate2 to be 0.05m above plate1's position
        self.move_plate(self.plate2, self.plate1.get_pose().p + [0, 0, 0.05])
        # Move plate3 to be 0.05m above plate2's position
        self.move_plate(self.plate3, self.plate2.get_pose().p + [0, 0, 0.075])

        self.info["info"] = {"{A}": f"002_plate/base0"}
        return self.info

    def check_success(self):
        plate1_pose = self.plate1.get_pose().p
        plate2_pose = self.plate2.get_pose().p
        plate3_pose = self.plate3.get_pose().p
        plate1_pose, plate2_pose, plate3_pose = sorted([plate1_pose, plate2_pose, plate3_pose], key=lambda x: x[2])
        target_height = [
            0.74 + self.table_z_bias,
            0.77 + self.table_z_bias,
            0.81 + self.table_z_bias,
        ]
        eps = 0.05
        eps2 = 0.08
        return (np.all(abs(plate1_pose[:2] - plate2_pose[:2]) < eps2)
                and np.all(abs(plate2_pose[:2] - plate3_pose[:2]) < eps2)
                and np.all(np.array([plate1_pose[2], plate2_pose[2], plate3_pose[2]]) - target_height < eps)
                and self.is_left_gripper_open() and self.is_right_gripper_open())
