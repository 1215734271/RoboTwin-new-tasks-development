from ._base_task import Base_Task
from .utils import *
import math
import sapien


class place_sandal(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def reset(self, seed=None):
        """每次 episode 清完场景后重新创建物体"""
        super().reset(seed=seed)
        self.load_actors()
        self.sandal = self.scene.get_actor("122_sandal")

    def load_actors(self):
        self.target_block = create_box(
            scene=self,
            pose=sapien.Pose([0, -0.08, 0.74], [1, 0, 0, 0]),
            half_size=(0.13, 0.05, 0.0005),
            color=(0, 0, 1),
            is_static=True,
            name="box",
        )
        self.target_block.config["functional_matrix"] = [[
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ], [
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]]

        sandal_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.1, 0.05],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            qpos=[0.707, 0.707, 0, 0],
        )
        while np.sum(pow(sandal_pose.get_p()[:2] - np.zeros(2), 2)) < 0.0225:
            sandal_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.1, 0.05],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
        self.sandal_id = 0
        self.sandal = create_actor(
            scene=self,
            pose=sandal_pose,
            modelname="122_sandal",
            convex=True,
            model_id=self.sandal_id,
        )

        self.prohibited_area.append([-0.2, -0.15, 0.2, -0.01])
        self.add_prohibit_area(self.sandal, padding=0.1)

    def play_once(self):
        sandal_pose = self.sandal.get_pose().p
        arm_tag = ArmTag("left" if sandal_pose[0] < 0 else "right")

        # Grasp the sandal with specified pre-grasp distance and gripper position
        self.move(self.grasp_actor(self.sandal, arm_tag=arm_tag, pre_grasp_dis=0.1, gripper_pos=0))

        # Lift the sandal up by 0.07 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        # Get target's functional point as target pose
        target_pose = self.target_block.get_functional_point(0)
        # Place the sandal on the target with alignment constraint and specified pre-placement distance
        self.move(
            self.place_actor(
                self.sandal,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.12,
                constrain="align",
            ))
        # Open the gripper to release the sandal
        self.move(self.open_gripper(arm_tag=arm_tag))

        self.info["info"] = {"{A}": f"122_sandal/base{self.sandal_id}", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        sandal_pose_p = np.array(self.sandal.get_pose().p)
        sandal_pose_q = np.array(self.sandal.get_pose().q)
        if sandal_pose_q[0] < 0:
            sandal_pose_q *= -1
        target_pose_p = np.array([0, -0.08])
        target_pose_q = np.array([0.5, 0.5, -0.5, -0.5])
        eps = np.array([0.05, 0.02, 0.07, 0.07, 0.07, 0.07])
        return (np.all(abs(sandal_pose_p[:2] - target_pose_p) < eps[:2])
                and np.all(abs(sandal_pose_q - target_pose_q) < eps[-4:]) and self.is_left_gripper_open()
                and self.is_right_gripper_open())
