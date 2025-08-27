from ._base_task import Base_Task
from .utils import *
import sapien

class put_fell_bottle_dustbin(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):
        # 四元数表示可能的初始旋转姿态
        # ori_qpos = [[0.707,0.707,0,0],[0.707,0.707,0,0],[0.707,0.707,0,0],[0.707,0.707,0,0]]
        ori_qpos = [[0.505, 0.505, 0.494, -0.495],[0.707,0,0.707,0],[0.496, 0.495, 0.505, -0.504],[0.707,0,0.707,0]]
        self.model_id = np.random.choice([1,2,3,4],1)[0]         # 随机模型版本选取
        bottle_pose = rand_pose(
            xlim=[-0.2, -0.01],
            ylim=[-0.2, 0.2],
            qpos=ori_qpos[self.model_id-1],
            zlim=[0.85, 0.85],
            rotate_rand=True,
            rotate_lim=[0, 0.8, 0],
        )            # 随机位姿

        self.bottle = create_actor(
            scene=self,
            pose=bottle_pose,
            modelname="114_bottle",
            # modelname="121_fell_bottle",
            convex=True,
            model_id=self.model_id,
        )
        if self.bottle is None:
            raise RuntimeError("create_actor 返回 None，姿态/路径问题")  # 立即崩溃，方便定位
        self.add_prohibit_area(self.bottle, padding=0.1)        # 添加禁区

        self.dustbin = create_actor(
            self.scene,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )
        self.delay(2)
        self.right_middle_pose = [0, 0.0, 0.88, 0, 1, 0, 0]

    def play_once(self):
        bottle = self.bottle

        # 根据瓶子位姿决定左右壁的优先使用
        arm_tag = ArmTag("left" if bottle.get_pose().p[0] < 0 else "right")

        delta_dis = 0.06

        # 为左臂定义结束位置
        left_end_action = Action("left", "move", [-0.35, -0.1, 0.93, 0.65, -0.25, 0.25, 0.65])

        if arm_tag == "left":
            # 左臂抓取瓶子
            self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
            # 左臂抬升
            self.move(self.move_by_displacement(arm_tag, z=0.1))
            # 左臂移至结束位置
            self.move((ArmTag("left"), [left_end_action]))
        else:
            # 用右臂抓取瓶子移至左臂初始位置
            right_action = self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1)
            print(right_action)
            right_action[1][0].target_pose[2] += delta_dis
            right_action[1][1].target_pose[2] += delta_dis
            self.move(right_action, self.back_to_origin("left"))
            # 右臂抬升
            self.move(self.move_by_displacement(arm_tag, z=0.1))
            # 将瓶子置于右臂中部高度位置
            self.move(
                self.place_actor(
                    bottle,
                    target_pose=self.right_middle_pose,
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.0,
                    dis=0.0,
                    is_open=False,
                    constrain="align",
                ))
            # 用左臂抓取（判断高度）
            left_action = self.grasp_actor(bottle, arm_tag="left", pre_grasp_dis=0.1)
            left_action[1][0].target_pose[2] -= delta_dis
            left_action[1][1].target_pose[2] -= delta_dis
            self.move(left_action)
            # 打开右臂夹爪
            self.move(self.open_gripper(ArmTag("right")))
            # 当右臂移至左臂初始位置时移动左臂到结束位置
            self.move((ArmTag("left"), [left_end_action]), self.back_to_origin("right"))
        # 打开左臂夹爪
        self.move(self.open_gripper("left"))

        self.info["info"] = {
            "{A}": f"114_bottle/base{self.model_id}",
            "{D}": f"011_dustbin/base0",
        }
        return self.info

    def stage_reward(self):
        taget_pose = [-0.45, 0]
        eps = np.array([0.221, 0.325])
        reward = 0
        reward_step = 1
        bottle_pose = self.bottle.get_pose().p
        if np.all(np.abs(bottle_pose[:2] - taget_pose) < eps) and 0.2 < bottle_pose[2] < 0.7:
            reward += reward_step
        return reward

    def check_success(self):
        taget_pose = [-0.45, 0]
        eps = np.array([0.221, 0.325])
        bottle_pose = self.bottle.get_pose().p
        if np.all(np.abs(bottle_pose[:2] - taget_pose) < eps) and 0.2 < bottle_pose[2] < 0.7:
            return True
        return False