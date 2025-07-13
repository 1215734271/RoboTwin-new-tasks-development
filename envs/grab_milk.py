from ._base_task import Base_Task   # 基础环境
from .utils import *  # 工具包整体
import sapien  # 物理引擎
import math
from ._GLOBAL_CONFIGS import *  # 全局配置
from copy import deepcopy  # 深拷贝功能


class grab_milk(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)       # 调用基类并初始化

    def load_actors(self):
        # 四元数表示可能的初始旋转姿态
        ori_qpos = [[0.707, 0.707, 0, 0],[0.707, 0.707, 0, 0],[0.650, 0.650, 0.279, 0.279]]
        self.model_id = np.random.choice([0,1,2],1)[0]         # 随机模型版本选取
        milk_pose = rand_pose(
            xlim=[-0.1, 0.1],
            ylim=[-0.05, 0.05],
            qpos=ori_qpos[self.model_id],
            rotate_rand=True,
            rotate_lim=[0, 0.8, 0],
        )            # 随机位姿
        self.milk = create_actor(
            scene=self,
            pose=milk_pose,
            modelname="038_milk-box",
            convex=True,
            model_id=self.model_id,
        )    # 生成模型
        self.add_prohibit_area(self.milk, padding=0.1)        # 添加禁区

    def play_once(self):
        milk_pose = self.milk.get_pose().p
        arm_tag = ArmTag("left" if milk_pose[0] < 0 else "right")   # 根据模型位置启用左右臂

        self.move(self.grasp_actor(self.milk, arm_tag=arm_tag, pre_grasp_dis=0.1, gripper_pos=0))  # 抓取

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))   # 抬升高度0.1米

        self.info["info"] = {"{A}": f"102_roller/base{self.model_id}"}

        return self.info         # 记录数据

    def check_success(self):
        milk_pose = self.milk.get_pose().p      # 获取牛奶盒当前位置
        return (
                (self.is_left_gripper_close() or self.is_right_gripper_close())
                and milk_pose[2] > 0.81
        )           #  成功条件为左或右爪夹闭合且牛奶盒高度不低于0.81米









