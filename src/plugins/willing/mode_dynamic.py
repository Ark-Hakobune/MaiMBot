import asyncio
import random
import time
import math
from typing import Dict
from loguru import logger
from ...common.database import db

from ..chat.config import global_config
from ..chat.chat_stream import ChatStream

# 全局意愿和概率倍数开关
# 最佳搭配是低意愿倍数和高概率倍数组合，会变得十分拟人。意愿过高容易变人机复读，概率过低容易变哑巴冷淡
WILLING_AMPLIFIER_ENABLED = True  # 回复意愿倍数开关
WILLING_AMPLIFIER_RATE = 0.8     # 回复意愿倍数，默认0.95   
WILLING_ADJUSTMENT = -0.15          # 回复意愿增减值，默认0.05，这一项是在最终的回复意愿基础上增加的，不能加太多
WILLING_RANGE_MIN = 0.1           # 回复意愿最小值，默认0.3
WILLING_RANGE_MAX = 1.1           # 回复意愿最大值，默认1.2

PROBABILITY_AMPLIFIER_ENABLED = True  # 回复概率倍数开关  
PROBABILITY_AMPLIFIER_RATE = 1.3      # 回复概率倍数，默认1.0
PROBABILITY_ADJUSTMENT = 0          # 回复概率增减值，默认0.0，这一项是在最终的回复概率基础上增加的

PROBABILITY_RANGE_MIN = 0.1         # 回复概率最小值，默认0.1 (10%)
PROBABILITY_RANGE_MAX = 1.1          # 回复概率最大值，默认1.1 (110%)

# 对话上下文增强设置
CONVERSATION_BOOST_ENABLED = True    # 对话上下文回复概率增强开关，基础回复概率是0.4，应用增强后会变为：0.4 × 1.5 + 0.15 = 0.75
CONVERSATION_BOOST_RATE = 1.5        # 对话上下文回复概率增强倍率，默认1.5
CONVERSATION_BOOST_ADJUSTMENT = 0.2 # 对话上下文回复概率增减值，默认0.15

# 表情包使用意愿开关
EMOJI_WILLING_ENABLED = True      # 表情包使用意愿开关
EMOJI_WILLING_RATE = 8.0          # 表情包使用意愿倍数，默认1.0
EMOJI_REPLY_RATE = 0.05            # 表情包回复概率为5%

# S曲线函数，用于平滑值的变化
def sigmoid(x, k=1):
    """返回一个介于0和1之间的S曲线值"""
    return 1 / (1 + math.exp(-k * x))

# 应用S曲线到值变化中
def apply_s_curve(current, target, factor=0.1):
    """平滑地将current值向target值靠近"""
    diff = target - current
    change = sigmoid(diff * 3) * abs(diff) * factor
    if diff < 0:
        change = -change
    return current + change

class WillingManager:
    def __init__(self):
        self.chat_reply_willing: Dict[str, float] = {}  # 存储每个聊天流的回复意愿
        self.chat_high_willing_mode: Dict[str, bool] = {}  # 存储每个聊天流是否处于高回复意愿期
        self.chat_msg_count: Dict[str, int] = {}  # 存储每个聊天流接收到的消息数量
        self.chat_last_mode_change: Dict[str, float] = {}  # 存储每个聊天流上次模式切换的时间
        self.chat_high_willing_duration: Dict[str, int] = {}  # 高意愿期持续时间(秒)
        self.chat_low_willing_duration: Dict[str, int] = {}  # 低意愿期持续时间(秒)
        self.chat_last_reply_time: Dict[str, float] = {}  # 存储每个聊天流上次回复的时间
        self.chat_last_sender_id: Dict[str, str] = {}  # 存储每个聊天流上次回复的用户ID
        self.chat_conversation_context: Dict[str, bool] = {}  # 标记是否处于对话上下文中
        self.chat_consecutive_replies: Dict[str, int] = {}  # 存储每个聊天流的连续回复次数
        self.chat_mode_transition: Dict[str, float] = {}  # 存储模式切换的过渡状态(0-1)
        self.chat_is_emoji_reply: Dict[str, bool] = {}  # 标记是否是对表情包的回复，如果是则只回复表情包
        
        # 消息历史记录 - 新功能3实现
        self.chat_message_history: Dict[str, list] = {}  # 存储每个聊天流的消息历史
        self.max_history_messages = 50  # 最大历史消息数量
        self.similarity_threshold = 0.8  # 相似度阈值，超过则认为是相似消息
        
        # 新增: 消息顺序记录 - 用于高回复期引用回复功能
        self.chat_message_sequence: Dict[str, list] = {}  # 存储每个聊天流的消息顺序(消息ID列表)
        self.force_quote_threshold = 5  # 当回复消息超过5条以前时，强制使用引用回复
        
        # 冷群检测相关属性
        self.group_activity: Dict[str, Dict] = {}  # 存储群组活跃度信息
        self.group_is_cold: Dict[str, bool] = {}  # 标记群组是否为冷群
        self.group_cold_recovery: Dict[str, float] = {}  # 存储冷群恢复状态(0-1)
        self.cold_group_check_interval = 300  # 冷群检测时间间隔(秒)，默认5分钟
        
        self._decay_task = None
        self._mode_switch_task = None
        self._cleanup_task = None  # 清理任务
        self._cold_group_check_task = None  # 冷群检测任务
        self._started = False

        #分享意愿
        self.daily_share_wiling: Dict[str, float] = {}
        
    async def _decay_reply_willing(self):
        """定期衰减回复意愿"""
        while True:
            await asyncio.sleep(5)
            for chat_id in list(self.chat_reply_willing.keys()):
                is_high_mode = self.chat_high_willing_mode.get(chat_id, False)
                current_willing = self.chat_reply_willing.get(chat_id, 0)
                transition = self.chat_mode_transition.get(chat_id, 1.0)
                
                # 计算目标意愿值 - 高回复期意愿中等，低回复期意愿高
                if is_high_mode:
                    # 高回复意愿期内的目标值更高
                    target_willing = random.uniform(0.5, 0.8)
                    # 应用S曲线平滑变化
                    new_willing = apply_s_curve(current_willing, target_willing, 0.1)
                    # 确保意愿在合理区间内
                    self.chat_reply_willing[chat_id] = max(0.5, min(1.2, new_willing))
                else:
                    # 低回复意愿期内保持高意愿
                    target_willing = random.uniform(0.7, 1.1)  # 高意愿，而不是过低
                    # 应用S曲线平滑变化
                    new_willing = apply_s_curve(current_willing, target_willing, 0.08)
                    # 确保意愿在合理区间内
                    self.chat_reply_willing[chat_id] = max(0.4, min(0.7, new_willing))
                
                # 应用全局意愿倍数和增减值（如果启用）
                if WILLING_AMPLIFIER_ENABLED:
                    # 首先应用倍数
                    self.chat_reply_willing[chat_id] *= WILLING_AMPLIFIER_RATE
                    # 然后应用增减值
                    self.chat_reply_willing[chat_id] += WILLING_ADJUSTMENT
                    # 最后确保意愿在全局设定的区间内
                    self.chat_reply_willing[chat_id] = max(WILLING_RANGE_MIN, min(WILLING_RANGE_MAX, self.chat_reply_willing[chat_id]))
                
    async def _mode_switch_check(self):
        """定期检查是否需要切换回复意愿模式"""
        while True:
            current_time = time.time()
            await asyncio.sleep(10)  # 每10秒检查一次

            for chat_id in list(self.chat_high_willing_mode.keys()):
                last_change_time = self.chat_last_mode_change.get(chat_id, 0)
                is_high_mode = self.chat_high_willing_mode.get(chat_id, False)

                # 获取当前模式的持续时间
                duration = 0
                if is_high_mode:
                    duration = self.chat_high_willing_duration.get(chat_id, 180)  # 使用已存储的持续时间或默认3分钟
                else:
                    duration = self.chat_low_willing_duration.get(chat_id, 300)  # 使用已存储的持续时间或默认5分钟
                
                # 检查是否需要切换模式
                if current_time - last_change_time > duration:
                    self._switch_willing_mode(chat_id)
                elif not is_high_mode and random.random() < 0.03:  # 降低随机切换概率到3%
                    # 低回复意愿期有小概率随机切换到高回复期
                    self._switch_willing_mode(chat_id)

                # 更新模式切换的过渡状态
                transition = self.chat_mode_transition.get(chat_id, 1.0)
                if is_high_mode and transition < 1.0:
                    # 平滑过渡到高回复期
                    self.chat_mode_transition[chat_id] = min(1.0, transition + 0.1)
                elif not is_high_mode and transition > 0.0:
                    # 平滑过渡到低回复期
                    self.chat_mode_transition[chat_id] = max(0.0, transition - 0.1)
                
                # 检查对话上下文状态是否需要重置
                last_reply_time = self.chat_last_reply_time.get(chat_id, 0)
                if current_time - last_reply_time > 300:  # 5分钟无交互，重置对话上下文
                    self.chat_conversation_context[chat_id] = False

                    # 重置连续回复计数
                    self.chat_consecutive_replies[chat_id] = 0
    
    def _switch_willing_mode(self, chat_id: str):
        """切换聊天流的回复意愿模式"""
        is_high_mode = self.chat_high_willing_mode.get(chat_id, False)
        
        # 初始化过渡状态
        if chat_id not in self.chat_mode_transition:
            self.chat_mode_transition[chat_id] = 1.0 if is_high_mode else 0.0

        if is_high_mode:
            # 从高回复期切换到低回复期
            self.chat_high_willing_mode[chat_id] = False
            # 不立即设置最低回复意愿，让它随着过渡状态平滑变化
            self.chat_low_willing_duration[chat_id] = random.randint(300, 600)  # 5-10分钟
            logger.debug(f"聊天流 {chat_id} 切换到低回复意愿期，持续 {self.chat_low_willing_duration[chat_id]} 秒")
        else:
            # 从低回复期切换到高回复期
            self.chat_high_willing_mode[chat_id] = True
            # 不立即设置最高回复意愿，让它随着过渡状态平滑变化
            self.chat_high_willing_duration[chat_id] = random.randint(300, 600)  # 5-10分钟
            logger.debug(f"聊天流 {chat_id} 切换到高回复意愿期，持续 {self.chat_high_willing_duration[chat_id]} 秒")

        self.chat_last_mode_change[chat_id] = time.time()
        self.chat_msg_count[chat_id] = 0  # 重置消息计数

    def get_willing(self, chat_stream: ChatStream) -> float:
        """获取指定聊天流的回复意愿"""
        stream = chat_stream
        if stream:
            return self.chat_reply_willing.get(stream.stream_id, 0)
        return 0

    def set_willing(self, chat_id: str, willing: float):
        """设置指定聊天流的回复意愿"""
        self.chat_reply_willing[chat_id] = willing

    def _ensure_chat_initialized(self, chat_id: str):
        """确保聊天流的所有数据已初始化"""
        current_time = time.time()
        
        if chat_id not in self.chat_reply_willing:

            self.chat_reply_willing[chat_id] = 0.3
        

        if chat_id not in self.chat_high_willing_mode:
            self.chat_high_willing_mode[chat_id] = False
            self.chat_last_mode_change[chat_id] = current_time
            self.chat_low_willing_duration[chat_id] = random.randint(300, 1200)  # 5-20分钟

        if chat_id not in self.chat_msg_count:
            self.chat_msg_count[chat_id] = 0

        if chat_id not in self.chat_conversation_context:
            self.chat_conversation_context[chat_id] = False

            
        if chat_id not in self.chat_consecutive_replies:
            self.chat_consecutive_replies[chat_id] = 0
            
        # 确保所有其他字典键也被初始化
        if chat_id not in self.chat_last_reply_time:
            self.chat_last_reply_time[chat_id] = 0
            
        if chat_id not in self.chat_last_sender_id:
            self.chat_last_sender_id[chat_id] = ""
            
        if chat_id not in self.chat_high_willing_duration:
            self.chat_high_willing_duration[chat_id] = random.randint(180, 240)  # 3-4分钟
            
        # 确保模式切换过渡状态被初始化
        if chat_id not in self.chat_mode_transition:
            is_high_mode = self.chat_high_willing_mode.get(chat_id, False)
            self.chat_mode_transition[chat_id] = 1.0 if is_high_mode else 0.0
            
        # 初始化表情包回复标记
        if chat_id not in self.chat_is_emoji_reply:
            self.chat_is_emoji_reply[chat_id] = False
        
    async def change_reply_willing_received(self, 
                                          chat_stream: ChatStream,
                                          topic: str = None,
                                          is_mentioned_bot: bool = False,
                                          config = None,
                                          is_emoji: bool = False,
                                          interested_rate: float = 0,
                                          sender_id: str = None,
                                          message_id: str = None) -> float:
        """根据消息内容改变回复意愿
        
        主要根据触发类型和关键词来改变回复意愿
        
        Args:
            chat_stream: 聊天流
            topic: 话题
            is_mentioned_bot: 是否@了bot
            config: 配置
            is_emoji: 是否是表情包
            interested_rate: 兴趣值
            sender_id: 发送者ID
            message_id: 消息ID，用于记录消息顺序

        Returns:
            float: 新的回复意愿
        """
        # 如果提供了消息ID，记录消息顺序
        if message_id:
            self.add_message_to_sequence(chat_stream, message_id)
            
        # 如果是表情包，设置表情包回复模式
        if is_emoji:
            self.set_emoji_reply_mode(chat_stream, True)
        else:
            self.set_emoji_reply_mode(chat_stream, False)
            

        # 获取或创建聊天流
        stream = chat_stream
        chat_id = stream.stream_id
        current_time = time.time()

        self._ensure_chat_initialized(chat_id)

        
        # 更新群组活跃度信息
        if chat_stream.group_info and sender_id:
            self._update_group_activity(chat_stream, sender_id)
        
        # 检查连续回复计数重置
        last_reply_time = self.chat_last_reply_time.get(chat_id, 0)
        if current_time - last_reply_time > 30:  # 30秒内没有新回复，重置连续回复计数
            self.chat_consecutive_replies[chat_id] = 0
            logger.debug(f"重置连续回复计数 - 聊天流 {chat_id}")

        # 增加消息计数
        self.chat_msg_count[chat_id] = self.chat_msg_count.get(chat_id, 0) + 1

        current_willing = self.chat_reply_willing.get(chat_id, 0)
        is_high_mode = self.chat_high_willing_mode.get(chat_id, False)
        transition = self.chat_mode_transition.get(chat_id, 1.0 if is_high_mode else 0.0)
        msg_count = self.chat_msg_count.get(chat_id, 0)
        in_conversation_context = self.chat_conversation_context.get(chat_id, False)

        consecutive_replies = self.chat_consecutive_replies.get(chat_id, 0)

        # 检查是否是对话上下文中的追问
        last_sender = self.chat_last_sender_id.get(chat_id, "")

        is_follow_up_question = False
        
        # 追问检测逻辑
        time_window = 240  # 扩展到4分钟
        max_msgs = 12      # 增加消息数量阈值
        
        # 1. 同一用户短时间内发送多条消息
        if sender_id and sender_id == last_sender and current_time - last_reply_time < time_window and msg_count <= max_msgs:
            is_follow_up_question = True
            in_conversation_context = True
            self.chat_conversation_context[chat_id] = True
            
            # 根据消息间隔动态调整回复意愿提升
            time_since_last = current_time - last_reply_time
            if time_since_last < 60:  # 1分钟内
                # 使用S曲线大幅增加回复意愿
                current_willing = apply_s_curve(current_willing, current_willing + 0.5, 0.6)
            elif time_since_last < 120:  # 1-2分钟
                # 使用S曲线中等增加回复意愿
                current_willing = apply_s_curve(current_willing, current_willing + 0.4, 0.5)
            else:
                # 使用S曲线小幅增加回复意愿
                current_willing = apply_s_curve(current_willing, current_willing + 0.2, 0.3)
                
            logger.debug(f"检测到追问 (同一用户), 提高回复意愿到 {current_willing:.2f}, 时间间隔: {time_since_last:.1f}秒")
            
        # 2. 即使不是同一用户，如果处于活跃对话中，也有可能是追问
        elif in_conversation_context and current_time - last_reply_time < time_window:
            # 处于活跃对话中，但不是同一用户，视为对话延续
            in_conversation_context = True
            # 对于对话延续，适度提高回复意愿
            time_since_last = current_time - last_reply_time
            if time_since_last < 90:  # 1.5分钟内
                current_willing = apply_s_curve(current_willing, current_willing + 0.15, 0.3)
                logger.debug(f"检测到对话延续 (不同用户), 轻微提高回复意愿到 {current_willing:.2f}")
            logger.debug(f"检测到对话延续 (不同用户), 保持对话上下文")
        
        # 特殊情况处理
        if is_mentioned_bot:
            # 被提及时立即切换到高回复期并增加回复意愿
            if not is_high_mode:
                self._switch_willing_mode(chat_id)
                self.chat_high_willing_mode[chat_id] = True
                self.chat_mode_transition[chat_id] = 1.0  # 立即完成过渡
                
            # 使用S曲线大幅增加回复意愿    
            current_willing = apply_s_curve(current_willing, 1.2, 0.8)
            in_conversation_context = True
            self.chat_conversation_context[chat_id] = True

            # 被提及时重置连续回复计数，允许新的对话开始
            self.chat_consecutive_replies[chat_id] = 0
            logger.debug(f"被提及, 当前意愿: {current_willing}, 重置连续回复计数, 切换到高回复期")
        
        # 图片消息处理
        if is_emoji:
            # 标记为表情包回复，这样后续处理可以知道只能回复表情包
            self.chat_is_emoji_reply[chat_id] = True
            logger.debug(f"表情包消息, 标记为只回复表情包")
            
            # 对表情包，固定回复概率为20%
            reply_probability = EMOJI_REPLY_RATE
            
            # 被提及时100%回复
            if is_mentioned_bot:
                reply_probability = 1.0
                logger.debug(f"被提及, 设置100%回复概率")
                
            # 记录当前发送者ID以便后续追踪
            if sender_id:
                self.chat_last_sender_id[chat_id] = sender_id
                
            # 保持标准的回复意愿更新逻辑（供其他功能使用）
            if is_high_mode:
                current_willing *= 0.5  # 基础降低为50%
            else:
                current_willing *= 0.8  # 低回复期降低得更少
                
            # 应用表情包使用意愿倍数（如果启用）
            if EMOJI_WILLING_ENABLED:
                current_willing *= EMOJI_WILLING_RATE
                
            # 确保表情包消息的回复意愿不会太低
            current_willing = max(current_willing, 0.1)
            
            # 更新回复意愿
            self.chat_reply_willing[chat_id] = min(max(current_willing, 0.3), 1.2)
            
            return reply_probability
        
        # 根据话题兴趣度适当调整
        if interested_rate > 0.5:
            # 使用S曲线增加回复意愿
            interest_boost = (interested_rate - 0.5) * 0.5
            current_willing = apply_s_curve(current_willing, current_willing + interest_boost, 0.3)
        
        # 确保意愿值在合理的区间内
        current_willing = max(0.3, min(1.2, current_willing))
            
        # 根据当前模式和过渡状态计算回复概率 - 这是主要控制回复率的因素
        base_probability = 0.0

        if in_conversation_context:
            # 在对话上下文中的基础概率，考虑过渡状态
            high_prob = 0.5  # 高回复期基础概率提高
            low_prob = 0.2   # 低回复期基础概率降低
            base_probability = low_prob + transition * (high_prob - low_prob)
            logger.debug(f"处于对话上下文中，基础回复概率: {base_probability:.2f}, 过渡状态: {transition:.2f}")
            
            # 检查是否为追问，如果是追问，则提高基础概率
            if is_follow_up_question:
                base_probability = min(0.75, base_probability * 1.3)  # 追问时基础概率至少提高30%，最高到0.75
                logger.debug(f"检测到追问，提高基础回复概率到: {base_probability:.2f}")
            
            # 应用对话上下文增强效果
            if CONVERSATION_BOOST_ENABLED:
                # 记录原始概率用于日志
                original_prob = base_probability
                # 应用倍率和增减值
                base_probability = base_probability * CONVERSATION_BOOST_RATE + CONVERSATION_BOOST_ADJUSTMENT
                # 追问时额外提高增强效果
                if is_follow_up_question:
                    base_probability += 0.1  # 追问时额外增加10%的概率
                logger.debug(f"应用对话上下文增强: {original_prob:.2f} → {base_probability:.2f} (倍率:{CONVERSATION_BOOST_RATE}, 增减:{CONVERSATION_BOOST_ADJUSTMENT})")
        else:
            # 根据消息计数和当前模式设置基础概率
            high_prob = 0.85 if 1 <= msg_count <= 3 else 0.4   # 高回复期概率提高
            low_prob = 0.40 if msg_count >= 15 else 0.03 * min(msg_count, 10)  # 低回复期概率降低
            # 应用过渡状态平滑变化
            base_probability = low_prob + transition * (high_prob - low_prob)
        
        # 应用S曲线对基础概率进行调整
        base_probability = sigmoid((base_probability - 0.5) * 4) * 0.8 + 0.1
        
        # 确保基础概率不会太低
        base_probability = max(base_probability, 0.05)
            
        # 考虑回复意愿的影响，使用S曲线组合概率和意愿
        willing_factor = sigmoid((current_willing - 0.7) * 2) * 0.4 + 0.6  # 降低意愿对最终概率的影响
        reply_probability = base_probability * willing_factor
        
        # 根据连续回复次数调整概率
        if consecutive_replies >= 4:
            reply_probability *= 0.01  # 连续回复4次或以上，降低到1%
            logger.debug(f"连续回复次数 >= 4, 降低回复概率到1%")
        elif consecutive_replies >= 3:
            reply_probability *= 0.1   # 连续回复3次，降低到10%
            logger.debug(f"连续回复次数 = 3, 降低回复概率到10%")
        elif consecutive_replies >= 2:
            reply_probability *= 0.5   # 连续回复2次，降低到50%
            logger.debug(f"连续回复次数 = 2, 降低回复概率到50%")
        
        # 检查是否为冷群，提高冷群的回复概率
        if chat_stream.group_info:
            group_id = self._get_group_id_from_chat_id(chat_id)
            is_cold_group = self.group_is_cold.get(group_id, False)
            
            if is_cold_group:
                # 冷群中提高回复概率
                recovery_state = self.group_cold_recovery.get(group_id, 0.0)
                # 根据恢复状态调整倍数
                boost_factor = 3.0 - recovery_state * 2.0  # 从3.0逐渐降低到1.0
                reply_probability = min(reply_probability * boost_factor, 0.8)
                logger.debug(f"检测到冷群 {group_id}，提高回复概率到: {reply_probability:.2f}, 恢复状态: {recovery_state:.2f}")
        
        # 检查群组权限（如果是群聊）
        if chat_stream.group_info and config:
            if chat_stream.group_info.group_id in config.talk_frequency_down_groups:
                reply_probability = reply_probability / global_config.down_frequency_rate

        # 应用概率放大系数和增减值
        if PROBABILITY_AMPLIFIER_ENABLED:
            # 首先应用倍数
            reply_probability *= PROBABILITY_AMPLIFIER_RATE
            # 然后应用增减值
            reply_probability += PROBABILITY_ADJUSTMENT
            
        # 限制回复概率在全局设定的范围内
        reply_probability = max(PROBABILITY_RANGE_MIN, min(PROBABILITY_RANGE_MAX, reply_probability))
        
        # 对于追问和被提及，设置更高回复概率
        if is_follow_up_question:
            # 追问时显著提高回复概率，根据时间间隔动态调整
            time_since_last = current_time - last_reply_time
            if time_since_last < 60:  # 1分钟内
                # 短时间内的追问给予更高回复概率（最高70%）
                follow_up_prob = 0.7
            elif time_since_last < 120:  # 1-2分钟
                # 中等时间的追问给予中等回复概率（50%）
                follow_up_prob = 0.5
            else:  # 2-3分钟
                # 较长时间的追问给予基础回复概率（40%）
                follow_up_prob = 0.4
                
            # 使用平滑过渡，确保追问概率不低于计算出的值
            reply_probability = max(reply_probability, follow_up_prob)
            logger.debug(f"追问响应: 将回复概率提高到 {reply_probability:.2f}, 时间间隔: {time_since_last:.1f}秒")
        
        # 被提及时100%回复
        if is_mentioned_bot:
            reply_probability = 1.0
            logger.debug(f"被提及, 设置100%回复概率")
        
        # 记录当前发送者ID以便后续追踪
        if sender_id:
            self.chat_last_sender_id[chat_id] = sender_id
            
        # 最终限制回复意愿范围
        self.chat_reply_willing[chat_id] = min(max(current_willing, WILLING_RANGE_MIN), WILLING_RANGE_MAX)
        
        return reply_probability

    def change_reply_willing_sent(self, chat_stream: ChatStream):
        """开始思考后降低聊天流的回复意愿"""
        stream = chat_stream
        if stream:
            chat_id = stream.stream_id
            self._ensure_chat_initialized(chat_id)
            current_willing = self.chat_reply_willing.get(chat_id, 0)

            # 增加连续回复计数
            self.chat_consecutive_replies[chat_id] = self.chat_consecutive_replies.get(chat_id, 0) + 1
            logger.debug(f"增加连续回复计数到 {self.chat_consecutive_replies[chat_id]} - 聊天流 {chat_id}")
            
            # 回复后减少回复意愿
            self.chat_reply_willing[chat_id] = max(0.0, current_willing - 0.3)

            # 标记为对话上下文中
            self.chat_conversation_context[chat_id] = True

            # 记录最后回复时间
            self.chat_last_reply_time[chat_id] = time.time()

            # 重置消息计数
            self.chat_msg_count[chat_id] = 0
    def change_reply_willing_not_sent(self, chat_stream: ChatStream):
        """决定不回复后提高聊天流的回复意愿"""
        stream = chat_stream
        if stream:
            chat_id = stream.stream_id
            self._ensure_chat_initialized(chat_id)
            is_high_mode = self.chat_high_willing_mode.get(chat_id, False)
            current_willing = self.chat_reply_willing.get(chat_id, 0)
            in_conversation_context = self.chat_conversation_context.get(chat_id, False)

            # 根据当前模式调整不回复后的意愿增加
            if is_high_mode:
                willing_increase = 0.1
            elif in_conversation_context:
                # 在对话上下文中但决定不回复，小幅增加回复意愿
                willing_increase = 0.15
            else:
                willing_increase = random.uniform(0.05, 0.1)

            self.chat_reply_willing[chat_id] = min(2.0, current_willing + willing_increase)

    def change_reply_willing_after_sent(self, chat_stream: ChatStream):
        """发送消息后提高聊天流的回复意愿"""
        # 由于已经在sent中处理，这个方法保留但不再需要额外调整
        pass

    async def _cleanup_inactive_chats(self):
        """定期清理长时间不活跃的聊天流数据"""
        while True:
            await asyncio.sleep(3600)  # 每小时执行一次清理
            current_time = time.time()
            inactive_threshold = 86400  # 24小时不活跃的聊天流将被清理
            
            # 收集需要清理的聊天流ID
            to_clean = []
            
            for chat_id in list(self.chat_last_reply_time.keys()):
                last_active = self.chat_last_reply_time.get(chat_id, 0)
                if current_time - last_active > inactive_threshold:
                    to_clean.append(chat_id)
            
            # 从所有字典中移除不活跃的聊天流
            for chat_id in to_clean:
                self._remove_chat_data(chat_id)
                
            if to_clean:
                logger.debug(f"已清理 {len(to_clean)} 个不活跃的聊天流数据")
    
    def _remove_chat_data(self, chat_id: str):
        """清理聊天流数据"""
        if chat_id in self.chat_reply_willing:
            del self.chat_reply_willing[chat_id]
        if chat_id in self.chat_high_willing_mode:
            del self.chat_high_willing_mode[chat_id]
        if chat_id in self.chat_msg_count:
            del self.chat_msg_count[chat_id]
        if chat_id in self.chat_last_mode_change:
            del self.chat_last_mode_change[chat_id]
        if chat_id in self.chat_high_willing_duration:
            del self.chat_high_willing_duration[chat_id]
        if chat_id in self.chat_low_willing_duration:
            del self.chat_low_willing_duration[chat_id]
        if chat_id in self.chat_last_reply_time:
            del self.chat_last_reply_time[chat_id]
        if chat_id in self.chat_consecutive_replies:
            del self.chat_consecutive_replies[chat_id]
        if chat_id in self.chat_mode_transition:
            del self.chat_mode_transition[chat_id]
        if chat_id in self.chat_conversation_context:
            del self.chat_conversation_context[chat_id]
        if chat_id in self.chat_is_emoji_reply:
            del self.chat_is_emoji_reply[chat_id]
        if chat_id in self.chat_message_history:
            del self.chat_message_history[chat_id]
        # 新增: 清理消息序列数据
        if chat_id in self.chat_message_sequence:
            del self.chat_message_sequence[chat_id]
                
        # 尝试清理相关的群组数据
        try:
            group_id = self._get_group_id_from_chat_id(chat_id)
            if group_id in self.group_activity and len(self.group_activity[group_id]["active_users"]) <= 1:
                # 如果只有一个活跃用户，可能就是这个被清理的聊天，整个清理群组数据
                self.group_activity.pop(group_id, None)
                self.group_is_cold.pop(group_id, None)
                self.group_cold_recovery.pop(group_id, None)
        except Exception as e:
            logger.error(f"尝试清理群组数据时出错: {e}")
                
        logger.debug(f"已移除聊天流 {chat_id} 的所有数据")
        
    async def stop(self):
        """停止所有异步任务"""
        if self._decay_task and not self._decay_task.done():
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
            
        if self._mode_switch_task and not self._mode_switch_task.done():
            self._mode_switch_task.cancel()
            try:
                await self._mode_switch_task
            except asyncio.CancelledError:
                pass
                
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        if self._cold_group_check_task and not self._cold_group_check_task.done():
            self._cold_group_check_task.cancel()
            try:
                await self._cold_group_check_task
            except asyncio.CancelledError:
                pass
                
        self._started = False
        logger.debug("已停止所有WillingManager任务")
        
    async def ensure_started(self):
        """确保所有任务已启动"""
        if not self._started:
            if self._decay_task is None:
                self._decay_task = asyncio.create_task(self._decay_reply_willing())
            if self._mode_switch_task is None:
                self._mode_switch_task = asyncio.create_task(self._mode_switch_check())
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._cleanup_inactive_chats())
            if self._cold_group_check_task is None:
                self._cold_group_check_task = asyncio.create_task(self._cold_group_check())
            self._started = True
            logger.debug("WillingManager所有任务已启动")

    def _get_group_id_from_chat_id(self, chat_id: str) -> str:
        """从聊天流ID提取群组ID，如果失败则返回原ID"""
        # 通常聊天流ID中会包含群组ID信息
        # 根据实际格式进行提取，这里假设格式为 platform:group_id:user_id
        try:
            parts = chat_id.split(':')
            if len(parts) >= 2:
                return f"{parts[0]}:{parts[1]}"  # platform:group_id作为群组标识
            return chat_id
        except Exception:
            return chat_id
    
    def _update_group_activity(self, chat_stream: ChatStream, sender_id: str = None):
        """更新群组活跃度信息"""
        if not chat_stream.group_info or not sender_id:
            return
            
        current_time = time.time()
        group_id = chat_stream.group_info.group_id
        
        # 初始化群组活跃度信息
        if group_id not in self.group_activity:
            self.group_activity[group_id] = {
                "message_count": 0,
                "active_users": set(),
                "last_message_time": current_time,
                "check_start_time": current_time,
                "last_5min_messages": [],  # 存储最近5分钟的消息记录
                "last_5min_users": set()   # 存储最近5分钟的活跃用户
            }
            
        # 更新群组活跃度信息
        activity = self.group_activity[group_id]
        activity["message_count"] += 1
        activity["active_users"].add(sender_id)
        activity["last_message_time"] = current_time
        
        # 更新5分钟窗口的消息记录
        activity["last_5min_messages"].append({
            "time": current_time,
            "user_id": sender_id
        })
        activity["last_5min_users"].add(sender_id)
        
        # 清理超过5分钟的消息记录
        five_minutes_ago = current_time - 300
        old_messages = []
        old_users = set()
        
        # 找出需要移除的旧消息
        for msg in activity["last_5min_messages"]:
            if msg["time"] < five_minutes_ago:
                old_messages.append(msg)
                old_users.add(msg["user_id"])
                
        # 移除旧消息
        for msg in old_messages:
            activity["last_5min_messages"].remove(msg)
            
        # 更新5分钟内的活跃用户集合
        if old_users:
            # 重新计算活跃用户列表
            current_users = set()
            for msg in activity["last_5min_messages"]:
                current_users.add(msg["user_id"])
            activity["last_5min_users"] = current_users
            
    def _check_cold_group(self, group_id: str):
        """检查群组是否为冷群，并更新状态"""
        if group_id not in self.group_activity:
            return
            
        activity = self.group_activity[group_id]
        current_time = time.time()
        
        # 分析最近5分钟的活跃度
        five_min_msg_count = len(activity["last_5min_messages"])
        five_min_users_count = len(activity["last_5min_users"])
        
        # 冷群判定标准：5分钟内发言人数少于8人，且发言次数少于30次
        is_cold = (five_min_users_count < 8 and five_min_msg_count < 30)
        was_cold = self.group_is_cold.get(group_id, False)
        
        # 更新冷群状态
        self.group_is_cold[group_id] = is_cold
        
        # 初始化或更新冷群恢复状态
        if group_id not in self.group_cold_recovery:
            self.group_cold_recovery[group_id] = 0.0
            
        # 如果群状态从冷变热，开始恢复过程
        if was_cold and not is_cold:
            # 开始恢复状态，设为0.0 (完全冷群状态)
            self.group_cold_recovery[group_id] = 0.0
            logger.debug(f"群 {group_id} 从冷群开始恢复")
        # 如果是持续热群，完全恢复
        elif not was_cold and not is_cold:
            # 逐渐增加恢复状态，使用S曲线平滑变化
            current_recovery = self.group_cold_recovery.get(group_id, 0.0)
            if current_recovery < 1.0:
                new_recovery = apply_s_curve(current_recovery, 1.0, 0.1)
                self.group_cold_recovery[group_id] = min(1.0, new_recovery)
        # 如果变回冷群，重置恢复状态
        elif not was_cold and is_cold:
            self.group_cold_recovery[group_id] = 0.0
            
        logger.debug(f"群 {group_id} 活跃度检查: 5分钟消息数={five_min_msg_count}, 5分钟活跃用户数={five_min_users_count}, 判定为{'冷群' if is_cold else '活跃群'}, 恢复状态={self.group_cold_recovery.get(group_id, 0.0):.2f}")
        
    async def _cold_group_check(self):
        """定期检查所有群组的冷热状态"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查一次
            current_time = time.time()
            
            for group_id in list(self.group_activity.keys()):
                # 每分钟都执行一次冷群检测
                self._check_cold_group(group_id)
                    
            # 清理太久没活动的群组记录
            self._cleanup_inactive_groups(current_time)
                    
    def _cleanup_inactive_groups(self, current_time: float):
        """清理长时间不活跃的群组记录"""
        inactive_threshold = 86400 * 3  # 3天不活跃则清理
        inactive_groups = []
        
        for group_id, activity in list(self.group_activity.items()):
            if current_time - activity["last_message_time"] > inactive_threshold:
                inactive_groups.append(group_id)
                
        for group_id in inactive_groups:
            if group_id in self.group_activity:
                self.group_activity.pop(group_id)
            if group_id in self.group_is_cold:
                self.group_is_cold.pop(group_id)
                
        if inactive_groups:
            logger.debug(f"已清理 {len(inactive_groups)} 个不活跃的群组记录")

    def is_emoji_reply(self, chat_stream: ChatStream) -> bool:
        """判断当前是否应该只回复表情包 - 修改实现功能2"""
        chat_id = chat_stream.stream_id
        self._ensure_chat_initialized(chat_id)
        
        # 强制检查是否是表情包回复模式
        if self.chat_is_emoji_reply.get(chat_id, False):
            return True
            
        # 随机判断是否回复表情包
        if EMOJI_WILLING_ENABLED and random.random() < EMOJI_REPLY_RATE:
            self.chat_is_emoji_reply[chat_id] = True
            return True
            
        return False
        
    def set_emoji_reply_mode(self, chat_stream: ChatStream, is_emoji_reply: bool):
        """设置表情包回复模式
        
        Args:
            chat_stream: 聊天流
            is_emoji_reply: 是否是表情包回复模式
        """
        chat_id = chat_stream.stream_id
        self._ensure_chat_initialized(chat_id)
        self.chat_is_emoji_reply[chat_id] = is_emoji_reply
        if is_emoji_reply:
            logger.info(f"设置表情包回复模式: {chat_id}")
        
    def need_quote_reply(self, chat_stream: ChatStream) -> bool:
        """判断是否需要引用回复
        
        Returns:
            bool: 是否需要引用回复
        """
        chat_id = chat_stream.stream_id
        self._ensure_chat_initialized(chat_id)
        is_high_mode = self.chat_high_willing_mode.get(chat_id, False)
        
        # 如果不是高回复期，则按原先的逻辑处理(低回复期通常不需要引用回复)
        if not is_high_mode:
            return False
            
        return False  # 默认不需要引用回复，在 should_force_quote_reply 中会进行实际判断

    def should_force_quote_reply(self, chat_stream: ChatStream, reply_message_id: str) -> bool:
        """判断是否需要强制引用回复
        
        Args:
            chat_stream: 聊天流
            reply_message_id: 正在回复的消息ID
            
        Returns:
            bool: 是否需要强制引用回复
        """
        chat_id = chat_stream.stream_id
        self._ensure_chat_initialized(chat_id)
        
        # 如果消息序列为空，则不需要引用回复
        if not self.chat_message_sequence.get(chat_id, []):
            return False
            
        # 获取回复消息在序列中的位置
        message_sequence = self.chat_message_sequence.get(chat_id, [])
        
        try:
            reply_index = message_sequence.index(reply_message_id)
            current_index = len(message_sequence) - 1
            message_distance = current_index - reply_index
            
            # 如果回复的消息距离当前消息超过阈值，则强制使用引用回复
            if message_distance >= self.force_quote_threshold:
                logger.info(f"高回复期引用回复: 消息间隔{message_distance}条，强制使用引用回复")
                return True
        except ValueError:
            # 消息ID不在序列中，可能是旧消息
            logger.warning(f"消息ID {reply_message_id} 不在序列中，无法判断是否强制引用回复")
            
        return False

    def add_message_to_sequence(self, chat_stream: ChatStream, message_id: str) -> None:
        """添加消息ID到序列
        
        Args:
            chat_stream: 聊天流
            message_id: 消息ID
        """
        chat_id = chat_stream.stream_id
        self._ensure_chat_initialized(chat_id)
        
        # 初始化消息序列
        if chat_id not in self.chat_message_sequence:
            self.chat_message_sequence[chat_id] = []
            
        # 添加消息ID到序列
        message_sequence = self.chat_message_sequence[chat_id]
        message_sequence.append(message_id)
        
        # 限制序列长度
        max_sequence_length = self.force_quote_threshold * 4  # 保持合理长度
        if len(message_sequence) > max_sequence_length:
            self.chat_message_sequence[chat_id] = message_sequence[-max_sequence_length:]

    def is_image_only_reply(self, chat_stream: ChatStream) -> bool:
        """判断是否只需要回复图片/表情包
        
        如果接收到的是纯图片或表情包消息，则只回复表情包
        
        Returns:
            bool: 是否只回复表情包
        """
        chat_id = chat_stream.stream_id
        self._ensure_chat_initialized(chat_id)
        
        # 获取是否为表情包回复模式
        is_emoji_reply = self.chat_is_emoji_reply.get(chat_id, False)
        
        return is_emoji_reply

    def add_message_to_history(self, chat_stream: ChatStream, message: str) -> None:
        """添加消息到历史记录 - 新功能3实现"""
        chat_id = chat_stream.stream_id
        if chat_id not in self.chat_message_history:
            self.chat_message_history[chat_id] = []
            
        # 添加消息到历史记录
        self.chat_message_history[chat_id].append(message)
        
        # 保持历史记录不过长
        if len(self.chat_message_history[chat_id]) > self.max_history_messages:
            self.chat_message_history[chat_id] = self.chat_message_history[chat_id][-self.max_history_messages:]
            
    def is_similar_to_previous_message(self, chat_stream: ChatStream, message: str) -> bool:
        """检查消息是否与历史消息相似 - 新功能3实现"""
        chat_id = chat_stream.stream_id
        if chat_id not in self.chat_message_history:
            return False
            
        # 简单相似度检查 - 可以根据需要替换为更复杂的算法
        for prev_msg in self.chat_message_history[chat_id]:
            # 如果消息完全相同
            if prev_msg == message:
                return True
                
            # 如果消息长度相近且有高度重叠
            if len(prev_msg) > 10 and len(message) > 10:
                # 计算Jaccard相似度
                if self._calculate_similarity(prev_msg, message) > self.similarity_threshold:
                    return True
                    
        return False
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度 - 新功能3实现"""
        # 分词
        words1 = set(text1)
        words2 = set(text2)
        
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # 避免除零错误
        if union == 0:
            return 0.0
            
        return intersection / union


    async def check_daily_share_wiling(self, chat_stream: ChatStream) -> float:
        """检查群聊消息活跃度，决定是否提高分享日常意愿
            目前仅支持群聊主动发起聊天，私聊不支持
        Args:
            chat_stream: 聊天流对象, 分享意愿不依赖聊天内容，只与群聊活跃度相关

        Returns:
            float: 分享意愿值
        """
        if not chat_stream or not chat_stream.group_info:
            return 0.0

        try:
            # 获取24小时前的时间戳
            one_day_ago = int(time.time()) - 86400  # 24小时 = 86400秒

            # 从数据库查询24小时内的总消息数
            daily_messages = list(db.messages.find({
                "chat_id": chat_stream.stream_id,
                "time": {"$gte": one_day_ago}
            }))

            # 如果24小时内消息数不超过100，不激活分享意愿
            # 仅统计bot运行期间的消息
            # 暂时不启用，配置文档指定群聊分享功能是否开启
            # if len(daily_messages) <= 100:
            #     logger.debug(f"群{chat_stream.group_info.group_id}在24小时内消息数{len(daily_messages)}条，小于阈值，鉴定为不活跃群，不激活分享意愿")
            #     self.daily_share_wiling[chat_stream.stream_id] = 0
            #     return 0.0

            # 获取60分钟前的时间戳
            thirty_minutes_ago = int(time.time()) - 36

            # 从数据库查询最近60分钟的消息
            recent_messages = list(db.messages.find({
                "chat_id": chat_stream.stream_id,
                "time": {"$gte": thirty_minutes_ago}
            }))

            # 如果没有最近消息，大幅提高分享意愿
            if not recent_messages:
                share_willing = self.daily_share_wiling.get(chat_stream.stream_id, global_config.daily_share_willing)
                new_willing = min(0.6, max(0.3, share_willing + 0.2))
                self.daily_share_wiling[chat_stream.stream_id] = new_willing
                logger.info(f"群{chat_stream.group_info.group_id}最近60分钟无消息，提高分享意愿至{new_willing}")
                return new_willing

            # 如果有消息，但消息数量很少（比如少于3条），适度提高意愿
            if len(recent_messages) < 3:
                share_willing = self.daily_share_wiling.get(chat_stream.stream_id, global_config.daily_share_willing)
                new_willing = min(0.4, max(0.2, share_willing + 0.1))
                self.daily_share_wiling[chat_stream.stream_id] = new_willing
                logger.info(f"群{chat_stream.group_info.group_id}最近60分钟消息较少，适度提高分享意愿至{new_willing}")
                return new_willing

            # 消息活跃度正常，保持当前意愿
            logger.debug(
                f"群{chat_stream.group_info.group_id}消息活跃度正常，保持分享意愿{self.daily_share_wiling.get(chat_stream.stream_id, global_config.daily_share_willing)}")
            return self.daily_share_wiling.get(chat_stream.stream_id, global_config.daily_share_willing)

        except Exception as e:
            logger.error(f"检查群聊活跃度时出错: {str(e)}")
            return global_config.daily_share_willing  # 出错时返回基础分享意愿

    async def reset_daily_share_wiling(self, chat_stream: ChatStream) -> float:
        """重置分享意愿"""
        self.daily_share_wiling[chat_stream.stream_id] = 0

# 创建全局实例
willing_manager = WillingManager()
