import random
import time
import asyncio
import re
from typing import Dict, List, Set, Tuple, Optional
from nonebot import get_bot, get_driver
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent

# 全局配置
TOPIC_FREQUENCY_RANGE = (1, 2)  # 话题提取频率范围（分钟）
IMAGE_FORWARD_PROBABILITY = 0.8  # 图片转发概率，1.0为只转图片不发文字，0为只发话题不转图片
BLOCKED_SOURCE_GROUPS = []  # 屏蔽的来源群，用英文逗号隔开
BLOCKED_TARGET_GROUPS = []  # 屏蔽的目标群
PRIORITIZED_QQ_NUMBERS = []  # 优先转发的QQ号
PRIORITY_BOOST_FACTOR = 3.0  # 优先QQ号的图片转发概率提升倍数
MAX_IMAGES_TO_FORWARD = 3  # 最大转发图片数
IMAGE_TIME_WINDOW = 10  # 连续图片时间窗口（秒）指的是多少秒之内的同一人发的图片才转发

# 多群转发设置
MULTI_GROUP_FORWARD = True  # 是否启用多群转发
MAX_TARGET_GROUPS = 1  # 最大目标群数量，仅在MULTI_GROUP_FORWARD=True时生效

# 图片内容过滤条件
FILTERED_IMAGE_TOPICS = [
    # 成人/限制级内容
    "色情", "裸露", "不适当内容",
    # 游戏相关
    "游戏", "游戏截图",
]

from ..models.utils_model import LLM_request
from .config import global_config
from .message_base import UserInfo, GroupInfo, Seg
from .message import MessageSending, MessageSet
from .message_sender import message_manager
from .chat_stream import chat_manager
from .utils import calculate_typing_time, get_recent_group_detailed_plain_text
from src.common.logger import get_module_logger, LogConfig, TOPIC_STYLE_CONFIG

# 定义日志配置
topic_config = LogConfig(
    # 使用专用样式
    console_format=TOPIC_STYLE_CONFIG["console_format"],
    file_format=TOPIC_STYLE_CONFIG["file_format"],
)

logger = get_module_logger("topic_identifier", config=topic_config)

driver = get_driver()
config = driver.config

class TopicIdentifier:
    def __init__(self):

        """初始化话题识别器"""
        # LLM 实例
        self.llm_topic_judge = LLM_request(model=global_config.llm_topic_judge, request_type='topic')
        self.response_llm = LLM_request(model=global_config.llm_normal)
        
        # 群组状态管理
        self.active_topics: Dict[int, str] = {}
        self.bot_active_groups: Set[int] = set()
        self.group_last_active_time: Dict[int, float] = {}
        self.group_has_image: Dict[int, List[str]] = {}
        self.image_sender_info: Dict[int, Dict[str, Dict]] = {}
        
        # 任务管理
        self._running = False
        self._task = None
        self.last_extraction_time = time.time()
        
        # 图片缓存 - 新增
        self.image_cache: Dict[str, str] = {}  # 存储图片URL到base64数据的映射
        
        logger.info("话题识别器初始化完成")

    async def identify_topic_llm(self, text: str) -> Optional[List[str]]:
        """识别消息主题，返回主题列表"""
        prompt = f"""判断这条消息的主题，如果没有明显主题请回复"无主题"，要求：
                1. 主题通常2-4个字，必须简短，要求精准概括，不要太具体。
                2. 建议给出多个主题，之间用英文逗号分割。只输出主题本身就好，不要有前后缀。
                消息内容：{text}"""


        topic, _ = await self.llm_topic_judge.generate_response(prompt)


        if not topic:
            logger.error("LLM API 返回为空")
            return None

        if not topic or topic == "无主题":
            return None

        topic_list = [t.strip() for t in topic.split(",") if t.strip()]
        logger.info(f"主题: {topic_list}")
        return topic_list if topic_list else None

    async def start_monitoring(self):
        """启动话题监控"""
        if self._running:
            logger.warning("话题监控已经在运行中")
            return
            
        self._running = True
        logger.info("话题监控启动成功 - 将定期在群组间转发有趣的话题")
        
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.success("话题监控任务已启动")
    
    async def stop_monitoring(self):
        """停止话题监控"""
        if not self._running:
            logger.warning("话题监控尚未启动，无法停止")
            return
            
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("话题监控已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        logger.info("话题监控循环开始运行")
        
        while self._running:
            try:
                min_interval = TOPIC_FREQUENCY_RANGE[0] * 60
                max_interval = TOPIC_FREQUENCY_RANGE[1] * 60
                interval = random.randint(min_interval, max_interval)
                
                logger.info(f"下一次话题提取将在 {interval} 秒后进行")
                await asyncio.sleep(interval)
                
                logger.info("开始执行话题提取与转发任务")
                await self.extract_and_forward_topic()
                
            except asyncio.CancelledError:
                logger.warning("话题监控循环被取消")
                break
            except Exception as e:
                logger.error(f"话题监控循环出错: {e}")
                await asyncio.sleep(60)

    async def extract_and_forward_topic(self):
        """提取并转发话题"""
        try:
            bot = get_bot()
            group_list = await bot.get_group_list()
            
            if not group_list:
                logger.warning("群列表为空，无法执行话题提取")
                return
                
            group_ids = [group["group_id"] for group in group_list]
            
            if len(group_ids) < 2:
                logger.info("群组数量不足，无法进行话题转发")
                return
                
            allowed_groups = [gid for gid in group_ids if gid in global_config.talk_allowed_groups]
            if len(allowed_groups) < 2:
                logger.info("允许交流的群组数量不足，无法进行话题转发")
                return
            
            source_group_id = await self._select_source_group(allowed_groups)
            if not source_group_id:
                logger.info("没有找到合适的源群组")
                return
            
            topic_options, has_images, image_urls, image_sender_data = await self._extract_topic(source_group_id)
            if not topic_options and not (has_images and image_urls):
                logger.info(f"群 {source_group_id} 没有提取到有效话题或可用图片")
                return
            
            # 记录原始数据，避免多次转发时的数据混淆
            original_topic_options = topic_options.copy() if topic_options else None
            original_image_urls = image_urls.copy() if image_urls else []
            # 深度复制sender_data字典，避免共享引用导致的问题
            original_image_sender_data = {}
            if image_sender_data:
                for url, data in image_sender_data.items():
                    original_image_sender_data[url] = data.copy()
            
            # 根据配置决定是单群转发还是多群转发
            if MULTI_GROUP_FORWARD:
                # 多群转发模式
                target_group_ids = await self._select_multiple_target_groups(allowed_groups, source_group_id, MAX_TARGET_GROUPS)
                if not target_group_ids:
                    logger.info("没有找到合适的目标群组")
                    return
                
                logger.info(f"将内容转发到 {len(target_group_ids)} 个群组: {target_group_ids}")
                
                # 为每个目标群组创建独立的转发任务
                forward_tasks = []
                for target_group_id in target_group_ids:
                    # 为每个群组创建独立的数据副本
                    current_topic_options = original_topic_options.copy() if original_topic_options else None
                    if current_topic_options and len(current_topic_options) > 1:
                        random.shuffle(current_topic_options)
                    
                    current_image_urls = original_image_urls.copy()
                    current_image_sender_data = {}
                    if original_image_sender_data:
                        for url, data in original_image_sender_data.items():
                            current_image_sender_data[url] = data.copy()
                    
                    task = asyncio.create_task(
                        self._forward_topic(
                            current_topic_options, 
                            target_group_id, 
                            has_images, 
                            current_image_urls, 
                            current_image_sender_data
                        )
                    )
                    forward_tasks.append(task)
                
                # 等待所有转发任务完成
                if forward_tasks:
                    await asyncio.gather(*forward_tasks)
                    logger.success(f"成功完成向 {len(forward_tasks)} 个群组的内容转发")
            else:
                # 单群转发模式（原有逻辑）
                target_group_id = await self._select_target_group(allowed_groups, source_group_id)
                if not target_group_id:
                    logger.info("没有找到合适的目标群组")
                    return
                
                await self._forward_topic(original_topic_options, target_group_id, has_images, original_image_urls, original_image_sender_data)
            
        except Exception as e:
            logger.error(f"话题提取与转发过程出错: {e}")

    async def _select_source_group(self, allowed_groups: List[int]) -> Optional[int]:
        """选择合适的源群组"""
        # 确保屏蔽的源群组不会被选中
        available_source_groups = [gid for gid in allowed_groups if gid not in BLOCKED_SOURCE_GROUPS]
        if not available_source_groups:
            logger.info("没有可用的源群组（所有群组都被屏蔽）")
            return None
        
        current_time = time.time()
        active_groups = []
        for gid in available_source_groups:
            last_active = self.group_last_active_time.get(gid, 0)
            if current_time - last_active < 300:  # 5分钟
                active_groups.append(gid)
        
        if active_groups:
            return random.choice(active_groups)
        elif available_source_groups:
            logger.info("没有在过去5分钟内活跃的群组，随机选择一个可用群组")
            return random.choice(available_source_groups)
        else:
            return None

    async def _select_target_group(self, allowed_groups: List[int], source_group_id: int) -> Optional[int]:
        """选择合适的目标群组"""
        # 严格过滤掉屏蔽的目标群组
        available_target_groups = [
            gid for gid in allowed_groups 
            if gid != source_group_id and gid not in self.bot_active_groups and gid not in BLOCKED_TARGET_GROUPS
        ]
        
        if not available_target_groups:
            logger.info("没有可用的目标群组（所有群组被屏蔽或正在活跃参与中或是源群组）")
            return None
        
        suitable_groups = []
        for gid in available_target_groups:
            if not await self._has_bot_recent_messages(gid):
                suitable_groups.append(gid)
        
        if suitable_groups:
            return random.choice(suitable_groups)
        elif available_target_groups:
            logger.info("所有可用群组最近都有麦麦的发言，随机选择一个群组")
            return random.choice(available_target_groups)
        else:
            return None

    async def _has_bot_recent_messages(self, group_id: int) -> bool:
        """检查群组最近的消息中是否有麦麦的发言"""
        try:
            temp_group_info = GroupInfo(group_id=group_id, group_name=f"群{group_id}", platform="qq")
            temp_user_info = UserInfo(user_id=global_config.BOT_QQ, user_nickname=global_config.BOT_NICKNAME, platform="qq")
            
            chat_stream = await chat_manager.get_or_create_stream(
                platform="qq", user_info=temp_user_info, group_info=temp_group_info
            )
            
            recent_messages = get_recent_group_detailed_plain_text(
                chat_stream.stream_id, limit=15, combine=False
            )
            
            if not recent_messages:
                return False
                
            # 获取麦麦昵称和QQ号
            bot_nickname = global_config.BOT_NICKNAME
            bot_qq = global_config.BOT_QQ
            
            # 检查三种情况：
            # 1. 消息以麦麦昵称开头加冒号（常规消息格式）
            # 2. 消息中包含麦麦QQ号（防止用不同昵称发送）
            # 3. 消息中包含麦麦昵称（防止其他格式的消息）
            
            # 正则表达式查找格式为 "昵称:" 的消息
            bot_pattern_colon = rf"{re.escape(bot_nickname)}[:：]"
            # 正则表达式查找消息中包含麦麦QQ号的情况
            bot_pattern_qq = rf"{bot_qq}"
            # 正则表达式查找消息中包含麦麦昵称的情况
            bot_pattern_name = rf"{re.escape(bot_nickname)}"
            
            for msg in recent_messages:
                if re.search(bot_pattern_colon, msg):
                    logger.debug(f"在群 {group_id} 的最近消息中发现麦麦的冒号格式发言: {msg[:30]}...")
                    return True
                elif re.search(bot_pattern_qq, msg):
                    logger.debug(f"在群 {group_id} 的最近消息中发现包含麦麦QQ号的消息: {msg[:30]}...")
                    return True
                elif re.search(bot_pattern_name, msg):
                    logger.debug(f"在群 {group_id} 的最近消息中发现包含麦麦昵称的消息: {msg[:30]}...")
                    return True
            
            logger.debug(f"在群 {group_id} 的最近 {len(recent_messages)} 条消息中未发现麦麦的发言")
            return False
            
        except Exception as e:
            logger.error(f"检查群组最近消息时出错: {e}")
            return False

    async def _extract_topic(self, group_id: int) -> Tuple[Optional[List[str]], bool, List[str], Dict]:
        """从指定群组提取话题和图片"""
        try:
            temp_group_info = GroupInfo(group_id=group_id, group_name=f"群{group_id}", platform="qq")
            temp_user_info = UserInfo(user_id=global_config.BOT_QQ, user_nickname=global_config.BOT_NICKNAME, platform="qq")
            
            chat_stream = await chat_manager.get_or_create_stream(
                platform="qq", user_info=temp_user_info, group_info=temp_group_info
            )
            
            # 获取更多的上下文消息，确保有足够的内容进行话题提取
            context_size = getattr(global_config, "max_context_size", 15)
            recent_messages = get_recent_group_detailed_plain_text(
                chat_stream.stream_id, limit=context_size, combine=False
            )
            
            if not recent_messages:
                logger.info(f"群 {group_id} 没有最近的消息记录")
                return None, False, [], {}
            
            # 记录获取到的消息数量
            logger.info(f"从群 {group_id} 获取到 {len(recent_messages)} 条最近消息用于话题提取")
                
            has_images = False
            image_urls = []
            image_sender_data = {}
            
            if group_id in self.group_has_image and self.group_has_image[group_id]:
                current_time = time.time()
                potential_images = []
                priority_images = []
                
                # 用于跟踪已处理的图片URL，避免重复处理
                processed_urls = set()
                
                for img_url in self.group_has_image[group_id]:
                    # 检查URL是否已处理，避免重复
                    if img_url in processed_urls:
                        logger.debug(f"图片 {img_url[:30]}... 已经处理过，跳过")
                        continue
                    
                    processed_urls.add(img_url)
                    
                    is_filtered = await self._is_image_filtered(img_url)
                    if is_filtered:
                        logger.info(f"图片 {img_url[:30]}... 因内容被过滤")
                        continue
                        
                    if group_id in self.image_sender_info and img_url in self.image_sender_info[group_id]:
                        sender_info = self.image_sender_info[group_id][img_url]
                        sender_id = sender_info["sender_id"]
                        message_time = sender_info["message_time"]
                        
                        # 检查是否是优先QQ号发送的图片
                        if sender_id in PRIORITIZED_QQ_NUMBERS:
                            # 直接添加到优先图片列表
                            priority_images.append({
                                "url": img_url,
                                "priority": PRIORITY_BOOST_FACTOR,
                                "recency": 1.0,  # 始终给最高时效性
                                "score": PRIORITY_BOOST_FACTOR * 1.0,
                                "sender_id": sender_id,
                                "message_time": message_time,
                                "sender_info": sender_info
                            })
                            continue  # 跳过普通图片处理
                            
                        priority = 1.0
                        recency = max(0.1, min(1.0, 1 - (current_time - message_time) / 3600))
                        
                        potential_images.append({
                            "url": img_url,
                            "priority": priority,
                            "recency": recency,
                            "score": priority * recency,
                            "sender_id": sender_id,
                            "message_time": message_time,
                            "sender_info": sender_info
                        })
                
                # 用于跟踪已添加到结果的URL
                added_urls = set()
                
                # 先处理优先QQ号的图片
                if priority_images:
                    logger.info(f"发现优先QQ号 {PRIORITIZED_QQ_NUMBERS} 发送的图片，优先处理")
                    # 按时间顺序排序优先图片
                    priority_images.sort(key=lambda x: x["message_time"], reverse=True)
                    
                    # 获取同一发送者的连续图片
                    continuous_priority_images = []
                    if priority_images:
                        continuous_priority_images.append(priority_images[0])
                        sender_id = priority_images[0]["sender_id"]
                        message_time = priority_images[0]["message_time"]
                        
                        for img in priority_images[1:]:
                            if (img["sender_id"] == sender_id and 
                                abs(img["message_time"] - message_time) <= IMAGE_TIME_WINDOW):
                                continuous_priority_images.append(img)
                                if len(continuous_priority_images) >= MAX_IMAGES_TO_FORWARD:
                                    break
                    
                    # 添加优先图片到结果，避免重复
                    for img in continuous_priority_images:
                        if img["url"] not in added_urls:
                            image_urls.append(img["url"])
                            image_sender_data[img["url"]] = img["sender_info"]
                            added_urls.add(img["url"])
                # 如果没有优先图片，则处理普通图片
                else:
                    potential_images.sort(key=lambda x: x["score"], reverse=True)
                    
                    continuous_images = []
                    if potential_images:
                        continuous_images.append(potential_images[0])
                        sender_id = potential_images[0]["sender_id"]
                        message_time = potential_images[0]["message_time"]
                        
                        for img in potential_images[1:]:
                            if (img["sender_id"] == sender_id and 
                                abs(img["message_time"] - message_time) <= IMAGE_TIME_WINDOW):
                                continuous_images.append(img)
                                if len(continuous_images) >= MAX_IMAGES_TO_FORWARD:
                                    break
                    
                    if continuous_images:
                        for img in continuous_images:
                            if img["url"] not in added_urls:
                                image_urls.append(img["url"])
                                image_sender_data[img["url"]] = img["sender_info"]
                                added_urls.add(img["url"])
                    elif potential_images:
                        if potential_images[0]["url"] not in added_urls:
                            image_urls.append(potential_images[0]["url"])
                            image_sender_data[potential_images[0]["url"]] = potential_images[0]["sender_info"]
                            added_urls.add(potential_images[0]["url"])
                
                has_images = len(image_urls) > 0
                
                if has_images:
                    logger.info(f"群 {group_id} 选择了 {len(image_urls)} 张图片用于转发: {image_urls}")
            
            # 确保消息内容有意义，格式化群聊消息以便LLM更好理解
            formatted_messages = []
            for msg in recent_messages:
                # 过滤掉可能的空消息
                if msg and len(msg.strip()) > 0:
                    formatted_messages.append(msg)
            
            if not formatted_messages:
                logger.info(f"群 {group_id} 的消息内容为空或无效")
                return None, has_images, image_urls, image_sender_data
                
            formatted_context = "\n".join(formatted_messages)
            
            # 记录将要发送给LLM的上下文长度
            logger.debug(f"发送给LLM的上下文长度为 {len(formatted_context)} 字符")
            
            prompt = f"""
以下是一个群聊中的最近对话：

{formatted_context}

请分析这段对话，提取出1-3个可能的新话题选项用来在其他的群里谈论新话题。这些话题可以是：
1. 主要讨论的话题
2. 对话中提到的有趣观点
3. 引人思考的问题
4. 值得讨论的相关延伸话题

以简单的列表形式返回，每个话题用一句话概括，不需要编号或其他格式。
如果对话混乱或没有明确话题，可以返回"无明确话题"。
"""
            
            logger.debug(f"开始总结群 {group_id} 的话题选项请求")
            topics_text, _ = await self.llm_topic_judge.generate_response(prompt)
            
            # 检查LLM返回结果
            if not topics_text:
                logger.warning(f"LLM返回为空，无法提取群 {group_id} 的话题")
                return None, has_images, image_urls, image_sender_data
                
            logger.debug(f"LLM返回的原始话题文本: {topics_text}")
            
            if "无明确话题" in topics_text or len(topics_text.strip()) < 5:
                logger.info(f"群 {group_id} 没有提取到有效话题")
                return None, has_images, image_urls, image_sender_data
            
            # 处理LLM返回的话题选项
            topic_options = [topic.strip() for topic in topics_text.strip().split('\n') if topic.strip()]
            topic_options = [topic for topic in topic_options if len(topic) >= 5]
            
            if not topic_options:
                logger.info(f"群 {group_id} 没有提取到有效话题选项")
                return None, has_images, image_urls, image_sender_data
                
            logger.info(f"成功从群 {group_id} 提取到 {len(topic_options)} 个话题选项: {topic_options}")
            return topic_options, has_images, image_urls, image_sender_data
            
        except Exception as e:
            logger.error(f"提取话题过程出错: {e}")
            return None, False, [], {}

    async def _is_image_filtered(self, image_url: str) -> bool:
        """检查图片是否应该被过滤"""
        try:
            # 使用现有的 utils_image.ImageManager 替代缺失的 image_classifier 模块
            from .utils_image import image_manager
            import hashlib
            import base64
            from ...common.database import db
            import os
            
            # 首先检查缓存中是否有此图片
            if image_url in self.image_cache:
                logger.info(f"图片 {image_url[:30]}... 已在缓存中")
                image_base64 = self.image_cache[image_url]
            else:
                # 尝试从数据库中获取图片信息
                # 如果是URL类型的图片，尝试通过URL查询数据库
                if image_url.startswith(('http://', 'https://')):
                    # 先查询数据库中是否有相同URL的图片记录
                    image_record = db.images.find_one({"url": image_url})
                    
                    if image_record and "path" in image_record and os.path.exists(image_record["path"]):
                        # 数据库中有记录且文件存在，直接使用
                        logger.info(f"从数据库记录中找到图片: {image_url[:30]}...")
                        image_path = image_record["path"]
                        with open(image_path, "rb") as f:
                            image_bytes = f.read()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            # 缓存图片
                            self.image_cache[image_url] = image_base64
                    else:
                        # 数据库中没有记录，需要下载图片
                        logger.info(f"数据库中未找到图片URL记录，需要下载: {image_url[:30]}...")
                        
                        # 尝试下载图片
                        max_retries = 3
                        retry_delay = 1
                        
                        for attempt in range(max_retries):
                            try:
                                import aiohttp
                                import ssl
                                
                                # 创建自定义SSL上下文
                                ssl_context = ssl.create_default_context()
                                ssl_context.check_hostname = False
                                ssl_context.verify_mode = ssl.CERT_NONE
                                
                                connector = aiohttp.TCPConnector(ssl=ssl_context)
                                
                                async with aiohttp.ClientSession(connector=connector) as session:
                                    async with session.get(image_url, timeout=10) as response:
                                        if response.status == 200:
                                            image_bytes = await response.read()
                                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                            
                                            # 计算图片哈希
                                            image_hash = hashlib.md5(image_bytes).hexdigest()
                                            
                                            # 存储图片到本地文件系统
                                            timestamp = int(time.time())
                                            from PIL import Image
                                            import io
                                            try:
                                                image_format = Image.open(io.BytesIO(image_bytes)).format.lower()
                                            except Exception:
                                                image_format = "jpg"  # 默认格式
                                                
                                            filename = f"{timestamp}_{image_hash[:8]}.{image_format}"
                                            image_dir = os.path.join("data", "image")
                                            if not os.path.exists(image_dir):
                                                os.makedirs(image_dir)
                                            file_path = os.path.join(image_dir, filename)
                                            
                                            with open(file_path, "wb") as f:
                                                f.write(image_bytes)
                                                
                                            # 更新数据库记录
                                            db.images.update_one(
                                                {"hash": image_hash},
                                                {
                                                    "$set": {
                                                        "hash": image_hash,
                                                        "path": file_path,
                                                        "url": image_url,
                                                        "type": "image",
                                                        "timestamp": timestamp,
                                                    }
                                                },
                                                upsert=True,
                                            )
                                            
                                            # 缓存图片
                                            self.image_cache[image_url] = image_base64
                                            logger.info(f"图片 {image_url[:30]}... 下载成功并保存至: {file_path}")
                                            break
                                        else:
                                            logger.warning(f"下载图片失败: {image_url}, 状态码: {response.status}, 尝试 {attempt + 1}/{max_retries}")
                                            if attempt < max_retries - 1:
                                                await asyncio.sleep(retry_delay)
                                                retry_delay *= 2  # 指数退避
                            except Exception as e:
                                logger.warning(f"下载图片时出错: {e}, 尝试 {attempt + 1}/{max_retries}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2
                                else:
                                    logger.error(f"下载图片失败，已达到最大重试次数: {image_url}")
                                    return False
                elif image_url.startswith('base64://'):
                    # 已经是base64格式，去掉前缀
                    image_base64 = image_url.replace('base64://', '')
                    # 缓存base64图片
                    self.image_cache[image_url] = image_base64
                    
                    # 计算图片哈希并尝试保存
                    try:
                        image_bytes = base64.b64decode(image_base64)
                        image_hash = hashlib.md5(image_bytes).hexdigest()
                        
                        # 检查数据库中是否已有该图片
                        existing_image = db.images.find_one({"hash": image_hash})
                        if not existing_image:
                            # 存储图片到本地文件系统
                            timestamp = int(time.time())
                            from PIL import Image
                            import io
                            try:
                                image_format = Image.open(io.BytesIO(image_bytes)).format.lower()
                            except Exception:
                                image_format = "jpg"  # 默认格式
                                
                            filename = f"{timestamp}_{image_hash[:8]}.{image_format}"
                            image_dir = os.path.join("data", "image")
                            if not os.path.exists(image_dir):
                                os.makedirs(image_dir)
                            file_path = os.path.join(image_dir, filename)
                            
                            with open(file_path, "wb") as f:
                                f.write(image_bytes)
                                
                            # 更新数据库记录
                            db.images.update_one(
                                {"hash": image_hash},
                                {
                                    "$set": {
                                        "hash": image_hash,
                                        "path": file_path,
                                        "type": "image",
                                        "timestamp": timestamp,
                                    }
                                },
                                upsert=True,
                            )
                            logger.info(f"Base64图片已保存至: {file_path}")
                    except Exception as e:
                        logger.error(f"保存Base64图片失败: {e}")
                else:
                    # 假设是文件路径，尝试直接读取
                    from .utils_image import image_path_to_base64
                    image_base64 = image_path_to_base64(image_url)
                    if not image_base64:
                        logger.error(f"无法读取图片文件: {image_url}")
                        return False
                    # 缓存文件图片
                    self.image_cache[image_url] = image_base64
            
            # 使用图片识别功能获取图片描述
            # 检查数据库中是否已有该图片的描述
            if image_url.startswith(('http://', 'https://')):
                # 对于URL图片，先尝试通过URL查询
                image_record = db.images.find_one({"url": image_url})
                if image_record and "hash" in image_record:
                    image_hash = image_record["hash"]
                    description_record = db.image_descriptions.find_one({"hash": image_hash, "type": "image"})
                    if description_record and "description" in description_record:
                        image_description = description_record["description"]
                        image_description_with_prefix = f"[图片：{image_description}]"
                        logger.info(f"从数据库中获取图片描述: {image_description}")
                    else:
                        # 数据库中没有描述，需要生成
                        image_description_with_prefix = await image_manager.get_image_description(image_base64)
                else:
                    # 没有找到图片记录，需要生成描述
                    image_description_with_prefix = await image_manager.get_image_description(image_base64)
            else:
                # 对于base64或文件路径图片，直接生成描述
                image_description_with_prefix = await image_manager.get_image_description(image_base64)
            
            # 提取纯描述内容，处理各种可能的格式
            if isinstance(image_description_with_prefix, str):
                # 去除前缀 "[图片：" 和后缀 "]"
                if image_description_with_prefix.startswith("[图片：") and image_description_with_prefix.endswith("]"):
                    image_description = image_description_with_prefix[5:-1]
                # 处理缓存中可能的其他格式
                elif image_description_with_prefix.startswith("[") and "：" in image_description_with_prefix and image_description_with_prefix.endswith("]"):
                    # 尝试提取冒号后的内容
                    parts = image_description_with_prefix.split("：", 1)
                    if len(parts) > 1:
                        image_description = parts[1].rstrip("]")
                    else:
                        image_description = image_description_with_prefix
                else:
                    image_description = image_description_with_prefix
                
                # 记录完整的图片描述，便于调试
                logger.info(f"图片描述(原始): {image_description_with_prefix}")
                logger.info(f"图片描述(处理后): {image_description}")
                
                # 更严格的过滤逻辑：确保能捕获各种形式的敏感内容
                image_description_lower = image_description.lower()  # 转为小写以进行不区分大小写的比较
                
                # 检查图片描述是否包含敏感内容
                for filtered_topic in FILTERED_IMAGE_TOPICS:
                    # 使用更严格的匹配：完整词匹配或词边界匹配
                    filtered_topic_lower = filtered_topic.lower()
                    if filtered_topic_lower in image_description_lower:
                        # 查找所有匹配位置进行进一步分析
                        matches = []
                        start = 0
                        while True:
                            start = image_description_lower.find(filtered_topic_lower, start)
                            if start == -1:
                                break
                            matches.append(start)
                            start += len(filtered_topic_lower)
                        
                        # 记录详细的匹配信息
                        match_contexts = []
                        for match_pos in matches:
                            context_start = max(0, match_pos - 10)
                            context_end = min(len(image_description), match_pos + len(filtered_topic) + 10)
                            match_context = image_description[context_start:context_end]
                            match_contexts.append(f"...{match_context}...")
                        
                        logger.info(f"图片 {image_url[:30]}... 包含敏感内容 '{filtered_topic}'，匹配位置: {matches}")
                        logger.info(f"匹配上下文: {match_contexts}")
                        return True
                
                return False
            else:
                logger.warning(f"图片描述格式异常: {type(image_description_with_prefix)}")
                return False
            
        except Exception as e:
            logger.error(f"检查图片内容时出错: {e}")
            return False

    async def _forward_topic(self, topic_options: List[str], target_group_id: int, has_images: bool = False, image_urls: List[str] = None, image_sender_data: Dict = None):
        """转发话题到目标群组"""
        try:
            bot = get_bot()
            
            has_valid_images = has_images and image_urls
            has_valid_topics = topic_options and len(topic_options) > 0
            
            should_send_image = False
            
            if has_valid_images:
                if not has_valid_topics:
                    should_send_image = True
                    logger.info(f"没有有效话题，决定向群 {target_group_id} 转发图片")
                else:
                    should_send_image = random.random() < IMAGE_FORWARD_PROBABILITY
                    if should_send_image:
                        logger.info(f"按概率决定向群 {target_group_id} 转发图片")
            
            bot_user_info = UserInfo(
                user_id=global_config.BOT_QQ,
                user_nickname=global_config.BOT_NICKNAME,
                platform="qq"
            )
            
            group_info = GroupInfo(
                group_id=target_group_id,
                group_name=f"群{target_group_id}",
                platform="qq"
            )
            
            chat_stream = await chat_manager.get_or_create_stream(
                platform="qq", user_info=bot_user_info, group_info=group_info
            )
            
            if should_send_image:
                # 确保图片列表不为空
                if not image_urls:
                    logger.warning(f"决定向群 {target_group_id} 转发图片，但图片列表为空")
                    return
                    
                # 确保不会超过最大转发图片数量
                selected_images = image_urls[:MAX_IMAGES_TO_FORWARD]
                message_id_base = f"topic_img_{int(time.time())}_{target_group_id}"
                
                logger.debug(f"将向群 {target_group_id} 转发 {len(selected_images)} 张图片: {selected_images}")
                
                # 追踪已发送的图片URL，避免重复发送
                sent_image_urls = set()
                
                sender_text = None
                message_text_from_image_sender = None
                
                # 仅获取一次sender_text，避免重复
                for img_url in selected_images:
                    if img_url in sent_image_urls:
                        continue
                        
                    if image_sender_data and img_url in image_sender_data:
                        sender_info = image_sender_data[img_url]
                        
                        if "context_messages" in sender_info:
                            for ctx_msg in sender_info["context_messages"]:
                                if str(sender_info["sender_id"]) in ctx_msg.get("username", ""):
                                    sender_text = ctx_msg.get("content", "")
                                    break
                        
                        if not sender_text and "message_text" in sender_info and sender_info["message_text"]:
                            message_text_from_image_sender = sender_info["message_text"]
                            
                    # 一旦找到了发送者信息，就跳出循环
                    if sender_text or message_text_from_image_sender:
                        break
                            
                for i, img_url in enumerate(selected_images):
                    # 跳过已发送的图片
                    if img_url in sent_image_urls:
                        logger.debug(f"图片 {img_url[:30]}... 已经发送过，跳过")
                        continue
                    
                    sent_image_urls.add(img_url)
                    
                    try:
                        message_id = f"{message_id_base}_{i}"
                        message_set = MessageSet(chat_stream, message_id)
                        
                        # 使用缓存或数据库中的图片
                        if img_url in self.image_cache:
                            # 直接从缓存中获取
                            base64_data = self.image_cache[img_url]
                            logger.debug(f"使用缓存图片数据: {img_url[:30]}...")
                        else:
                            # 尝试从数据库获取
                            from ...common.database import db
                            import base64
                            import hashlib
                            import os
                            
                            # 如果是URL，尝试通过URL查询数据库
                            if img_url.startswith(('http://', 'https://')):
                                image_record = db.images.find_one({"url": img_url})
                                
                                if image_record and "path" in image_record and os.path.exists(image_record["path"]):
                                    # 从本地文件加载图片
                                    logger.debug(f"从数据库记录中的本地文件加载图片: {img_url[:30]}...")
                                    with open(image_record["path"], "rb") as f:
                                        image_bytes = f.read()
                                        base64_data = base64.b64encode(image_bytes).decode('utf-8')
                                        # 缓存结果
                                        self.image_cache[img_url] = base64_data
                                else:
                                    # 没有找到记录，使用CQCode获取图片
                                    logger.debug(f"数据库中没有图片记录，使用CQCode获取: {img_url[:30]}...")
                                    from .cq_code import CQCode
                                    temp_cq = CQCode(
                                        type="image",
                                        params={"url": img_url},
                                        group_info=group_info,
                                        user_info=bot_user_info
                                    )
                                    base64_data = await temp_cq.get_img()
                                    
                                    if base64_data:
                                        # 保存到数据库和本地文件
                                        image_bytes = base64.b64decode(base64_data)
                                        image_hash = hashlib.md5(image_bytes).hexdigest()
                                        
                                        # 存储图片到本地文件
                                        timestamp = int(time.time())
                                        from PIL import Image
                                        import io
                                        try:
                                            image_format = Image.open(io.BytesIO(image_bytes)).format.lower()
                                        except Exception:
                                            image_format = "jpg"  # 默认格式
                                            
                                        filename = f"{timestamp}_{image_hash[:8]}.{image_format}"
                                        image_dir = os.path.join("data", "image")
                                        if not os.path.exists(image_dir):
                                            os.makedirs(image_dir)
                                        file_path = os.path.join(image_dir, filename)
                                        
                                        with open(file_path, "wb") as f:
                                            f.write(image_bytes)
                                            
                                        # 更新数据库记录
                                        db.images.update_one(
                                            {"hash": image_hash},
                                            {
                                                "$set": {
                                                    "hash": image_hash,
                                                    "path": file_path,
                                                    "url": img_url,
                                                    "type": "image",
                                                    "timestamp": timestamp,
                                                }
                                            },
                                            upsert=True,
                                        )
                                        
                                        # 缓存图片
                                        self.image_cache[img_url] = base64_data
                                    else:
                                        logger.error(f"无法获取图片: {img_url}")
                                        continue
                            elif img_url.startswith('base64://'):
                                # 已经是base64格式，去掉前缀
                                base64_data = img_url.replace('base64://', '')
                                self.image_cache[img_url] = base64_data
                            else:
                                # 假设是文件路径，直接读取
                                from .utils_image import image_path_to_base64
                                base64_data = image_path_to_base64(img_url)
                                if base64_data:
                                    self.image_cache[img_url] = base64_data
                                else:
                                    logger.error(f"无法读取图片文件: {img_url}")
                                    continue
                        
                        message_segment = Seg(type="image", data=base64_data)
                        
                        bot_message = MessageSending(
                            message_id=message_id,
                            chat_stream=chat_stream,
                            bot_user_info=bot_user_info,
                            sender_info=None,
                            message_segment=message_segment,
                            reply=None,
                            is_head=True,
                            is_emoji=False,
                        )
                        message_set.add_message(bot_message)
                        message_manager.add_message(message_set)
                        logger.debug(f"成功发送图片 {i+1}/{len(selected_images)}: {img_url[:30]}...")
                        
                    except Exception as e:
                        logger.error(f"处理图片时出错: {e}")
                        continue
                
                # 只发送一次文本消息
                if sender_text or message_text_from_image_sender:
                    text_to_send = sender_text or message_text_from_image_sender
                    
                    message_id = f"topic_img_text_{int(time.time())}_{target_group_id}"
                    message_set = MessageSet(chat_stream, message_id)
                    
                    message_segment = Seg(type="text", data=text_to_send)
                    
                    bot_message = MessageSending(
                        message_id=message_id,
                        chat_stream=chat_stream,
                        bot_user_info=bot_user_info,
                        sender_info=None,
                        message_segment=message_segment,
                        reply=None,
                        is_head=True,
                        is_emoji=False,
                    )
                    
                    message_set.add_message(bot_message)
                    message_manager.add_message(message_set)
                    
                    logger.info(f"发送了图片发送者的文本：{text_to_send[:20]}...")
                
                self.bot_active_groups.add(target_group_id)
                logger.info(f"成功将图片转发到群 {target_group_id}")
                
                clear_time = getattr(global_config, "topic_active_clear_time", 30) * 60
                asyncio.create_task(self._clear_group_active_status(target_group_id, clear_time))
                return
            
            if not has_valid_topics:
                logger.info(f"没有有效话题可转发到群 {target_group_id}，且决定不转发图片")
                return
                
            selected_topic = random.choice(topic_options)
            logger.info(f"随机选择话题: {selected_topic}")
            
            natural_topic_prompt = f"""
请将以下话题转化为一条自然的群聊发言，就像是你自己突然想到的话题：

话题："{selected_topic}"

要求：
1. 非常自然地引入话题，就像是自己突然遇到的一个事情，但不能有类似"刚刚想到"、"你们怎么看"的表达方式
2. 不要提及这来自于其他群聊
3. 口吻犀利、直接
4. 可以略带讽刺或挑衅，也可以当做一件开心事来说
5. 非常简短，不超过15字
6. 不要使用引号或冒号
7. 避免过于正式的表达
8. 不需要客套话或引导语
9. 不需要解释或分析

直接输出你的发言内容，不要包含任何解释。
"""
            
            logger.debug(f"开始生成自然话题表达")
            final_message, _ = await self.response_llm.generate_response(natural_topic_prompt)
            
            if not final_message or len(final_message) > 50:
                if final_message and len(final_message) > 50:
                    final_message = final_message[:30]
                else:
                    logger.info(f"LLM生成话题表达失败，放弃向群 {target_group_id} 发送消息")
                    return
            
            logger.info(f"生成的自然话题表达: {final_message}")
            
            message_id = f"topic_{int(time.time())}_{target_group_id}"
            message_set = MessageSet(chat_stream, message_id)
            
            typing_time = calculate_typing_time(final_message)
            
            message_segment = Seg(type="text", data=final_message)
            
            bot_message = MessageSending(
                message_id=message_id,
                chat_stream=chat_stream,
                bot_user_info=bot_user_info,
                sender_info=None,
                message_segment=message_segment,
                reply=None,
                is_head=True,
                is_emoji=False,
            )
            
            message_set.add_message(bot_message)
            message_manager.add_message(message_set)
            
            self.bot_active_groups.add(target_group_id)
            
            logger.info(f"成功将话题转发到群 {target_group_id}")
            
            clear_time = getattr(global_config, "topic_active_clear_time", 30) * 60
            asyncio.create_task(self._clear_group_active_status(target_group_id, clear_time))
            
        except Exception as e:
            logger.error(f"转发话题过程出错: {e}")

    async def on_group_message(self, event: GroupMessageEvent):
        """处理群组消息事件"""
        group_id = event.group_id
        sender_id = event.user_id
        current_time = time.time()
        message_text = event.message.extract_plain_text()
        
        self.group_last_active_time[group_id] = current_time
        
        if '[CQ:image' in str(event.message):
            image_urls = []
            # 从消息中提取图片URL
            for segment in event.message:
                if segment.type == 'image':
                    if 'url' in segment.data:
                        image_urls.append(segment.data['url'])
                    elif 'file' in segment.data:
                        file_data = segment.data['file']
                        if file_data.startswith('base64://'):
                            image_urls.append(file_data)
                        else:
                            image_urls.append(file_data)
            
            if image_urls:
                # 初始化群组的图片存储
                if group_id not in self.group_has_image:
                    self.group_has_image[group_id] = []
                
                # 记录新图片前的数量
                previous_image_count = len(self.group_has_image[group_id])
                
                # 使用集合去重操作
                unique_new_urls = [url for url in image_urls if url not in set(self.group_has_image[group_id])]
                
                if unique_new_urls:
                    # 添加新的唯一图片URL
                    self.group_has_image[group_id].extend(unique_new_urls)
                    
                    # 保持最近的10张图片
                    if len(self.group_has_image[group_id]) > 10:
                        self.group_has_image[group_id] = self.group_has_image[group_id][-10:]
                    
                    current_image_count = len(self.group_has_image[group_id])
                    logger.debug(f"群 {group_id} 新增 {len(unique_new_urls)} 张唯一图片，移除 {previous_image_count - (current_image_count - len(unique_new_urls))} 张旧图片，当前共有 {current_image_count} 张图片")
                    
                    # 初始化或更新发送者信息
                    if group_id not in self.image_sender_info:
                        self.image_sender_info[group_id] = {}
                    
                    context_messages = await self._get_recent_context_messages(group_id, 3)
                    
                    # 只为新的唯一图片添加发送者信息
                    for img_url in unique_new_urls:
                        self.image_sender_info[group_id][img_url] = {
                            "sender_id": sender_id,
                            "message_time": current_time,
                            "context_messages": context_messages,
                            "message_text": message_text
                        }
                else:
                    logger.debug(f"群 {group_id} 没有新的唯一图片需要添加")
        
        if event.is_tome():
            self.bot_active_groups.add(group_id)
            clear_time = getattr(global_config, "topic_active_clear_time", 30) * 60
            asyncio.create_task(self._clear_group_active_status(group_id, clear_time))

    async def _get_recent_context_messages(self, group_id: int, limit: int = 3) -> List[Dict]:
        """获取群组最近的上下文消息"""
        try:
            temp_group_info = GroupInfo(group_id=group_id, group_name=f"群{group_id}", platform="qq")
            temp_user_info = UserInfo(user_id=global_config.BOT_QQ, user_nickname=global_config.BOT_NICKNAME, platform="qq")
            
            chat_stream = await chat_manager.get_or_create_stream(
                platform="qq", user_info=temp_user_info, group_info=temp_group_info
            )
            
            recent_messages = get_recent_group_detailed_plain_text(
                chat_stream.stream_id, limit=limit, combine=False
            )
            
            context = []
            for msg in recent_messages:
                match = re.match(r"([^:：]+)[：:](.*)", msg)
                if match:
                    username, content = match.groups()
                    context.append({
                        "username": username.strip(),
                        "content": content.strip()
                    })
            
            return context
            
        except Exception as e:
            logger.error(f"获取上下文消息时出错: {e}")
            return []

    async def _clear_group_active_status(self, group_id: int, clear_time: int = 1800):
        """清理群组活跃状态"""
        await asyncio.sleep(clear_time)
        if group_id in self.bot_active_groups:
            self.bot_active_groups.remove(group_id)
            logger.debug(f"群 {group_id} 的活跃状态已清除")

    async def _select_multiple_target_groups(self, allowed_groups: List[int], source_group_id: int, max_count: int) -> List[int]:
        """选择多个合适的目标群组"""
        # 严格过滤掉屏蔽的目标群组
        available_target_groups = [
            gid for gid in allowed_groups 
            if gid != source_group_id and gid not in BLOCKED_TARGET_GROUPS
        ]
        
        if not available_target_groups:
            logger.info("没有可用的目标群组（所有群组被屏蔽或是源群组）")
            return []
        
        # 根据麦麦活跃状态和最近消息情况筛选合适的群组
        suitable_groups = []
        less_suitable_groups = []
        
        for gid in available_target_groups:
            # 检查是否是麦麦活跃的群组
            is_bot_active = gid in self.bot_active_groups
            
            # 检查是否有最近的麦麦消息
            has_recent_messages = await self._has_bot_recent_messages(gid)
            
            if not is_bot_active and not has_recent_messages:
                suitable_groups.append(gid)
                logger.debug(f"群 {gid} 既不在活跃状态，也没有最近的麦麦消息，是合适的目标群")
            else:
                if is_bot_active:
                    logger.debug(f"群 {gid} 在麦麦活跃状态中，不是合适的目标群")
                if has_recent_messages:
                    logger.debug(f"群 {gid} 最近有麦麦消息，不是合适的目标群")
                less_suitable_groups.append(gid)
        
        selected_groups = []
        
        # 首先从合适的群组中选择
        if suitable_groups:
            logger.info(f"找到 {len(suitable_groups)} 个合适的目标群组: {suitable_groups}")
            # 随机打乱以实现随机选择
            random.shuffle(suitable_groups)
            # 选择最多max_count个群
            selected_groups.extend(suitable_groups[:max_count])
        
        # 如果合适的群不够，从不太合适的群中补充
        if len(selected_groups) < max_count and less_suitable_groups:
            remaining_count = max_count - len(selected_groups)
            logger.info(f"合适的群组不足，从 {len(less_suitable_groups)} 个次要群组中补充 {remaining_count} 个")
            random.shuffle(less_suitable_groups)
            selected_groups.extend(less_suitable_groups[:remaining_count])
        
        # 确保不超过最大数量
        selected_groups = selected_groups[:max_count]
        
        if not selected_groups:
            logger.info("没有找到任何合适的目标群组")
            return []
            
        logger.info(f"最终选择了 {len(selected_groups)} 个目标群组: {selected_groups}")
        return selected_groups


# 创建全局单例
topic_identifier = TopicIdentifier()

# 自动启动话题监控
@driver.on_startup
async def _():
    logger.info("开始启动话题识别器的监控功能")
    await topic_identifier.start_monitoring()
    logger.success("话题识别器监控功能已启动")

# 注册消息处理器
from nonebot.plugin import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent

group_msg_handler = on_message(priority=5, block=False)

@group_msg_handler.handle()
async def handle_group_message(event: GroupMessageEvent):
    # 处理群消息事件
    await topic_identifier.on_group_message(event)
