import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

# 👇 新增：导入这几个工具，用来定义回复规则
from nonebot import on_command, on_fullmatch
from nonebot.adapters.onebot.v11 import Bot, Event

nonebot.init()
driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)
nonebot.load_from_toml("pyproject.toml")

# ==========================================
# 👇 这里就是机器人的“技能”区域
# ==========================================

# 技能 1：完全匹配。当你发 "你好" 时触发
hello = on_fullmatch("你好")


@hello.handle()
async def _(bot: Bot, event: Event):
    # 发送回复
    await hello.finish("你好呀！我是你的机器人！🎉")


# 技能 2：命令匹配。当你发 "/ping" 时触发
ping = on_command("ping")


@ping.handle()
async def _(bot: Bot, event: Event):
    await ping.finish("Pong! 🏓")


# ==========================================

if __name__ == "__main__":
    nonebot.run()
