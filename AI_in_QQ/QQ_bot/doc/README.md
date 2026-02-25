## 如何配置属于你的QQ机器人:Napcat-qq&nonebot（环境：windows)

### 原理
napcat-qq和qq进行连接，然后主动反向socket连接nonebot;nonebot连接机器人端。
### napcat-qq配置
1. 下载源文件：前往 NapCatQQ 的 Releases 页面 下载 NapCat.Shell.zip 并解压：https://github.com/NapNeko/NapCatQQ/releases
2. 双击目录下 launcher.bat 即可启动，后续只用`laucher.bat <your qq number>`就可以实现采集消息了
3. 网络配置：
   1. 网页在启动launcher.bat之后的命令行输出的较前部分，格式是`WebUi User Panel Url: http://127.0.0.1:6099/<my_token>`。登入后面的网页即可进行配置
   2. 添加网络配置里面的websocket客户端，用于主动连接Nonebot。token自拟。url是`ws://127.0.0.1:8080/onebot/v11/ws`

### nonebot配置
1. 创建一个虚拟环境之后，可以选用包管理工具uv 进行下载:`uv pip install nb-cli`
   **有可能你会发现你的python版本不支持，可以复制报错给ai知道下载哪个版本比较好，然后去官网下一个新的python，然后`Ctrl+shift+P`选择你下的那个解释器**
2. 下载成功后，在终端里输入`nb create`一个新的项目，bookstrap即可，**不选择适配器（包括QQ适配器，因为这个适配器是用来连qq官方的，不是连napcat的）**,选择驱动器是 FastAPI 驱动器（我还选了HTTPX 驱动器和 WebSockets驱动器，应当是没有用的，因为fastAPI是用来被别人连的，而这两个驱动器是用来连别人的）。让它帮你创建虚拟环境和下载依赖。
3. 自行下载适配器：`uv pip install nonebot-adapter-onebot`
4. 创建bot.py，不用官方的而是用这个启动的主要目的是导入我们刚刚自己下载的适配器。这样写下：
```python
import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter #导入我们的那个适配器(v11)

# 测试用的一些功能，后续开发可用
from nonebot import on_command, on_fullmatch
from nonebot.adapters.onebot.v11 import Bot, Event


nonebot.init() #初始化，读入.env的配置
driver = nonebot.get_driver() #获取.env中的驱动器
driver.register_adapter(OneBotV11Adapter) #加入我们刚刚下载的适配器
nonebot.load_from_toml("pyproject.toml")

# ==========================================
# 👇 这里就是机器人的“技能”区域，后续会改成核心的ai 
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
```
5. 修改同级目录下.env配置
```env
DRIVER=~fastapi+~httpx 
LOCALSTORE_USE_CWD=true
ADAPTERS=nonebot.adapters.onebot.v11
ONEBOT_ACCESS_TOKEN=<my_token_in_napcat>
```
6. 运行`python bot.py`



