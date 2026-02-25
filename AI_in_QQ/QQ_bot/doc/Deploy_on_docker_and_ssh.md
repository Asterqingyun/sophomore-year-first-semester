## 配置步骤
### napcat in docker
1. 拉取并运行容器
``` shell
docker run -d \
-e NAPCAT_GID=$(id -g) \
-e NAPCAT_UID=$(id -u) \
-p 3000:3000 \
-p 3001:3001 \
-p 6099:6099 \
--name napcat \
--restart=always \
mlikiowa/napcat-docker:latest
```
2. ` docker logs napcat` 扫描二维码登录
3. 观察终端输出一个叫做 http://127.0.0.1:6099/webui?token=xxx 的网站，点击用来配置napcat。
   配置细节：
   反向socket(主动连接)，网址是`ws://172.17.0.1:8080/onebot/v11/ws`，其中172.17.0.1是容器连外界的ip，8080是nonebot的默认端口
### nonebot in linux
1. 采用包管理器uv `python3 -m pip install uv`
2. 用uv下载 `uv pip install nb-cli`
3. 创建项目选择 `nb create`，适配器是onebot-v11,驱动器是fastapi，似乎http也是需要的
4. 编辑.env文件，设置端口(8080）和IP（0.0.0.0），表示本机容器内napcat的也可以访问）；编辑bot.py，导入库和指定端口之类
5. 运行文件

## 个人所遇问题&所学
qwq((*´▽｀)ノノ)
1. 服务器代理（最开始）不可用，（查询资料似乎绕不过去docker代理配置文件），所以不用代理的话似乎只能删配置文件，（公用服务器不好意思删），所以就在本地拉取镜像然后保存然后发送
   学习（复习？）到了：
        docker保存镜像命令：`docker save -o myimage.tar myimage:latest`
        docker从压缩包导入镜像命令：`docker load -i mynginx.tar`
        scp发送命令:`scp <源文件> <user_name>@<remote_host>:<目标位置>`
2. 因为不会在服务器上登录网站（^C一下子就从本机登录），于是用端口转发命令
    `ssh -p <当前连接端口> -L <本机端口>:<ip>:<对面端口> user@服务器IP`
    相当于ssh -p <当前连接端口>，建立本机与服务器之间的ssh隧道；然后本机所有的在本机端口上的访问，都会被转发给服务器，让服务器去访问这个ip和相应的端口。

    如果是内网，我们可以直接在本机上访问相应的ip；但是并非内网，所以需要端口转发
    然后访问`http://127.0.0.1:<本机端口>/xxxx`即可
3. 每一个nonebot project有一个单独的虚拟环境（如果在创建的时候选择了配置），所以要记得激活。
4. 所有端口总览www
5. 
   docker和服务器本身端口之间：p1：p2 左边是宿主机的端口，右边是容器的端口，端口监听通信。
   
   外部监听内部用的是0.0.0.0（来者不拒）（所以nonebot是这个IP，相当于来者访问都可以）；端口映射	从外向容器里`127.0.0.1:宿主机端口`；而`docker inspect 的 IP`得到的是虚拟网卡分配的地址（用于容器互连？）；而从容器内看容器外的ip是`172.17.0.1`（所以napcat写的配置是这个！）

   ssh -P xxxx 是ssh 的端口
