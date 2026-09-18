"""人设生成：正常用户与水军账号的昵称/简介/头像/兴趣。"""
from __future__ import annotations
import random, string

SURNAMES = list("王李张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文")
GIVEN = ["子涵","一诺","浩然","欣怡","梓萱","雨泽","思远","可欣","嘉懿","晨曦","若曦","嘉豪","语桐","俊杰","安琪","泽宇","诗涵","文博","佳琪","宇航","梦琪","博文","雅静","志强","小雨","阿杰","明明","静静","小北","阿宁"]
NICK_STYLES = [
    lambda r: r.choice(["爱吃火锅的","不想上班的","在西安的","早睡早起的","考研中的","养猫的","跑步的","摄影的","喜欢旅行的","熬夜的"]) + r.choice(["小王","阿杰","栗子","豆豆","橘子","柚子","阿涛","小鹿","团子","一一"]),
    lambda r: r.choice(SURNAMES) + r.choice(GIVEN),
    lambda r: r.choice(["Lena","Kevin","Mia","Leo","Yuki","Nora","Ethan","Zoe","Ivan","Luna"]) + r.choice(["_","·",""]) + r.choice(["在长安","的日常","Wang","Chen","Li",""]),
    lambda r: r.choice(["月亮","星河","晚风","山海","清风","岁月","青柠","盐汽水","北纬34","长安夜"]) + r.choice(["不睡","很甜","与你","有信","便利店","下的猫",""]),
    lambda r: r.choice(GIVEN) + r.choice(["同学","er","呀","~","本人","酱"]),
]
BIOS = ["人间烟火气，最抚凡人心。","西工大 | 研一 | 每天进步一点点","记录生活，分享日常。","热爱美食、旅行和电影 🎬","工作日打工人，周末徒步党。","INFJ / 摄影 / 咖啡 ☕","考研上岸中，勿扰。","养了两只猫，一只叫花卷，一只叫馒头。","跑步是我对抗焦虑的方式。","从前慢，车马邮件都慢。","一个普通的设计师，偶尔画画。","西安土著，带你吃遍回民街。","看书、听歌、发呆。","读研日常｜科研｜生活碎片","佛系青年，随缘更新。","",""]
AVATARS = list("😀😄😊🙂😎🤓🥳🤠🧐😺🐶🐱🐼🦊🐨🐯🦁🐸🐵🐧🦄🌸🌻🌈⭐🍉🍓🍕🎧📷🎸⚽🏀🎮✈️🚲")
TOPICS = ["美食","旅行","科技","学习","运动","影视","校园","宠物","音乐","摄影","游戏","职场","数码","汽车","读书"]


def normal_persona(rng: random.Random, idx: int) -> dict:
    nick = rng.choice(NICK_STYLES)(rng)
    return {
        "username": f"u{idx:04d}_{''.join(rng.choices(string.ascii_lowercase, k=4))}",
        "nickname": nick,
        "bio": rng.choice(BIOS),
        "avatar": rng.choice(AVATARS),
        "interests": rng.sample(TOPICS, k=rng.randint(2, 4)),
        "activity": rng.choice([0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5]),   # 每日行为强度（含大量潜水用户）
        "night_owl": rng.random() < 0.25,
    }


BOT_NICK_STYLES = [
    lambda r: "用户" + "".join(r.choices(string.digits, k=r.choice([6, 7, 8]))),
    lambda r: "".join(r.choices(string.ascii_lowercase, k=r.randint(5, 8))) + "".join(r.choices(string.digits, k=r.randint(2, 4))),
    lambda r: r.choice(["小","大","阿","老"]) + r.choice(["明","红","强","伟","芳","丽","军","杰","静","磊"]) + "".join(r.choices(string.digits, k=r.choice([3, 4]))),
    lambda r: r.choice(["清风","明月","流年","浮生","初心","如梦","似水","云淡","风轻","星辰"]) + "".join(r.choices(string.digits, k=2)),
]
BOT_BIOS = ["", "", "", "这个人很懒，什么都没留下", "暂无简介", "生活不易，且行且珍惜", "感恩遇见"]


def bot_persona(rng: random.Random, idx: int, group: str, disguise: int = 0) -> dict:
    """disguise=0 朴素水军（数字昵称、空简介、默认头像）；
       disguise>=1 人设伪装：昵称/简介/头像与正常用户同分布，使 profile 类特征失效。"""
    uname = f"b{idx:04d}_{''.join(rng.choices(string.ascii_lowercase, k=3))}"
    if disguise >= 1:
        p = normal_persona(rng, idx)
        p["username"] = uname
        p["group"] = group
        return p
    style = rng.choice(BOT_NICK_STYLES)
    return {
        "username": uname,
        "nickname": style(rng),
        "bio": rng.choice(BOT_BIOS),
        "avatar": rng.choice(["🙂", "🙂", "🙂", "😀", "👤", "👤"]),   # 大量默认头像
        "interests": rng.sample(TOPICS, k=1),
        "group": group,
    }
