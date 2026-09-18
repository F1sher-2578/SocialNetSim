"""内容生成：正常用户帖子/评论 + 水军模板内容（可选 LLM 后端）。

所有内容仅在本地沙箱生成与使用；水军内容的高相似度、模板化特征正是后续检测器要捕捉的对象。
"""
from __future__ import annotations
import os, random, re

# ---------------- 正常用户内容（按主题的模板+槽位） ----------------
NORMAL_TEMPLATES = {
    "美食": ["今天去{place}吃了{food}，{feel}！", "自己在家做{food}，翻车了但还挺好吃 😂", "求推荐{city}好吃的{food}店！", "深夜放毒：{food} 🤤", "{food}配{drink}，这才是周末的正确打开方式"],
    "旅行": ["{city}三日游结束，{feel}，下次还想来。", "在{place}拍到了绝美日落 🌇", "计划十一去{city}，有没有攻略？", "一个人去{city}，反而更自在。", "火车上看风景，{feel}。"],
    "科技": ["新出的{gadget}值得买吗？纠结中。", "把{gadget}升级了，体验{feel}。", "读了一篇关于{tech}的论文，收获很大。", "{tech}真的要改变世界了吗？", "折腾了一晚上{tech}环境，终于跑通了 🎉"],
    "学习": ["图书馆学习打卡第{n}天 📚", "{course}期末复习，头秃。", "今天效率{feel}，完成了{n}个番茄钟。", "求问{course}怎么学比较好？", "论文终于投出去了，等结果。"],
    "运动": ["夜跑{n}公里，{feel}。", "健身第{n}天，坚持就是胜利 💪", "打了两小时球，累但快乐。", "报名了{city}马拉松，开始训练！", "今天骑行去了{place}，风很舒服。"],
    "影视": ["刚看完《{movie}》，{feel}，强烈推荐。", "《{movie}》的结局我不理解…", "周末补番/补剧计划：《{movie}》", "《{movie}》配乐太好听了。", "有人一起看《{movie}》吗？"],
    "校园": ["食堂新开了{food}窗口，排队半小时。", "{course}老师今天讲得{feel}。", "宿舍断电，只能早睡了。", "校园里的银杏黄了 🍂", "社团招新好热闹！"],
    "宠物": ["我家猫今天又把{thing}打翻了 🐱", "遛狗遇到了另一只柯基，{feel}。", "猫咪睡姿合集 😸", "给狗子洗澡，全身湿透的是我。", "捡到一只流浪猫，求收养建议。"],
    "音乐": ["单曲循环《{song}》一整天。", "去看了{singer}的演唱会，{feel}！", "学吉他第{n}天，手指疼。", "求推荐适合学习时听的歌。", "耳机坏了，人生失去了色彩。"],
    "摄影": ["{place}随拍 📷", "换了新镜头，{feel}。", "学修图第{n}天。", "胶片扫描回来了，颗粒感真好。", "日出没等到，等到了雾。"],
    "游戏": ["今晚开黑，缺一个。", "{game}新赛季上分好难。", "终于通关了{game}，{feel}。", "被队友坑了一晚上 😤", "{game}的剧情比我想的好。"],
    "职场": ["加班到十点，打工人的日常。", "面试了{n}家公司，等结果中。", "今天摸鱼被抓，尴尬。", "第一次带新人，感觉{feel}。", "周报写了两小时。"],
    "数码": ["{gadget}续航实测：{n}小时。", "新手机拍照{feel}。", "键盘换了轴，手感{feel}。", "有没有人用{gadget}，求真实评价。", "清理了电脑，快了不少。"],
    "汽车": ["提车一个月，油耗{n}个。", "自驾去{city}，路况不错。", "洗车刚洗完就下雨…", "新能源到底值不值？", "堵在高速上两小时。"],
    "读书": ["读完《{book}》，{feel}。", "本月读书目标：{n}本。", "《{book}》里的一句话让我想了很久。", "二手书店淘到宝了。", "求推荐好看的悬疑小说。"],
}
SLOTS = {
    "place": ["回民街","大唐不夜城","钟楼","曲江","秦岭","大雁塔","永兴坊","小寨","城墙","太白山"],
    "food": ["羊肉泡馍","肉夹馍","凉皮","biangbiang面","火锅","烤肉","小龙虾","寿司","烩麻食","葫芦头"],
    "drink": ["冰峰","奶茶","咖啡","啤酒","酸梅汤"],
    "city": ["成都","重庆","西安","杭州","青岛","大理","厦门","长沙","北京","上海"],
    "feel": ["超级满足","有点失望","非常惊喜","还不错","一言难尽","超出预期","一般般","太爽了","很治愈","很值"],
    "gadget": ["新款手机","降噪耳机","平板","机械键盘","智能手表","相机","显示器"],
    "tech": ["大模型","多模态","强化学习","扩散模型","具身智能","RAG","联邦学习"],
    "course": ["机器学习","信号与系统","操作系统","高数","数据结构","AI安全","计算机网络"],
    "n": [str(i) for i in range(1, 40)],
    "movie": ["流浪地球","星际穿越","让子弹飞","盗梦空间","三体","繁花","漫长的季节","沙丘"],
    "thing": ["水杯","花盆","键盘","泡面","台灯"],
    "song": ["晴天","夜曲","起风了","孤勇者","富士山下","平凡之路"],
    "singer": ["周杰伦","陈奕迅","五月天","邓紫棋","薛之谦"],
    "game": ["原神","王者荣耀","黑神话","星露谷","塞尔达","文明6"],
    "book": ["活着","三体","百年孤独","人类简史","围城","小王子","平凡的世界"],
}
EMOJIS = ["", "", "", " 😊", " 😂", " 🥲", " ✨", " 🙌", " 👍", " ❤️", " 🤔"]
HASHTAG_POOL = ["#日常", "#生活记录", "#今日份快乐", "#打卡", "#分享", "#随手拍", "#碎碎念", "#周末", "#学习", "#西安"]

NORMAL_COMMENTS = ["哈哈哈太真实了", "同感！", "羡慕了", "求地址", "这个我也想去", "太好看了吧", "加油加油", "笑死我了", "确实", "好家伙", "这也太可爱了", "有被治愈到", "支持！", "我也是这样", "哈哈哈哈", "感谢分享", "厉害了", "吃了吗？", "居然是这样", "一样一样", "冲冲冲", "怎么做到的？", "路过点赞", "好想去啊", "哇，好棒", "记得早点休息", "求推荐", "666", "学到了", "这也可以？"]


def fill(template: str, rng: random.Random) -> str:
    def rep(m):
        key = m.group(1)
        return rng.choice(SLOTS.get(key, ["…"]))
    return re.sub(r"\{(\w+)\}", rep, template)


def normal_post(rng: random.Random, interests: list[str]) -> tuple[str, str]:
    topic = rng.choice(interests) if interests else rng.choice(list(NORMAL_TEMPLATES))
    text = fill(rng.choice(NORMAL_TEMPLATES[topic]), rng) + rng.choice(EMOJIS)
    tags = []
    if rng.random() < 0.5:
        tags = rng.sample(HASHTAG_POOL, k=rng.randint(1, 2))
        if rng.random() < 0.5:
            tags.append(f"#{topic}")
    return text + (" " + " ".join(tags) if tags else ""), " ".join(tags)


def normal_comment(rng: random.Random) -> str:
    return rng.choice(NORMAL_COMMENTS) + rng.choice(EMOJIS)


# ---------------- 水军模板内容 ----------------
BOOST_COMMENTS = ["这个真的好用，已经回购三次了！", "亲测有效，推荐给大家", "楼主说得太对了，支持！", "质量非常好，物超所值", "用了一周，效果明显，大家可以试试", "良心推荐，绝对不踩雷", "支持支持，必须顶起来", "已下单，期待到货", "太棒了，正是我需要的", "五星好评，值得入手", "买它！买它！", "这家真的靠谱，朋友都在用", "顶一下，好东西要分享", "感谢楼主推荐，很实用", "已经安利给全家了"]
BOOST_POSTS = ["最近发现一个宝藏{product}，{praise}！{cta}", "{praise}的{product}，谁用谁知道 {tag}", "用了{product}之后，{praise}，强烈推荐 {tag}", "被朋友安利了{product}，{praise}！{cta}"]
SMEAR_COMMENTS = ["这就是个骗子，大家别信", "垃圾产品，用一次就坏了", "楼主收钱了吧？说话这么假", "呵呵，智商税罢了", "这种货色也好意思发出来", "全是托，别被骗了", "差评！退款都不给", "假的，别买，我被坑了", "又是水军在刷，恶心", "脑子有问题才会买这个", "骗子公司，投诉了", "这么烂的东西也有人吹？", "傻子才信这个", "滚吧，别来恶心人", "劝大家远离，全是套路"]
SMEAR_POSTS = ["曝光{product}，{blame}，大家避雷！{tag}", "{product}翻车了！{blame}，千万别买 {tag}", "实名举报{product}：{blame}。{tag}"]
HASHTAG_PUSH_POSTS = ["{tag} 今天也要{slogan}！", "{tag} 大家一起来支持，{slogan}", "转发扩散 {tag} {slogan}！！！", "{tag} {slogan}，冲上热搜", "{slogan}！{tag} {tag}"]
AD_SCAM_COMMENTS = ["加V信 xk8899 领取内部优惠", "兼职刷单日入三百，私信我", "低价代购，需要的加微信 lala2024", "点击链接领红包 t.cn/xxxx", "急用钱？无抵押秒批，联系QQ 123456789", "棋牌平台首充返水50%，速来", "看片加群，你懂的", "招代理，躺赚，v：abc123"]
BOT_SLOTS = {
    "product": ["XX精华液","星光洗面奶","云端奶茶","极速充电宝","飞鹿跑鞋","晨光咖啡","悦享护肤套装","飞鸟耳机"],
    "praise": ["效果超好","性价比爆表","真的太好用","质量一流","颜值在线"],
    "cta": ["赶紧冲！","链接在评论区","评论区自取","私我拿优惠"],
    "tag": ["#好物推荐","#必买清单","#年度最佳","#支持国货","#避雷指南","#良心测评","#热搜"],
    "blame": ["用完过敏","客服态度极差","虚假宣传","货不对板","质量太差"],
    "slogan": ["为XX打call","支持到底","一起加油","不要输","顶上去"],
}
VARIATIONS_PUNCT = ["", "！", "!!", "~", "。", "…", " ", "！！！"]
SYNONYM_SWAP = {"真的": ["确实", "真心", "实在"], "推荐": ["安利", "种草"], "大家": ["各位", "朋友们"], "太": ["很", "超", "非常"], "好用": ["实用", "好使"], "支持": ["顶", "赞成"], "骗子": ["骗人的", "诈骗犯"], "垃圾": ["破烂", "劣质"]}


def _fill_bot(template: str, rng: random.Random) -> str:
    def rep(m):
        key = m.group(1)
        return rng.choice(BOT_SLOTS.get(key, SLOTS.get(key, ["…"])))
    return re.sub(r"\{(\w+)\}", rep, template)


def mutate(text: str, rng: random.Random, strength: float = 0.5) -> str:
    """对模板文本做轻微变体（同义词替换/标点/表情），模拟水军“换皮”行为。"""
    out = text
    for k, vs in SYNONYM_SWAP.items():
        if k in out and rng.random() < strength * 0.6:
            out = out.replace(k, rng.choice(vs), 1)
    if rng.random() < strength:
        out = out.rstrip("！!。~") + rng.choice(VARIATIONS_PUNCT)
    if rng.random() < strength * 0.6:
        out += rng.choice(["", " 👍", " 🔥", " 💯", " ✨", " 😍"])
    return out


def bot_comment(strategy: str, rng: random.Random, variation: float = 0.5) -> str:
    if strategy == "boost":
        base = rng.choice(BOOST_COMMENTS)
    elif strategy == "smear":
        base = rng.choice(SMEAR_COMMENTS)
    elif strategy == "spam":
        base = rng.choice(AD_SCAM_COMMENTS)
    else:
        base = rng.choice(BOOST_COMMENTS)
    return mutate(base, rng, variation)


def bot_post(strategy: str, rng: random.Random, target_tag: str = "", variation: float = 0.5) -> tuple[str, str]:
    if strategy == "boost":
        t = _fill_bot(rng.choice(BOOST_POSTS), rng)
    elif strategy == "smear":
        t = _fill_bot(rng.choice(SMEAR_POSTS), rng)
    elif strategy == "hashtag_push":
        t = _fill_bot(rng.choice(HASHTAG_PUSH_POSTS).replace("{tag}", target_tag or "#热搜"), rng)
    elif strategy == "spam":
        t = rng.choice(AD_SCAM_COMMENTS)
    else:
        t = _fill_bot(rng.choice(BOOST_POSTS), rng)
    t = mutate(t, rng, variation)
    tags = " ".join(re.findall(r"#[\w一-鿿]+", t))
    return t, tags


# ---------------- 可选 LLM 后端 ----------------
def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def llm_generate(prompt: str, max_tokens: int = 120) -> str | None:
    """若配置了 API key 则用大模型生成更自然的文本，否则返回 None。"""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(model=os.environ.get("SOCIALSIM_LLM", "claude-sonnet-5"), max_tokens=max_tokens,
                                         messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text.strip()
    except Exception:
        return None
    return None
