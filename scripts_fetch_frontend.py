"""拉取前端第三方文件到 socialsim/app/static/vendor/ 并校验 SHA-256（只用标准库，不需要 Node）。

vendor 文件已经提交进仓库，平时不用运行；只在升级版本或文件损坏时使用。
走 npmmirror 的单文件直取接口，国内网络可直接访问。选型依据见 docs/调研_前端界面模板与可视化.md。
"""
import hashlib
import urllib.request
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "socialsim" / "app" / "static" / "vendor"
MIRROR = "https://registry.npmmirror.com"
# (包, 版本, 包内路径, 本地相对路径, sha256；许可证文件不校验)
FILES = [
    ("vue", "3.5.44", "dist/vue.global.prod.js", "vue/3.5.44/vue.global.prod.js",
     "e0a2a0866a8de8887542e24f3c218aad3d907b88f30966068792ba992296dc9f"),
    ("element-plus", "2.14.7", "dist/index.full.min.js", "element-plus/2.14.7/index.full.min.js",
     "7f38e2550c046e823d84ed16c2a70caaca6d7e7071b060c96526766d46d56ab1"),
    ("element-plus", "2.14.7", "dist/index.css", "element-plus/2.14.7/index.css",
     "d78229097f1ec25311afa285e86aa5af6e768d541a65dd0c6d4c7ac0831b02f0"),
    ("element-plus", "2.14.7", "dist/locale/zh-cn.min.js", "element-plus/2.14.7/locale/zh-cn.min.js",
     "7909a2cfe5b826f0686a7abb4845fe8c9e0bf3d31b90a9a00f0d3052b5c217e6"),
    ("@element-plus/icons-vue", "2.3.2", "dist/index.iife.min.js", "element-plus-icons/2.3.2/index.iife.min.js",
     "b6d3da03d73bfa394a28f442bf896b31119f502ff3b59608bb73a88562e5d306"),
    ("echarts", "6.1.0", "dist/echarts.min.js", "echarts/6.1.0/echarts.min.js",
     "b66b25aeb4df84e33199dc21694014d336d222cbd9deb0e5a7c14bd6aa0d0fd0"),
    ("vue", "3.5.44", "LICENSE", "vue/3.5.44/LICENSE", None),
    ("element-plus", "2.14.7", "LICENSE", "element-plus/2.14.7/LICENSE", None),
    ("@element-plus/icons-vue", "2.3.2", "LICENSE", "element-plus-icons/2.3.2/LICENSE", None),
    ("echarts", "6.1.0", "LICENSE", "echarts/6.1.0/LICENSE", None),
    ("echarts", "6.1.0", "NOTICE", "echarts/6.1.0/NOTICE", None),     # Apache-2.0 要求分发时附带
]


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main():
    for pkg, ver, path, dest, sha in FILES:
        out = VENDOR / dest
        if out.exists() and (sha is None or hashlib.sha256(out.read_bytes()).hexdigest() == sha):
            print("已存在", dest)
            continue
        data = get(f"{MIRROR}/{pkg}/{ver}/files/{path}")
        if sha is not None and hashlib.sha256(data).hexdigest() != sha:
            raise SystemExit(f"校验失败：{pkg}@{ver}/{path}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print("下载", dest, f"{len(data):,} B")


if __name__ == "__main__":
    main()
