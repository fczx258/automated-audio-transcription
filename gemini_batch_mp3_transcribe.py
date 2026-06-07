# -*- coding: utf-8 -*-
"""
批量将文件夹中的 MP3 交给 Gemini 多模态 API，生成 UTF-8 文本（.txt）。

依赖: pip install google-genai
密钥（优先级从高到低）:
  1) 环境变量 GOOGLE_API_KEY 或 GEMINI_API_KEY
  2) 本文件下方的 API_KEY_IN_THIS_FILE（勿上传公开仓库；密钥若曾截图建议轮换）

默认输出：当前用户桌面下的「录音转换」文件夹（未指定 -o 时）。可用 -o 改成其它目录。

代理：HTTP_PROXY_URL 必须与本机 V2Ray「入站」端口一致（表格下载脚本里若写 10809，你电脑上可能只有 10808 在监听，会报 10061 拒绝连接）。
另：与表格脚本相同，启动时会关闭 SSL 证书校验（仅适合本机调试）。

SOCKS5 示例：socks5://127.0.0.1:10808（需 pip install "httpx[socks]"）。

若接口返回「User location is not supported」：这是 Google 服务器根据出口 IP 做的限制，
无法靠改模型名、改保存路径或「多写几行代码」绕过。可行方向（遵守当地法规前提下）：
  · 让 Python 流量经「API 认可地区」出口：换代理节点、开 TUN/全局、确认 google 相关域名走代理；
  · 运行环境换到境外：如 Colab、海外 VPS 上跑同一脚本；
  · 换产品线：Vertex AI（GCP 计费）等；
  · 换能力：本地 Whisper / 其它语音识别服务做转写，不经过 Gemini API。

启动转录时**自动**用代码请求 ipwho.is 打印当前出口 IP 与国家（与即将调 Gemini 的代理/TUN 设置一致），无需手打 --ip。
更详细双源检测仍可用：python 本脚本.py --ip ；TUN 可加 --no-proxy。
加 --skip-egress-check 可跳过启动时这次检测以节省时间。

免费层建议：控制每分钟请求数（默认每个文件处理完后休眠 4 秒，约 ≤15 次/分钟）。
模型可通过环境变量 GEMINI_MODEL 覆盖；默认 gemini-1.5-flash（与 google.generativeai 常用名称一致）。

Code Runner 只会执行「python 本文件.py」，不会带上 -i；请设置 DEFAULT_INPUT_DIR，或在终端手动加参数。
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import ssl
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import urllib3

from google import genai
from google.genai import errors, types

# 若不想每次设环境变量，可把 AI Studio 里的密钥粘贴到下方引号内（不要提交到 Git / 不要发给别人）。
API_KEY_IN_THIS_FILE = ""

# Code Runner 直接运行时没有命令行参数：填写 MP3 所在文件夹；留空则必须在终端使用 -i "路径"
DEFAULT_INPUT_DIR = r"D:\BaiduNetdiskDownload\胡y悦【水晶疗愈系列课程】证书班\【水晶转运全科班】水晶摆阵与能量感应：链接能量，疗愈身心，你也能成为仪式达人"
# 仅当「未传任何参数启动」且使用上一行目录时，是否递归子文件夹（等价于命令行 -r）
DEFAULT_RUN_RECURSIVE = True

# 本机代理：在 V2RayN「设置-入站」看实际端口。常见 10808=SOCKS 或混合，10809=仅 HTTP（无监听会 WinError 10061）
# 表格下载代码若用 10809 而你这里是 10808，请改成与系统代理一致的端口。
HTTP_PROXY_URL = "http://127.0.0.1:10808"

DEFAULT_PROMPT = """请根据所附音频，输出一份可供阅读的文档。

要求：
1. 使用简体中文，尽可能完整、准确地转写口语音频内容。
2. 适当分段并加小标题，便于复习；口语中的重复、口癖可适度精简，但不要遗漏知识点。
3. 若听不清某句，用【听不清】占位并注明大致位置。
4. 结构清晰即可（小标题、分段）；保存为 .txt，避免复杂排版符号。

不要输出与转写无关的寒暄或自我说明。"""


def _desktop_dir() -> Path:
    """当前用户桌面（兼容中文系统「桌面」与英文 Desktop）。"""
    home = Path.home()
    for sub in ("Desktop", "桌面"):
        d = home / sub
        if d.is_dir():
            return d
    return home / "Desktop"


def _default_transcript_output_dir() -> Path:
    """默认：桌面/录音转换"""
    return _desktop_dir() / "录音转换"


def apply_network_stack_like_drive_script() -> None:
    """与 20241106谷歌店铺数据下载代码.py 一致：关 SSL 校验、清证书环境变量、静默 urllib3 告警。"""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore


def _hint_proxy_connect_failed(exc: BaseException) -> None:
    if "10061" in str(exc) or "拒绝" in str(exc) or "refused" in str(exc).lower():
        print(
            "\n>>> 连不上本机代理（例如 10061 积极拒绝）。说明当前 HTTP_PROXY_URL 里的「IP:端口」上没有服务在听。\n"
            ">>> 请打开 V2RayN 核对入站端口：有 HTTP 入站可用 http://127.0.0.1:该端口；"
            "若只有 SOCKS，改为 socks5://127.0.0.1:10808 并 pip install \"httpx[socks]\"。\n"
            ">>> 与「谷歌表格下载」脚本端口不必相同，以你本机实际为准。\n",
            file=sys.stderr,
        )


def _is_gemini_location_blocked(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "location" in s and "not supported" in s


def _hint_if_location_blocked(
    exc: BaseException,
    using_proxy: str | None = None,
    startup_country_code: str | None = None,
) -> None:
    if not _is_gemini_location_blocked(exc):
        return
    print(
        "\n>>> 说明：「User location is not supported」是 Google 返回的业务错误，表示「当前这次请求」的出口 "
        "仍被判定在 Gemini Developer API 不可用地区；与脚本写法、保存路径无关。\n",
        file=sys.stderr,
    )
    if using_proxy:
        if startup_country_code == "US":
            print(
                ">>> 你启动时「[网络]」已显示美国 (US)，仍报地区不可用：说明 Google 不只按 ipwho 这类「公网 IP 国家」判断。\n"
                ">>> 常见情况：数据中心/VPN/部分 ASN 仍被策略拦截；或 AI Studio 账号/项目地区与 API 策略有关。\n"
                ">>> 可尝试：换「非机房/家宽」类美国节点或其它国家节点；在 ai.google.dev 重新创建 API Key；"
                "在海外 VPS 或 Google Colab 运行同一脚本；或使用 Vertex AI / 本地 Whisper 等替代转录。\n",
                file=sys.stderr,
            )
        else:
            print(
                ">>> 你已配置代理且能连上 Google（否则不会是这条 JSON 提示）。接下来请在代理软件里：\n"
                ">>>   · 换「美国等常用开放区」节点再试；香港/部分机房也可能仍被拒。\n"
                ">>>   · 确认 generativelanguage.googleapis.com 走代理，不要规则里直连。\n"
                ">>>   · 可开 TUN / 全局排除遗漏；启动时脚本已自动打印 [网络] 出口，也可运行：python 本脚本.py --ip\n",
                file=sys.stderr,
            )
    else:
        print(
            ">>> 未配置代理时：需使用 API 支持的地区的网络出口（请遵守当地法规），或改用 Colab/境外主机跑脚本。\n",
            file=sys.stderr,
        )


def _resolve_http_proxy(cli_proxy: str | None) -> str:
    """优先级：命令行 --proxy > 脚本 HTTP_PROXY_URL > 环境变量（HTTPS_PROXY / HTTP_PROXY / ALL_PROXY / GEMINI_HTTP_PROXY）。"""
    for candidate in (
        (cli_proxy or "").strip(),
        (HTTP_PROXY_URL or "").strip(),
        (os.environ.get("GEMINI_HTTP_PROXY") or "").strip(),
        (os.environ.get("HTTPS_PROXY") or "").strip(),
        (os.environ.get("HTTP_PROXY") or "").strip(),
        (os.environ.get("ALL_PROXY") or "").strip(),
    ):
        if candidate:
            return candidate
    return ""


def _client_http_options(cli_proxy: str | None) -> types.HttpOptions | None:
    proxy = _resolve_http_proxy(cli_proxy)
    if not proxy:
        return None
    # 与 Drive 脚本里 session.verify=False 一致：经本地 HTTP 代理时不对证书做强校验（httpx）
    args: dict = {"proxy": proxy, "trust_env": True, "verify": False}
    return types.HttpOptions(client_args=args.copy(), async_client_args=args.copy())


def _client_http_options_tun_no_explicit_proxy() -> types.HttpOptions:
    """已开 TUN 全局时：不显式连 127.0.0.1:10808，由系统把流量交给 TUN；不信任环境变量里的代理以免混用。"""
    args: dict = {"trust_env": False, "verify": False}
    return types.HttpOptions(client_args=args.copy(), async_client_args=args.copy())


def _httpx_kwargs_egress(cli_proxy: str | None, tun_only: bool) -> dict:
    """与 print_egress_diagnostic / 启动检测 使用相同的代理逻辑。"""
    kw: dict = {"timeout": 15.0, "verify": False}
    if tun_only:
        kw["trust_env"] = False
    else:
        kw["trust_env"] = True
        proxy = _resolve_http_proxy(cli_proxy)
        if proxy:
            kw["proxy"] = proxy
    return kw


def fetch_egress_country(cli_proxy: str | None, tun_only: bool) -> tuple[str, str, str] | None:
    """用代码查询当前出口：返回 (ip, country_code, country_name)，失败返回 None。"""
    try:
        kwargs = _httpx_kwargs_egress(cli_proxy, tun_only)
        with httpx.Client(**kwargs) as client:
            r = client.get("https://ipwho.is/")
            r.raise_for_status()
            data = r.json()
        if data.get("success") is False:
            return None
        ip = str(data.get("ip") or "")
        cc = (data.get("country_code") or "").upper()
        name = str(data.get("country") or "?")
        return (ip, cc, name)
    except Exception:
        return None


def print_startup_egress_line(cli_proxy: str | None, tun_only: bool, skip: bool) -> str | None:
    """每次跑转录前自动打印出口 IP/国家；返回两位国家码（如 US）供错误提示使用。"""
    if skip:
        return None
    path_note = "TUN/不显式代理" if tun_only else (_resolve_http_proxy(cli_proxy) or "无显式代理(直连)")
    print("[网络] 正在自动检测出口 IP / 国家…", flush=True)
    got = fetch_egress_country(cli_proxy, tun_only)
    if got:
        ip, cc, name = got
        print(f"[网络] 对外 IP: {ip}  |  国家/地区: {name} ({cc})  |  路径: {path_note}", flush=True)
        if cc in ("CN", "HK"):
            print(
                "[网络] 提示: 当前出口在 CN/HK，Gemini Developer API 常会报地区不可用，请换美国等节点后再试。",
                file=sys.stderr,
            )
        return cc
    print(
        "[网络] 自动检测失败（可检查代理）。手动详测请运行: python 本脚本.py --ip",
        file=sys.stderr,
    )
    return None


def print_egress_diagnostic(cli_proxy: str | None, tun_only: bool = False) -> None:
    """经与 Gemini 相同的网络路径访问公网，查看对外 IP / 国家。

    V2Ray 不提供「让 Python 直接查询当前出口 IP」的标准接口；用同一代理请求 IP 检测服务，
    就是在看「外网（含 Google）看到的你」。
    """
    if tun_only:
        print("=== 对外 IP / 地区（TUN 模式：不显式代理，由系统路由进 TUN）===\n")
    else:
        proxy = _resolve_http_proxy(cli_proxy)
        print("=== 对外 IP / 地区（与脚本里 HTTP_PROXY_URL / --proxy 一致）===\n")
        print(f"代理: {proxy or '（无 — 下面为直连）'}\n")
    try:
        kw = _httpx_kwargs_egress(cli_proxy, tun_only)
        kw["timeout"] = 25.0
        with httpx.Client(**kw) as client:
            r = client.get("https://ipwho.is/")
            r.raise_for_status()
            data = r.json()
            if data.get("success") is False:
                print("ipwho.is 返回失败:", data, file=sys.stderr)
                return
            ip1 = data.get("ip")
            country = data.get("country")
            cc = (data.get("country_code") or "").upper()
            print(f"[ipwho.is]  IP: {ip1}")
            print(f"[ipwho.is]  国家/地区: {country} ({cc})")
            ip2 = None
            try:
                r2 = client.get("https://api.ipify.org?format=json")
                r2.raise_for_status()
                ip2 = r2.json().get("ip")
                print(f"[ipify]     IP: {ip2}")
            except Exception as ex:
                print(f"[ipify]     未获取: {ex}")
            if ip1 and ip2 and str(ip1) != str(ip2):
                print(
                    "\n注意: 两个服务看到的 IP 不一致，可能存在分流/多出口，Gemini 以实际连 Google 的为准。\n",
                    file=sys.stderr,
                )

        print("\n--- 和 Gemini 的关系 ---")
        print(
            "若这里是 CN/HK，而 Gemini 报 User location is not supported，属正常；"
            "请在 V2RayN 里选「美国」等节点 → 应用 → 再运行本命令确认国家已变。\n"
            "若已是 US 等仍报错：换节点、规则里让 *.googleapis.com 走代理、或开 TUN。\n"
            "连接失败时：10808 可能是 SOCKS，试 HTTP_PROXY_URL=socks5://127.0.0.1:10808 "
            '并 pip install "httpx[socks]"。\n'
        )
    except Exception as e:
        print(f"检测失败: {e}", file=sys.stderr)
        print(
            '提示: SOCKS5 请用 socks5:// 前缀并执行 pip install "httpx[socks]"。',
            file=sys.stderr,
        )


def _api_key() -> str:
    key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or (API_KEY_IN_THIS_FILE or "").strip()
    )
    if not key:
        print(
            "请设置环境变量 GOOGLE_API_KEY / GEMINI_API_KEY，"
            "或在脚本里填写 API_KEY_IN_THIS_FILE（见文件顶部说明）。",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _wait_file_ready(client: genai.Client, f: types.File, poll_s: float = 15.0) -> types.File:
    name = f.name
    cur = f
    while cur.state == types.FileState.PROCESSING:
        time.sleep(poll_s)
        cur = client.files.get(name=name)
    if cur.state != types.FileState.ACTIVE:
        raise RuntimeError(f"上传文件未就绪: state={cur.state!r}, name={name!r}")
    return cur


def _generate_with_retries(
    client: genai.Client,
    model: str,
    uploaded: types.File,
    prompt: str,
    max_retries: int = 5,
) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, uploaded],
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("模型返回空文本，可能被安全策略拦截或响应异常。")
            return text
        except errors.ClientError as e:
            last_err = e
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code == 429 and attempt < max_retries - 1:
                wait = min(120.0, (2**attempt) * 15.0) + random.uniform(0, 3.0)
                print(f"  遇到 429 限流，{wait:.0f} 秒后重试 ({attempt + 1}/{max_retries})…")
                time.sleep(wait)
                continue
            raise
    assert last_err is not None
    raise last_err


def _local_path_for_upload(src: Path) -> tuple[Path, Path | None]:
    """google-genai 会把本机文件名放进 HTTP 头 X-Goog-Upload-File-Name，必须为 ASCII。

    若原名含中文等字符，则复制到临时目录下仅含 [0-9a-f] 的文件名再上传；返回 (上传路径, 临时文件或 None)。
    """
    try:
        src.name.encode("ascii")
        return src, None
    except UnicodeEncodeError:
        suffix = src.suffix if src.suffix else ".mp3"
        tmp = Path(tempfile.gettempdir()) / f"gemini_up_{uuid.uuid4().hex}{suffix}"
        shutil.copy2(src, tmp)
        return tmp, tmp


def collect_mp3(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(root.rglob("*.mp3")) + sorted(root.rglob("*.MP3"))
    else:
        files = sorted(root.glob("*.mp3")) + sorted(root.glob("*.MP3"))
    # 去重（大小写重复时）
    seen: set[Path] = set()
    out: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量 MP3 → 文本 .txt（Gemini API）",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="包含 MP3 的文件夹路径（省略时使用脚本中的 DEFAULT_INPUT_DIR）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出目录（默认: 桌面/录音转换）",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite"),
        help="模型名；也可用环境变量 GEMINI_MODEL（默认 gemini-1.5-flash）",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="递归子文件夹",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=4.0,
        help="每个文件处理完成后休眠秒数，降低 RPM 触发 429 的概率（默认 4）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标 .txt 已存在则跳过",
    )
    parser.add_argument(
        "--delete-uploaded",
        action="store_true",
        help="成功后删除云端已上传的临时文件",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多处理多少个文件（0 表示不限制；便于控制每日配额）",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="从文本文件读取提示词（UTF-8），覆盖默认转写说明",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP/SOCKS 代理 URL，如 http://127.0.0.1:7890（覆盖环境变量与脚本内 HTTP_PROXY_URL）",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="不显式使用 HTTP_PROXY_URL/环境变量代理（适合已开 V2RayN TUN+全局，由系统把流量交给 TUN）",
    )
    parser.add_argument(
        "--check-egress",
        "--ip",
        action="store_true",
        dest="check_egress",
        help="仅检测对外 IP/国家后退出；配合 --no-proxy 可测 TUN 路径（同 --ip）",
    )
    parser.add_argument(
        "--skip-egress-check",
        action="store_true",
        help="跳过启动时自动检测出口 IP/国家（默认每次转录前会自动查一次）",
    )
    args = parser.parse_args()

    if args.check_egress:
        print_egress_diagnostic(args.proxy, tun_only=args.no_proxy)
        sys.exit(0)

    resolved_input_via_default = False
    if args.input is None:
        fallback = (DEFAULT_INPUT_DIR or "").strip()
        if not fallback:
            parser.error(
                "缺少 MP3 文件夹：请在终端使用 -i \"路径\"，或在脚本里设置 DEFAULT_INPUT_DIR。"
            )
        args.input = Path(fallback)
        resolved_input_via_default = True
    if resolved_input_via_default and DEFAULT_RUN_RECURSIVE:
        args.recursive = True

    input_dir = args.input.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"输入路径不是文件夹: {input_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = (
        args.output.expanduser().resolve()
        if args.output
        else _default_transcript_output_dir().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    else:
        prompt = DEFAULT_PROMPT

    mp3_list = collect_mp3(input_dir, args.recursive)
    if args.max_files > 0:
        mp3_list = mp3_list[: args.max_files]

    if not mp3_list:
        print("未找到任何 .mp3 文件。")
        sys.exit(0)

    startup_cc = print_startup_egress_line(args.proxy, args.no_proxy, args.skip_egress_check)

    if args.no_proxy:
        http_opts = _client_http_options_tun_no_explicit_proxy()
        net_line = "  |  网络: TUN/系统路由（--no-proxy）"
        net_tag = "TUN/系统路由"
    else:
        http_opts = _client_http_options(args.proxy)
        proxy_show = _resolve_http_proxy(args.proxy)
        net_line = f"  |  代理: {proxy_show}" if proxy_show else ""
        net_tag = proxy_show or None
    client = genai.Client(api_key=_api_key(), http_options=http_opts)
    print(f"模型: {args.model}  |  共 {len(mp3_list)} 个文件  |  输出: {out_dir}{net_line}")

    for idx, mp3_path in enumerate(mp3_list, start=1):
        rel = mp3_path.relative_to(input_dir) if mp3_path.is_relative_to(input_dir) else mp3_path.name
        safe_stem = rel.as_posix().replace("/", "__").replace("\\", "__")
        out_path = out_dir / f"{safe_stem}.txt"

        if args.skip_existing and out_path.exists():
            print(f"[{idx}/{len(mp3_list)}] 跳过（已存在）: {out_path.name}")
            continue

        print(f"[{idx}/{len(mp3_list)}] 处理: {mp3_path.name}")
        uploaded: types.File | None = None
        tmp_upload: Path | None = None
        try:
            upload_src, tmp_upload = _local_path_for_upload(mp3_path)
            uploaded = client.files.upload(file=os.fspath(upload_src))
            uploaded = _wait_file_ready(client, uploaded)
            text = _generate_with_retries(client, args.model, uploaded, prompt)
            header = (
                f"{mp3_path.stem}\n"
                f"来源文件: {mp3_path.name}\n"
                f"{'-' * 40}\n\n"
            )
            out_path.write_text(header + text.strip() + "\n", encoding="utf-8")
            print(f"  已保存: {out_path}")
        except httpx.ConnectError as e:
            _hint_proxy_connect_failed(e)
            raise
        except errors.ClientError as e:
            _hint_if_location_blocked(e, net_tag, startup_cc)
            if _is_gemini_location_blocked(e):
                print(
                    "\n[退出] 地区策略阻止本次 Gemini 调用；已输出说明，不再打印完整堆栈。",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise
        finally:
            if tmp_upload is not None and tmp_upload.exists():
                try:
                    tmp_upload.unlink()
                except OSError:
                    pass
            if uploaded and uploaded.name and args.delete_uploaded:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception as de:
                    print(f"  （删除云端文件失败，可忽略）{de}", file=sys.stderr)

        if idx < len(mp3_list) and args.min_interval > 0:
            time.sleep(args.min_interval)

    print("全部完成。")


if __name__ == "__main__":
    apply_network_stack_like_drive_script()
    main()
