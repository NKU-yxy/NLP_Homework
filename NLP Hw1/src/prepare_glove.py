"""从 Stanford 官方压缩包中仅下载并校验作业需要的百维词向量。"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.client import IncompleteRead
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
import zlib


ROOT = Path(__file__).resolve().parents[1]  # 当前实验的根目录。
URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"  # 作业指定资源的官方跳转地址。
MEMBER = "glove.6B.100d.txt"  # 仅使用百维版本。


def fetch_range(byte_range):
    """读取指定字节区间并在临时网络失败后重试。"""
    for attempt in range(8):
        try:
            request = Request(URL, headers={"Range": f"bytes={byte_range}"})
            with urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise ValueError("服务器未返回分段内容，不能安全提取压缩包。")
                content_range = response.headers.get("Content-Range", "")
                data = response.read()
                span = content_range.split()[1].split("/")[0]
                start, end = map(int, span.split("-"))
                if len(data) != end - start + 1:
                    raise ValueError("分段下载长度不完整。")
                if not byte_range.startswith("-") and span != byte_range:
                    raise ValueError("服务器返回的区间与请求不一致。")
                return data
        except (OSError, ValueError, IncompleteRead):
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 10))


def cached_range(byte_range, directory):
    """复用下载成功的分段，避免网络中断后从头下载。"""
    path = directory / byte_range
    start, end = map(int, byte_range.split("-"))
    if path.exists() and path.stat().st_size == end - start + 1:
        return path.read_bytes()
    data = fetch_range(byte_range)
    path.write_bytes(data)
    return data


def find_member(tail):
    """从压缩包中央目录读取目标文件的大小、偏移和校验值。"""
    position = tail.find(b"PK\x01\x02")
    while position >= 0 and tail[position:position + 4] == b"PK\x01\x02":
        fields = struct.unpack_from("<4s6H3I5H2I", tail, position)
        name_size, extra_size, comment_size = fields[10:13]
        name = tail[position + 46:position + 46 + name_size].decode("utf-8")
        if name == MEMBER:
            if fields[4] != 8:
                raise ValueError("目标文件不是预期的压缩格式。")
            return {"crc32": fields[7], "compressed_size": fields[8],
                    "size": fields[9], "offset": fields[16]}
        position += 46 + name_size + extra_size + comment_size
    raise ValueError("官方压缩包中没有找到百维词向量。")


def validate_file(path, metadata):
    """使用官方压缩包记录的长度与校验值验证本地文件。"""
    if not path.exists() or path.stat().st_size != metadata["size"]:
        return False
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == metadata["crc32"]


def main():
    """分段下载目标词向量、校验完整性并清理下载临时文件。"""
    directory = ROOT / "data/glove"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MEMBER
    metadata = find_member(fetch_range("-65536"))
    if validate_file(target, metadata):
        print(f"现有词向量完整性校验通过：{target}")
        return
    offset = metadata["offset"]
    header = fetch_range(f"{offset}-{offset + 29}")
    fields = struct.unpack("<4s5H3I2H", header)
    if fields[0] != b"PK\x03\x04":
        raise ValueError("压缩包局部文件头错误。")
    start = offset + 30 + fields[-2] + fields[-1]  # 跳过文件名和扩展字段。
    chunk_size = 4 * 1024 * 1024  # 每个请求读取四兆字节。
    ranges = [f"{start + part}-{start + min(part + chunk_size, metadata['compressed_size']) - 1}"
              for part in range(0, metadata["compressed_size"], chunk_size)]
    temporary = target.with_suffix(".txt.part")
    cache = directory / ".download_chunks"
    cache.mkdir(exist_ok=True)
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)  # 解压压缩包内的原始数据流。
    checksum, written = 0, 0
    with ThreadPoolExecutor(max_workers=2) as executor, temporary.open("wb") as handle:
        for index, compressed in enumerate(executor.map(partial(cached_range, directory=cache), ranges), 1):
            data = decompressor.decompress(compressed)
            handle.write(data)
            checksum = zlib.crc32(data, checksum)
            written += len(data)
            print(f"词向量下载进度：{index}/{len(ranges)} 段", flush=True)
        final = decompressor.flush()
        handle.write(final)
        checksum = zlib.crc32(final, checksum)
        written += len(final)
    if not decompressor.eof or written != metadata["size"] or checksum != metadata["crc32"]:
        raise ValueError("词向量完整性校验失败，请重新下载。")
    temporary.replace(target)
    for byte_range in ranges:
        (cache / byte_range).unlink(missing_ok=True)
    cache.rmdir()
    print(f"词向量长度与校验值均正确：{target}")


if __name__ == "__main__":
    main()
