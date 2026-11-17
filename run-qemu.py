#!/usr/bin/env python3
"""Boot the slate-translated kernel under QEMU with a busybox shell.

    ./run-qemu.py            translated vmlinux (<linux>/build-slate), rebuilding bzImage if vmlinux is newer
    ./run-qemu.py --native   native clang kernel (build-clang), for comparison
    ./run-qemu.py -- -s -S   extra arguments go to qemu (here: wait for gdb on :1234)

Quit with Ctrl-A X, or `poweroff` in the shell.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOT = HERE / "boot"
BUSYBOX = BOOT / "busybox"
INIT = """#!/bin/sh
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
mount -t tmpfs tmpfs /tmp
echo
echo "{banner}: $(uname -a)"
setsid cttyhack sh
poweroff -f
"""


def bzimage(linux, build):
    image = build / "arch/x86/boot/bzImage"
    vmlinux = build / "vmlinux"
    if not image.exists() or image.stat().st_mtime < vmlinux.stat().st_mtime:
        print(f"building {image} from {vmlinux}", file=sys.stderr)
        subprocess.run(["make", f"O={build.name}", "CC=clang", "HOSTCC=clang", f"-j{os.cpu_count()}", "bzImage"],
                       cwd=linux, check=True, stdout=subprocess.DEVNULL)
    return image


def initramfs(banner):
    target = BOOT / f"initramfs-{banner.split()[0]}.cpio.gz"
    if target.exists() and target.stat().st_mtime > max(BUSYBOX.stat().st_mtime, Path(__file__).stat().st_mtime):
        return target
    BOOT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for directory in ("bin", "sbin", "proc", "sys", "dev", "tmp", "etc", "root"):
            (root / directory).mkdir()
        shutil.copy2(BUSYBOX, root / "bin/busybox")
        applets = subprocess.run([BUSYBOX, "--list"], check=True, capture_output=True, text=True).stdout.split()
        for applet in applets:
            if applet != "busybox":
                (root / "bin" / applet).symlink_to("busybox")
        (root / "init").write_text(INIT.format(banner=banner))
        (root / "init").chmod(0o755)
        files = subprocess.run(["find", "."], cwd=root, check=True, capture_output=True).stdout
        archive = subprocess.run(["cpio", "-o", "-H", "newc", "--quiet"], cwd=root, input=files,
                                 check=True, capture_output=True).stdout
        target.write_bytes(subprocess.run(["gzip", "-9"], input=archive, check=True, capture_output=True).stdout)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--native", action="store_true", help="boot the native clang build instead")
    parser.add_argument("--linux", type=Path, default=Path(os.environ.get("SLATE_LINUX", HERE / "linux")),
                        help="Linux source checkout (default: $SLATE_LINUX or ./linux)")
    parser.add_argument("--memory", default="2G")
    parser.add_argument("--cpus", default="2")
    parser.add_argument("--append", default="", help="extra kernel command line")
    parser.add_argument("qemu_args", nargs="*")
    args = parser.parse_args()

    linux = args.linux.resolve()
    if not BUSYBOX.exists():
        raise SystemExit(f"{BUSYBOX} is missing: put a static x86_64 busybox there")
    build = linux / ("build-clang" if args.native else "build-slate")
    banner = "native clang kernel" if args.native else "slate-translated kernel"
    command = ["qemu-system-x86_64", "-kernel", bzimage(linux, build), "-initrd", initramfs(banner),
               "-m", args.memory, "-smp", args.cpus, "-nographic", "-no-reboot",
               "-append", f"console=ttyS0 rdinit=/init panic=-1 {args.append}".strip()]
    if os.access("/dev/kvm", os.R_OK | os.W_OK):
        command += ["-enable-kvm", "-cpu", "host"]
    else:
        command += ["-cpu", "max"]
    os.execvp(command[0], [str(part) for part in command + args.qemu_args])


if __name__ == "__main__":
    main()
