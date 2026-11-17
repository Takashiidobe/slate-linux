# slate-linux

Rust translation of the Linux kernel by
[slate](https://github.com/Takashiidobe/slate), a C to Rust translator.

Linux 7.3-rc5
([`6addb4f38557`](https://github.com/torvalds/linux/commit/6addb4f385570ebc11c4eb499a4f1c149f313e84)),
x86_64 `defconfig`, built with clang:

- All 3005 C translation units translate, in 84 crates (shards).
- The 57 shards that go into `vmlinux` replace their C objects, and `vmlinux`
  links with 0 build errors, 0 link errors and 0 objtool warnings.
- The kernel boots under QEMU to a busybox shell, with the same boot log as the
  native build.
- 10 kselftest collections (31 tests, 205 subtests) give the same results as the
  native kernel.

## Layout

| Path                   | What it is                                                     |
| ---------------------- | -------------------------------------------------------------- |
| `shards/<name>/crate/` | One translated crate: a `staticlib` with one module per C file |
| `kernel.config`        | The `.config` the translation was made against                 |
| `link.py`              | Links `vmlinux` with the shard objects in place of the C ones  |
| `run-qemu.py`          | Boots the result with a busybox initramfs                      |
| `vendor/`              | The `bitfields` crate the shards depend on                     |

Each shard's `link-order.json` maps every C file to its Rust module. `link.py`
replaces each C object in `vmlinux.a` with that module's codegen units, at the
same archive position, so initcall and other section order stays the same.

## Requirements

- clang 23 and GNU ld (`kernel.config` records `CONFIG_CLANG_VERSION=230104`;
  another clang can flip `CONFIG_CC_HAS_*` options and change the struct layouts
  the Rust was translated against)
- [rustup](https://rustup.rs): `rust-toolchain.toml` pins `nightly-2026-10-25`
  with `rust-src` for `-Zbuild-std`
- The usual kernel build dependencies (`make`, `flex`, `bison`, `bc`, `libelf`,
  `openssl`)
- `qemu-system-x86_64`, `cpio` and `gzip` to boot
- About 125 GB of disk for the shards' cargo target directories

## Build and boot

Get the kernel source at the translated commit:

```sh
git clone https://github.com/Takashiidobe/slate-linux
cd slate-linux
git init linux
git -C linux fetch --depth 1 https://github.com/torvalds/linux 6addb4f385570ebc11c4eb499a4f1c149f313e84
git -C linux checkout FETCH_HEAD
```

`link.py` and `run-qemu.py` look for the kernel in `./linux`; pass `--linux` or
set `SLATE_LINUX` to use another checkout.

Put a static x86_64 busybox in `boot/`:

```sh
mkdir -p boot
curl -Lo boot/busybox https://busybox.net/downloads/binaries/1.35.0-x86_64-linux-musl/busybox
chmod +x boot/busybox
```

Build each shard with cargo. Each one is an `x86_64-unknown-none` staticlib
built with the kernel's codegen flags
(`shards/<name>/crate/.cargo/config.toml`):

```sh
for crate in shards/*/crate; do (cd $crate && cargo build --release) || break; done
```

Only the 57 shards that go into `vmlinux` are needed to link. The `mod_*` and
`unlinked_*` shards (modules, the boot stub and the vDSO) and the four
build-time offset generators can be skipped.

Link `vmlinux`:

```sh
./link.py
```

The first run builds the native kernel in `linux/build-clang` from
`kernel.config`. `link.py` then copies it to `linux/build-slate`, swaps the
shard objects into its `vmlinux.a`, and runs `make vmlinux`.
`./link.py block mm` swaps in only those shards and leaves the rest as C. Build
every shard with the same toolchain: `link.py` refuses to link shards whose
copies of `core` or `compiler_builtins` differ.

Boot:

```sh
./run-qemu.py            # linux/build-slate, the translated kernel
./run-qemu.py --native   # linux/build-clang, for comparison
```

The shell prints `slate-translated kernel: Linux ...`. Quit with `Ctrl-A X` or
`poweroff`. Arguments after `--` go to QEMU (`./run-qemu.py -- -s -S` waits for
gdb on port 1234).
