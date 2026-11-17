#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "io_uring.rs"]
mod __slate_unit_io_uring;
#[path = "opdef.rs"]
mod __slate_unit_opdef;
#[path = "kbuf.rs"]
mod __slate_unit_kbuf;
#[path = "rsrc.rs"]
mod __slate_unit_rsrc;
#[path = "notif.rs"]
mod __slate_unit_notif;
#[path = "tctx.rs"]
mod __slate_unit_tctx;
#[path = "filetable.rs"]
mod __slate_unit_filetable;
#[path = "rw.rs"]
mod __slate_unit_rw;
#[path = "poll.rs"]
mod __slate_unit_poll;
#[path = "tw.rs"]
mod __slate_unit_tw;
#[path = "wait.rs"]
mod __slate_unit_wait;
#[path = "eventfd.rs"]
mod __slate_unit_eventfd;
#[path = "uring_cmd.rs"]
mod __slate_unit_uring_cmd;
#[path = "openclose.rs"]
mod __slate_unit_openclose;
#[path = "sqpoll.rs"]
mod __slate_unit_sqpoll;
#[path = "xattr.rs"]
mod __slate_unit_xattr;
#[path = "nop.rs"]
mod __slate_unit_nop;
#[path = "fs.rs"]
mod __slate_unit_fs;
#[path = "splice.rs"]
mod __slate_unit_splice;
#[path = "sync.rs"]
mod __slate_unit_sync;
#[path = "msg_ring.rs"]
mod __slate_unit_msg_ring;
#[path = "advise.rs"]
mod __slate_unit_advise;
#[path = "statx.rs"]
mod __slate_unit_statx;
#[path = "timeout.rs"]
mod __slate_unit_timeout;
#[path = "cancel.rs"]
mod __slate_unit_cancel;
#[path = "waitid.rs"]
mod __slate_unit_waitid;
#[path = "register.rs"]
mod __slate_unit_register;
#[path = "truncate.rs"]
mod __slate_unit_truncate;
#[path = "memmap.rs"]
mod __slate_unit_memmap;
#[path = "alloc_cache.rs"]
mod __slate_unit_alloc_cache;
#[path = "query.rs"]
mod __slate_unit_query;
#[path = "loop.rs"]
mod __slate_unit_loop;
#[path = "zcrx.rs"]
mod __slate_unit_zcrx;
#[path = "io_wq.rs"]
mod __slate_unit_io_wq;
#[path = "futex.rs"]
mod __slate_unit_futex;
#[path = "epoll.rs"]
mod __slate_unit_epoll;
#[path = "napi.rs"]
mod __slate_unit_napi;
#[path = "net.rs"]
mod __slate_unit_net;
#[path = "cmd_net.rs"]
mod __slate_unit_cmd_net;
#[path = "fdinfo.rs"]
mod __slate_unit_fdinfo;
#[path = "bpf_filter.rs"]
mod __slate_unit_bpf_filter;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

