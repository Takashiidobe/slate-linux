#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "open.rs"]
mod __slate_unit_open;
#[path = "read_write.rs"]
mod __slate_unit_read_write;
#[path = "file_table.rs"]
mod __slate_unit_file_table;
#[path = "super.rs"]
mod __slate_unit_super;
#[path = "char_dev.rs"]
mod __slate_unit_char_dev;
#[path = "stat.rs"]
mod __slate_unit_stat;
#[path = "exec.rs"]
mod __slate_unit_exec;
#[path = "pipe.rs"]
mod __slate_unit_pipe;
#[path = "namei.rs"]
mod __slate_unit_namei;
#[path = "fcntl.rs"]
mod __slate_unit_fcntl;
#[path = "ioctl.rs"]
mod __slate_unit_ioctl;
#[path = "readdir.rs"]
mod __slate_unit_readdir;
#[path = "select.rs"]
mod __slate_unit_select;
#[path = "dcache.rs"]
mod __slate_unit_dcache;
#[path = "inode.rs"]
mod __slate_unit_inode;
#[path = "attr.rs"]
mod __slate_unit_attr;
#[path = "bad_inode.rs"]
mod __slate_unit_bad_inode;
#[path = "file.rs"]
mod __slate_unit_file;
#[path = "filesystems.rs"]
mod __slate_unit_filesystems;
#[path = "namespace.rs"]
mod __slate_unit_namespace;
#[path = "seq_file.rs"]
mod __slate_unit_seq_file;
#[path = "xattr.rs"]
mod __slate_unit_xattr;
#[path = "libfs.rs"]
mod __slate_unit_libfs;
#[path = "fs_writeback.rs"]
mod __slate_unit_fs_writeback;
#[path = "pnode.rs"]
mod __slate_unit_pnode;
#[path = "splice.rs"]
mod __slate_unit_splice;
#[path = "sync.rs"]
mod __slate_unit_sync;
#[path = "utimes.rs"]
mod __slate_unit_utimes;
#[path = "d_path.rs"]
mod __slate_unit_d_path;
#[path = "stack.rs"]
mod __slate_unit_stack;
#[path = "fs_struct.rs"]
mod __slate_unit_fs_struct;
#[path = "statfs.rs"]
mod __slate_unit_statfs;
#[path = "fs_pin.rs"]
mod __slate_unit_fs_pin;
#[path = "nsfs.rs"]
mod __slate_unit_nsfs;
#[path = "fs_dirent.rs"]
mod __slate_unit_fs_dirent;
#[path = "fs_context.rs"]
mod __slate_unit_fs_context;
#[path = "fs_parser.rs"]
mod __slate_unit_fs_parser;
#[path = "fsopen.rs"]
mod __slate_unit_fsopen;
#[path = "init.rs"]
mod __slate_unit_init;
#[path = "kernel_read_file.rs"]
mod __slate_unit_kernel_read_file;
#[path = "mnt_idmapping.rs"]
mod __slate_unit_mnt_idmapping;
#[path = "remap_range.rs"]
mod __slate_unit_remap_range;
#[path = "pidfs.rs"]
mod __slate_unit_pidfs;
#[path = "file_attr.rs"]
mod __slate_unit_file_attr;
#[path = "fserror.rs"]
mod __slate_unit_fserror;
#[path = "nullfs.rs"]
mod __slate_unit_nullfs;
#[path = "failfs.rs"]
mod __slate_unit_failfs;
#[path = "buffer.rs"]
mod __slate_unit_buffer;
#[path = "mpage.rs"]
mod __slate_unit_mpage;
#[path = "proc_namespace.rs"]
mod __slate_unit_proc_namespace;
#[path = "direct_io.rs"]
mod __slate_unit_direct_io;
#[path = "eventpoll.rs"]
mod __slate_unit_eventpoll;
#[path = "anon_inodes.rs"]
mod __slate_unit_anon_inodes;
#[path = "signalfd.rs"]
mod __slate_unit_signalfd;
#[path = "timerfd.rs"]
mod __slate_unit_timerfd;
#[path = "eventfd.rs"]
mod __slate_unit_eventfd;
#[path = "aio.rs"]
mod __slate_unit_aio;
#[path = "locks.rs"]
mod __slate_unit_locks;
#[path = "binfmt_misc.rs"]
mod __slate_unit_binfmt_misc;
#[path = "binfmt_script.rs"]
mod __slate_unit_binfmt_script;
#[path = "binfmt_elf.rs"]
mod __slate_unit_binfmt_elf;
#[path = "compat_binfmt_elf.rs"]
mod __slate_unit_compat_binfmt_elf;
#[path = "mbcache.rs"]
mod __slate_unit_mbcache;
#[path = "posix_acl.rs"]
mod __slate_unit_posix_acl;
#[path = "coredump.rs"]
mod __slate_unit_coredump;
#[path = "drop_caches.rs"]
mod __slate_unit_drop_caches;
#[path = "sysctls.rs"]
mod __slate_unit_sysctls;
#[path = "fhandle.rs"]
mod __slate_unit_fhandle;

mod __slate_panic {
    unsafe extern "C" {
        fn panic(fmt: *const core::ffi::c_char, ...) -> !;
    }

    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        unsafe { panic(c"rust panic\n".as_ptr()) }
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

