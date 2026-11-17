#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "fsnotify.rs"]
mod __slate_unit_fsnotify;
#[path = "notification.rs"]
mod __slate_unit_notification;
#[path = "group.rs"]
mod __slate_unit_group;
#[path = "mark.rs"]
mod __slate_unit_mark;
#[path = "fdinfo.rs"]
mod __slate_unit_fdinfo;
#[path = "dnotify.rs"]
mod __slate_unit_dnotify;
#[path = "inotify_fsnotify.rs"]
mod __slate_unit_inotify_fsnotify;
#[path = "inotify_user.rs"]
mod __slate_unit_inotify_user;
#[path = "nfsacl.rs"]
mod __slate_unit_nfsacl;
#[path = "grace.rs"]
mod __slate_unit_grace;
#[path = "common.rs"]
mod __slate_unit_common;
#[path = "iomap_trace.rs"]
mod __slate_unit_iomap_trace;
#[path = "iter.rs"]
mod __slate_unit_iter;
#[path = "buffered_io.rs"]
mod __slate_unit_buffered_io;
#[path = "direct_io.rs"]
mod __slate_unit_direct_io;
#[path = "ioend.rs"]
mod __slate_unit_ioend;
#[path = "fiemap.rs"]
mod __slate_unit_fiemap;
#[path = "seek.rs"]
mod __slate_unit_seek;
#[path = "bio.rs"]
mod __slate_unit_bio;
#[path = "swapfile.rs"]
mod __slate_unit_swapfile;
#[path = "dquot.rs"]
mod __slate_unit_dquot;
#[path = "quota_v2.rs"]
mod __slate_unit_quota_v2;
#[path = "quota_tree.rs"]
mod __slate_unit_quota_tree;
#[path = "quota.rs"]
mod __slate_unit_quota;
#[path = "kqid.rs"]
mod __slate_unit_kqid;
#[path = "quota_netlink.rs"]
mod __slate_unit_quota_netlink;
#[path = "mount.rs"]
mod __slate_unit_mount;
#[path = "kernfs_inode.rs"]
mod __slate_unit_kernfs_inode;
#[path = "kernfs_dir.rs"]
mod __slate_unit_kernfs_dir;
#[path = "kernfs_file.rs"]
mod __slate_unit_kernfs_file;
#[path = "kernfs_symlink.rs"]
mod __slate_unit_kernfs_symlink;
#[path = "devpts_inode.rs"]
mod __slate_unit_devpts_inode;
#[path = "buffered_read.rs"]
mod __slate_unit_buffered_read;
#[path = "buffered_write.rs"]
mod __slate_unit_buffered_write;
#[path = "direct_read.rs"]
mod __slate_unit_direct_read;
#[path = "direct_write.rs"]
mod __slate_unit_direct_write;
#[path = "iterator.rs"]
mod __slate_unit_iterator;
#[path = "locking.rs"]
mod __slate_unit_locking;
#[path = "__slate_unit_main.rs"]
mod __slate_unit_main;
#[path = "netfs_misc.rs"]
mod __slate_unit_netfs_misc;
#[path = "objects.rs"]
mod __slate_unit_objects;
#[path = "read_collect.rs"]
mod __slate_unit_read_collect;
#[path = "read_pgpriv2.rs"]
mod __slate_unit_read_pgpriv2;
#[path = "read_retry.rs"]
mod __slate_unit_read_retry;
#[path = "read_single.rs"]
mod __slate_unit_read_single;
#[path = "rolling_buffer.rs"]
mod __slate_unit_rolling_buffer;
#[path = "write_collect.rs"]
mod __slate_unit_write_collect;
#[path = "write_issue.rs"]
mod __slate_unit_write_issue;
#[path = "write_retry.rs"]
mod __slate_unit_write_retry;
#[path = "transaction.rs"]
mod __slate_unit_transaction;
#[path = "commit.rs"]
mod __slate_unit_commit;
#[path = "recovery.rs"]
mod __slate_unit_recovery;
#[path = "checkpoint.rs"]
mod __slate_unit_checkpoint;
#[path = "revoke.rs"]
mod __slate_unit_revoke;
#[path = "journal.rs"]
mod __slate_unit_journal;
#[path = "ramfs_inode.rs"]
mod __slate_unit_ramfs_inode;
#[path = "file_mmu.rs"]
mod __slate_unit_file_mmu;
#[path = "hugetlbfs_inode.rs"]
mod __slate_unit_hugetlbfs_inode;
#[path = "cache.rs"]
mod __slate_unit_cache;
#[path = "fat_dir.rs"]
mod __slate_unit_fat_dir;
#[path = "fatent.rs"]
mod __slate_unit_fatent;
#[path = "fat_file.rs"]
mod __slate_unit_fat_file;
#[path = "fat_inode.rs"]
mod __slate_unit_fat_inode;
#[path = "fat_misc.rs"]
mod __slate_unit_fat_misc;
#[path = "nfs.rs"]
mod __slate_unit_nfs;
#[path = "namei_vfat.rs"]
mod __slate_unit_namei_vfat;
#[path = "namei_msdos.rs"]
mod __slate_unit_namei_msdos;
#[path = "namei.rs"]
mod __slate_unit_namei;
#[path = "isofs_inode.rs"]
mod __slate_unit_isofs_inode;
#[path = "isofs_dir.rs"]
mod __slate_unit_isofs_dir;
#[path = "util.rs"]
mod __slate_unit_util;
#[path = "rock.rs"]
mod __slate_unit_rock;
#[path = "export.rs"]
mod __slate_unit_export;
#[path = "joliet.rs"]
mod __slate_unit_joliet;
#[path = "compress.rs"]
mod __slate_unit_compress;
#[path = "expfs.rs"]
mod __slate_unit_expfs;
#[path = "clntlock.rs"]
mod __slate_unit_clntlock;
#[path = "clntproc.rs"]
mod __slate_unit_clntproc;
#[path = "clntxdr.rs"]
mod __slate_unit_clntxdr;
#[path = "host.rs"]
mod __slate_unit_host;
#[path = "svc.rs"]
mod __slate_unit_svc;
#[path = "svclock.rs"]
mod __slate_unit_svclock;
#[path = "svcshare.rs"]
mod __slate_unit_svcshare;
#[path = "svcproc.rs"]
mod __slate_unit_svcproc;
#[path = "svcsubs.rs"]
mod __slate_unit_svcsubs;
#[path = "mon.rs"]
mod __slate_unit_mon;
#[path = "lockd_trace.rs"]
mod __slate_unit_lockd_trace;
#[path = "lockd_netlink.rs"]
mod __slate_unit_lockd_netlink;
#[path = "nlm3xdr_gen.rs"]
mod __slate_unit_nlm3xdr_gen;
#[path = "clnt4xdr.rs"]
mod __slate_unit_clnt4xdr;
#[path = "svc4proc.rs"]
mod __slate_unit_svc4proc;
#[path = "nlm4xdr_gen.rs"]
mod __slate_unit_nlm4xdr_gen;
#[path = "procfs.rs"]
mod __slate_unit_procfs;
#[path = "nls_base.rs"]
mod __slate_unit_nls_base;
#[path = "nls_cp437.rs"]
mod __slate_unit_nls_cp437;
#[path = "nls_ascii.rs"]
mod __slate_unit_nls_ascii;
#[path = "nls_iso8859_1.rs"]
mod __slate_unit_nls_iso8859_1;
#[path = "nls_utf8.rs"]
mod __slate_unit_nls_utf8;
#[path = "init.rs"]
mod __slate_unit_init;
#[path = "autofs_inode.rs"]
mod __slate_unit_autofs_inode;
#[path = "root.rs"]
mod __slate_unit_root;
#[path = "autofs_symlink.rs"]
mod __slate_unit_autofs_symlink;
#[path = "waitq.rs"]
mod __slate_unit_waitq;
#[path = "expire.rs"]
mod __slate_unit_expire;
#[path = "dev_ioctl.rs"]
mod __slate_unit_dev_ioctl;
#[path = "vfs_super.rs"]
mod __slate_unit_vfs_super;
#[path = "vfs_inode.rs"]
mod __slate_unit_vfs_inode;
#[path = "vfs_inode_dotl.rs"]
mod __slate_unit_vfs_inode_dotl;
#[path = "vfs_addr.rs"]
mod __slate_unit_vfs_addr;
#[path = "vfs_file.rs"]
mod __slate_unit_vfs_file;
#[path = "vfs_dir.rs"]
mod __slate_unit_vfs_dir;
#[path = "vfs_dentry.rs"]
mod __slate_unit_vfs_dentry;
#[path = "v9fs.rs"]
mod __slate_unit_v9fs;
#[path = "fid.rs"]
mod __slate_unit_fid;
#[path = "xattr.rs"]
mod __slate_unit_xattr;
#[path = "debugfs_inode.rs"]
mod __slate_unit_debugfs_inode;
#[path = "debugfs_file.rs"]
mod __slate_unit_debugfs_file;
#[path = "tracefs_inode.rs"]
mod __slate_unit_tracefs_inode;
#[path = "event_inode.rs"]
mod __slate_unit_event_inode;

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

