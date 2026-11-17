#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "client.rs"]
mod __slate_unit_client;
#[path = "dir.rs"]
mod __slate_unit_dir;
#[path = "file.rs"]
mod __slate_unit_file;
#[path = "getroot.rs"]
mod __slate_unit_getroot;
#[path = "inode.rs"]
mod __slate_unit_inode;
#[path = "super.rs"]
mod __slate_unit_super;
#[path = "io.rs"]
mod __slate_unit_io;
#[path = "direct.rs"]
mod __slate_unit_direct;
#[path = "pagelist.rs"]
mod __slate_unit_pagelist;
#[path = "read.rs"]
mod __slate_unit_read;
#[path = "symlink.rs"]
mod __slate_unit_symlink;
#[path = "unlink.rs"]
mod __slate_unit_unlink;
#[path = "write.rs"]
mod __slate_unit_write;
#[path = "namespace.rs"]
mod __slate_unit_namespace;
#[path = "mount_clnt.rs"]
mod __slate_unit_mount_clnt;
#[path = "nfstrace.rs"]
mod __slate_unit_nfstrace;
#[path = "export.rs"]
mod __slate_unit_export;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "fs_context.rs"]
mod __slate_unit_fs_context;
#[path = "nfsroot.rs"]
mod __slate_unit_nfsroot;
#[path = "sysctl.rs"]
mod __slate_unit_sysctl;
#[path = "nfs3super.rs"]
mod __slate_unit_nfs3super;
#[path = "nfs3client.rs"]
mod __slate_unit_nfs3client;
#[path = "nfs3proc.rs"]
mod __slate_unit_nfs3proc;
#[path = "nfs3xdr.rs"]
mod __slate_unit_nfs3xdr;
#[path = "nfs3acl.rs"]
mod __slate_unit_nfs3acl;
#[path = "nfs4proc.rs"]
mod __slate_unit_nfs4proc;
#[path = "nfs4xdr.rs"]
mod __slate_unit_nfs4xdr;
#[path = "nfs4state.rs"]
mod __slate_unit_nfs4state;
#[path = "nfs4renewd.rs"]
mod __slate_unit_nfs4renewd;
#[path = "nfs4super.rs"]
mod __slate_unit_nfs4super;
#[path = "nfs4file.rs"]
mod __slate_unit_nfs4file;
#[path = "delegation.rs"]
mod __slate_unit_delegation;
#[path = "nfs4idmap.rs"]
mod __slate_unit_nfs4idmap;
#[path = "callback.rs"]
mod __slate_unit_callback;
#[path = "callback_xdr.rs"]
mod __slate_unit_callback_xdr;
#[path = "callback_proc.rs"]
mod __slate_unit_callback_proc;
#[path = "nfs4namespace.rs"]
mod __slate_unit_nfs4namespace;
#[path = "nfs4getroot.rs"]
mod __slate_unit_nfs4getroot;
#[path = "nfs4client.rs"]
mod __slate_unit_nfs4client;
#[path = "nfs4session.rs"]
mod __slate_unit_nfs4session;
#[path = "dns_resolve.rs"]
mod __slate_unit_dns_resolve;
#[path = "nfs4trace.rs"]
mod __slate_unit_nfs4trace;
#[path = "pnfs.rs"]
mod __slate_unit_pnfs;
#[path = "pnfs_dev.rs"]
mod __slate_unit_pnfs_dev;
#[path = "pnfs_nfs.rs"]
mod __slate_unit_pnfs_nfs;
#[path = "nfs4sysctl.rs"]
mod __slate_unit_nfs4sysctl;
#[path = "nfs40client.rs"]
mod __slate_unit_nfs40client;
#[path = "nfs40proc.rs"]
mod __slate_unit_nfs40proc;
#[path = "filelayout.rs"]
mod __slate_unit_filelayout;
#[path = "filelayoutdev.rs"]
mod __slate_unit_filelayoutdev;
#[path = "blocklayout.rs"]
mod __slate_unit_blocklayout;
#[path = "dev.rs"]
mod __slate_unit_dev;
#[path = "extent_tree.rs"]
mod __slate_unit_extent_tree;
#[path = "rpc_pipefs.rs"]
mod __slate_unit_rpc_pipefs;
#[path = "flexfilelayout.rs"]
mod __slate_unit_flexfilelayout;
#[path = "flexfilelayoutdev.rs"]
mod __slate_unit_flexfilelayoutdev;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

