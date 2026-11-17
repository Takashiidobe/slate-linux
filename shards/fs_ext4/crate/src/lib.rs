#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "balloc.rs"]
mod __slate_unit_balloc;
#[path = "bitmap.rs"]
mod __slate_unit_bitmap;
#[path = "block_validity.rs"]
mod __slate_unit_block_validity;
#[path = "dir.rs"]
mod __slate_unit_dir;
#[path = "ext4_jbd2.rs"]
mod __slate_unit_ext4_jbd2;
#[path = "extents.rs"]
mod __slate_unit_extents;
#[path = "extents_status.rs"]
mod __slate_unit_extents_status;
#[path = "file.rs"]
mod __slate_unit_file;
#[path = "fsmap.rs"]
mod __slate_unit_fsmap;
#[path = "fsync.rs"]
mod __slate_unit_fsync;
#[path = "hash.rs"]
mod __slate_unit_hash;
#[path = "ialloc.rs"]
mod __slate_unit_ialloc;
#[path = "indirect.rs"]
mod __slate_unit_indirect;
#[path = "inline.rs"]
mod __slate_unit_inline;
#[path = "inode.rs"]
mod __slate_unit_inode;
#[path = "ioctl.rs"]
mod __slate_unit_ioctl;
#[path = "mballoc.rs"]
mod __slate_unit_mballoc;
#[path = "migrate.rs"]
mod __slate_unit_migrate;
#[path = "mmp.rs"]
mod __slate_unit_mmp;
#[path = "move_extent.rs"]
mod __slate_unit_move_extent;
#[path = "namei.rs"]
mod __slate_unit_namei;
#[path = "page_io.rs"]
mod __slate_unit_page_io;
#[path = "readpage.rs"]
mod __slate_unit_readpage;
#[path = "resize.rs"]
mod __slate_unit_resize;
#[path = "super.rs"]
mod __slate_unit_super;
#[path = "symlink.rs"]
mod __slate_unit_symlink;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "xattr.rs"]
mod __slate_unit_xattr;
#[path = "xattr_hurd.rs"]
mod __slate_unit_xattr_hurd;
#[path = "xattr_trusted.rs"]
mod __slate_unit_xattr_trusted;
#[path = "xattr_user.rs"]
mod __slate_unit_xattr_user;
#[path = "fast_commit.rs"]
mod __slate_unit_fast_commit;
#[path = "orphan.rs"]
mod __slate_unit_orphan;
#[path = "acl.rs"]
mod __slate_unit_acl;
#[path = "xattr_security.rs"]
mod __slate_unit_xattr_security;

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

