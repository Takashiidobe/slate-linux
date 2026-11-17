#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "gc.rs"]
mod __slate_unit_gc;
#[path = "key.rs"]
mod __slate_unit_key;
#[path = "keyring.rs"]
mod __slate_unit_keyring;
#[path = "keyctl.rs"]
mod __slate_unit_keyctl;
#[path = "permission.rs"]
mod __slate_unit_permission;
#[path = "process_keys.rs"]
mod __slate_unit_process_keys;
#[path = "request_key.rs"]
mod __slate_unit_request_key;
#[path = "request_key_auth.rs"]
mod __slate_unit_request_key_auth;
#[path = "user_defined.rs"]
mod __slate_unit_user_defined;
#[path = "compat.rs"]
mod __slate_unit_compat;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "sysctl.rs"]
mod __slate_unit_sysctl;
#[path = "keyctl_pkey.rs"]
mod __slate_unit_keyctl_pkey;
#[path = "commoncap.rs"]
mod __slate_unit_commoncap;
#[path = "lsm_syscalls.rs"]
mod __slate_unit_lsm_syscalls;
#[path = "min_addr.rs"]
mod __slate_unit_min_addr;
#[path = "security.rs"]
mod __slate_unit_security;
#[path = "lsm_notifier.rs"]
mod __slate_unit_lsm_notifier;
#[path = "lsm_init.rs"]
mod __slate_unit_lsm_init;
#[path = "avc.rs"]
mod __slate_unit_avc;
#[path = "hooks.rs"]
mod __slate_unit_hooks;
#[path = "selinuxfs.rs"]
mod __slate_unit_selinuxfs;
#[path = "netlink.rs"]
mod __slate_unit_netlink;
#[path = "nlmsgtab.rs"]
mod __slate_unit_nlmsgtab;
#[path = "netif.rs"]
mod __slate_unit_netif;
#[path = "netnode.rs"]
mod __slate_unit_netnode;
#[path = "netport.rs"]
mod __slate_unit_netport;
#[path = "status.rs"]
mod __slate_unit_status;
#[path = "initcalls.rs"]
mod __slate_unit_initcalls;
#[path = "ebitmap.rs"]
mod __slate_unit_ebitmap;
#[path = "hashtab.rs"]
mod __slate_unit_hashtab;
#[path = "symtab.rs"]
mod __slate_unit_symtab;
#[path = "sidtab.rs"]
mod __slate_unit_sidtab;
#[path = "avtab.rs"]
mod __slate_unit_avtab;
#[path = "policydb.rs"]
mod __slate_unit_policydb;
#[path = "services.rs"]
mod __slate_unit_services;
#[path = "conditional.rs"]
mod __slate_unit_conditional;
#[path = "mls.rs"]
mod __slate_unit_mls;
#[path = "context.rs"]
mod __slate_unit_context;
#[path = "netlabel.rs"]
mod __slate_unit_netlabel;
#[path = "lsm_audit.rs"]
mod __slate_unit_lsm_audit;
#[path = "device_cgroup.rs"]
mod __slate_unit_device_cgroup;
#[path = "iint.rs"]
mod __slate_unit_iint;
#[path = "integrity_audit.rs"]
mod __slate_unit_integrity_audit;
#[path = "efi_secureboot.rs"]
mod __slate_unit_efi_secureboot;

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

