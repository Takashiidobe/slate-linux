#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "clnt.rs"]
mod __slate_unit_clnt;
#[path = "xprt.rs"]
mod __slate_unit_xprt;
#[path = "socklib.rs"]
mod __slate_unit_socklib;
#[path = "xprtsock.rs"]
mod __slate_unit_xprtsock;
#[path = "sched.rs"]
mod __slate_unit_sched;
#[path = "auth.rs"]
mod __slate_unit_auth;
#[path = "auth_null.rs"]
mod __slate_unit_auth_null;
#[path = "auth_tls.rs"]
mod __slate_unit_auth_tls;
#[path = "auth_unix.rs"]
mod __slate_unit_auth_unix;
#[path = "svc.rs"]
mod __slate_unit_svc;
#[path = "svcsock.rs"]
mod __slate_unit_svcsock;
#[path = "svcauth.rs"]
mod __slate_unit_svcauth;
#[path = "svcauth_unix.rs"]
mod __slate_unit_svcauth_unix;
#[path = "addr.rs"]
mod __slate_unit_addr;
#[path = "rpcb_clnt.rs"]
mod __slate_unit_rpcb_clnt;
#[path = "timer.rs"]
mod __slate_unit_timer;
#[path = "xdr.rs"]
mod __slate_unit_xdr;
#[path = "sunrpc_syms.rs"]
mod __slate_unit_sunrpc_syms;
#[path = "cache.rs"]
mod __slate_unit_cache;
#[path = "rpc_pipe.rs"]
mod __slate_unit_rpc_pipe;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "svc_xprt.rs"]
mod __slate_unit_svc_xprt;
#[path = "xprtmultipath.rs"]
mod __slate_unit_xprtmultipath;
#[path = "netlink.rs"]
mod __slate_unit_netlink;
#[path = "backchannel_rqst.rs"]
mod __slate_unit_backchannel_rqst;
#[path = "stats.rs"]
mod __slate_unit_stats;
#[path = "sysctl.rs"]
mod __slate_unit_sysctl;
#[path = "auth_gss.rs"]
mod __slate_unit_auth_gss;
#[path = "gss_mech_switch.rs"]
mod __slate_unit_gss_mech_switch;
#[path = "svcauth_gss.rs"]
mod __slate_unit_svcauth_gss;
#[path = "gss_rpc_upcall.rs"]
mod __slate_unit_gss_rpc_upcall;
#[path = "gss_rpc_xdr.rs"]
mod __slate_unit_gss_rpc_xdr;
#[path = "trace.rs"]
mod __slate_unit_trace;
#[path = "gss_krb5_mech.rs"]
mod __slate_unit_gss_krb5_mech;
#[path = "gss_krb5_seal.rs"]
mod __slate_unit_gss_krb5_seal;
#[path = "gss_krb5_unseal.rs"]
mod __slate_unit_gss_krb5_unseal;
#[path = "gss_krb5_wrap.rs"]
mod __slate_unit_gss_krb5_wrap;
#[path = "gss_krb5_crypto.rs"]
mod __slate_unit_gss_krb5_crypto;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

