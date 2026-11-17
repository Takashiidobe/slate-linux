#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions, deref_nullptr)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "devres.rs"]
mod __slate_unit_devres;
#[path = "socket.rs"]
mod __slate_unit_socket;
#[path = "compat.rs"]
mod __slate_unit_compat;
#[path = "eth.rs"]
mod __slate_unit_eth;
#[path = "sch_generic.rs"]
mod __slate_unit_sch_generic;
#[path = "sch_mq.rs"]
mod __slate_unit_sch_mq;
#[path = "sch_frag.rs"]
mod __slate_unit_sch_frag;
#[path = "sch_api.rs"]
mod __slate_unit_sch_api;
#[path = "sch_blackhole.rs"]
mod __slate_unit_sch_blackhole;
#[path = "cls_api.rs"]
mod __slate_unit_cls_api;
#[path = "act_api.rs"]
mod __slate_unit_act_api;
#[path = "sch_fifo.rs"]
mod __slate_unit_sch_fifo;
#[path = "cls_cgroup.rs"]
mod __slate_unit_cls_cgroup;
#[path = "ematch.rs"]
mod __slate_unit_ematch;
#[path = "af_netlink.rs"]
mod __slate_unit_af_netlink;
#[path = "genetlink.rs"]
mod __slate_unit_genetlink;
#[path = "policy.rs"]
mod __slate_unit_policy;
#[path = "xfrm_policy.rs"]
mod __slate_unit_xfrm_policy;
#[path = "xfrm_state.rs"]
mod __slate_unit_xfrm_state;
#[path = "xfrm_hash.rs"]
mod __slate_unit_xfrm_hash;
#[path = "xfrm_input.rs"]
mod __slate_unit_xfrm_input;
#[path = "xfrm_output.rs"]
mod __slate_unit_xfrm_output;
#[path = "xfrm_sysctl.rs"]
mod __slate_unit_xfrm_sysctl;
#[path = "xfrm_replay.rs"]
mod __slate_unit_xfrm_replay;
#[path = "xfrm_device.rs"]
mod __slate_unit_xfrm_device;
#[path = "xfrm_nat_keepalive.rs"]
mod __slate_unit_xfrm_nat_keepalive;
#[path = "xfrm_algo.rs"]
mod __slate_unit_xfrm_algo;
#[path = "xfrm_user.rs"]
mod __slate_unit_xfrm_user;
#[path = "af_unix.rs"]
mod __slate_unit_af_unix;
#[path = "garbage.rs"]
mod __slate_unit_garbage;
#[path = "sysctl_net_unix.rs"]
mod __slate_unit_sysctl_net_unix;
#[path = "af_packet.rs"]
mod __slate_unit_af_packet;
#[path = "wireless_core.rs"]
mod __slate_unit_wireless_core;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "radiotap.rs"]
mod __slate_unit_radiotap;
#[path = "util.rs"]
mod __slate_unit_util;
#[path = "reg.rs"]
mod __slate_unit_reg;
#[path = "scan.rs"]
mod __slate_unit_scan;
#[path = "nl80211.rs"]
mod __slate_unit_nl80211;
#[path = "mlme.rs"]
mod __slate_unit_mlme;
#[path = "ibss.rs"]
mod __slate_unit_ibss;
#[path = "sme.rs"]
mod __slate_unit_sme;
#[path = "chan.rs"]
mod __slate_unit_chan;
#[path = "ethtool.rs"]
mod __slate_unit_ethtool;
#[path = "mesh.rs"]
mod __slate_unit_mesh;
#[path = "ap.rs"]
mod __slate_unit_ap;
#[path = "wireless_trace.rs"]
mod __slate_unit_wireless_trace;
#[path = "ocb.rs"]
mod __slate_unit_ocb;
#[path = "michael_mic.rs"]
mod __slate_unit_michael_mic;
#[path = "pmsr.rs"]
mod __slate_unit_pmsr;
#[path = "netlabel_user.rs"]
mod __slate_unit_netlabel_user;
#[path = "netlabel_kapi.rs"]
mod __slate_unit_netlabel_kapi;
#[path = "netlabel_domainhash.rs"]
mod __slate_unit_netlabel_domainhash;
#[path = "netlabel_addrlist.rs"]
mod __slate_unit_netlabel_addrlist;
#[path = "netlabel_mgmt.rs"]
mod __slate_unit_netlabel_mgmt;
#[path = "netlabel_unlabeled.rs"]
mod __slate_unit_netlabel_unlabeled;
#[path = "netlabel_cipso_v4.rs"]
mod __slate_unit_netlabel_cipso_v4;
#[path = "netlabel_calipso.rs"]
mod __slate_unit_netlabel_calipso;
#[path = "rfkill_core.rs"]
mod __slate_unit_rfkill_core;
#[path = "input.rs"]
mod __slate_unit_input;
#[path = "mod.rs"]
mod __slate_unit_mod;
#[path = "client.rs"]
mod __slate_unit_client;
#[path = "error.rs"]
mod __slate_unit_error;
#[path = "protocol.rs"]
mod __slate_unit_protocol;
#[path = "trans_common.rs"]
mod __slate_unit_trans_common;
#[path = "trans_fd.rs"]
mod __slate_unit_trans_fd;
#[path = "trans_virtio.rs"]
mod __slate_unit_trans_virtio;
#[path = "sysctl_net.rs"]
mod __slate_unit_sysctl_net;
#[path = "dns_key.rs"]
mod __slate_unit_dns_key;
#[path = "dns_query.rs"]
mod __slate_unit_dns_query;
#[path = "alert.rs"]
mod __slate_unit_alert;
#[path = "genl.rs"]
mod __slate_unit_genl;
#[path = "netlink.rs"]
mod __slate_unit_netlink;
#[path = "request.rs"]
mod __slate_unit_request;
#[path = "tlshd.rs"]
mod __slate_unit_tlshd;
#[path = "handshake_trace.rs"]
mod __slate_unit_handshake_trace;

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

