#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "core.rs"]
mod __slate_unit_core;
#[path = "nf_log.rs"]
mod __slate_unit_nf_log;
#[path = "nf_queue.rs"]
mod __slate_unit_nf_queue;
#[path = "nf_sockopt.rs"]
mod __slate_unit_nf_sockopt;
#[path = "utils.rs"]
mod __slate_unit_utils;
#[path = "nfnetlink.rs"]
mod __slate_unit_nfnetlink;
#[path = "nfnetlink_log.rs"]
mod __slate_unit_nfnetlink_log;
#[path = "nf_conntrack_core.rs"]
mod __slate_unit_nf_conntrack_core;
#[path = "nf_conntrack_standalone.rs"]
mod __slate_unit_nf_conntrack_standalone;
#[path = "nf_conntrack_expect.rs"]
mod __slate_unit_nf_conntrack_expect;
#[path = "nf_conntrack_helper.rs"]
mod __slate_unit_nf_conntrack_helper;
#[path = "nf_conntrack_proto.rs"]
mod __slate_unit_nf_conntrack_proto;
#[path = "nf_conntrack_proto_generic.rs"]
mod __slate_unit_nf_conntrack_proto_generic;
#[path = "nf_conntrack_proto_tcp.rs"]
mod __slate_unit_nf_conntrack_proto_tcp;
#[path = "nf_conntrack_proto_udp.rs"]
mod __slate_unit_nf_conntrack_proto_udp;
#[path = "nf_conntrack_proto_icmp.rs"]
mod __slate_unit_nf_conntrack_proto_icmp;
#[path = "nf_conntrack_extend.rs"]
mod __slate_unit_nf_conntrack_extend;
#[path = "nf_conntrack_acct.rs"]
mod __slate_unit_nf_conntrack_acct;
#[path = "nf_conntrack_seqadj.rs"]
mod __slate_unit_nf_conntrack_seqadj;
#[path = "nf_conntrack_proto_icmpv6.rs"]
mod __slate_unit_nf_conntrack_proto_icmpv6;
#[path = "nf_conntrack_netlink.rs"]
mod __slate_unit_nf_conntrack_netlink;
#[path = "nf_conntrack_ftp.rs"]
mod __slate_unit_nf_conntrack_ftp;
#[path = "nf_conntrack_irc.rs"]
mod __slate_unit_nf_conntrack_irc;
#[path = "nf_conntrack_sip.rs"]
mod __slate_unit_nf_conntrack_sip;
#[path = "nf_nat_core.rs"]
mod __slate_unit_nf_nat_core;
#[path = "nf_nat_proto.rs"]
mod __slate_unit_nf_nat_proto;
#[path = "nf_nat_helper.rs"]
mod __slate_unit_nf_nat_helper;
#[path = "nf_nat_masquerade.rs"]
mod __slate_unit_nf_nat_masquerade;
#[path = "nf_nat_ftp.rs"]
mod __slate_unit_nf_nat_ftp;
#[path = "nf_nat_irc.rs"]
mod __slate_unit_nf_nat_irc;
#[path = "nf_nat_sip.rs"]
mod __slate_unit_nf_nat_sip;
#[path = "x_tables.rs"]
mod __slate_unit_x_tables;
#[path = "xt_tcpudp.rs"]
mod __slate_unit_xt_tcpudp;
#[path = "xt_CONNSECMARK.rs"]
mod __slate_unit_xt_CONNSECMARK;
#[path = "xt_NFLOG.rs"]
mod __slate_unit_xt_NFLOG;
#[path = "xt_SECMARK.rs"]
mod __slate_unit_xt_SECMARK;
#[path = "xt_TCPMSS.rs"]
mod __slate_unit_xt_TCPMSS;
#[path = "xt_conntrack.rs"]
mod __slate_unit_xt_conntrack;
#[path = "xt_policy.rs"]
mod __slate_unit_xt_policy;
#[path = "xt_state.rs"]
mod __slate_unit_xt_state;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

