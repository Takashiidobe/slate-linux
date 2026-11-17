#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "route.rs"]
mod __slate_unit_route;
#[path = "inetpeer.rs"]
mod __slate_unit_inetpeer;
#[path = "protocol.rs"]
mod __slate_unit_protocol;
#[path = "ip_input.rs"]
mod __slate_unit_ip_input;
#[path = "ip_fragment.rs"]
mod __slate_unit_ip_fragment;
#[path = "ip_forward.rs"]
mod __slate_unit_ip_forward;
#[path = "ip_options.rs"]
mod __slate_unit_ip_options;
#[path = "ip_output.rs"]
mod __slate_unit_ip_output;
#[path = "ip_sockglue.rs"]
mod __slate_unit_ip_sockglue;
#[path = "inet_hashtables.rs"]
mod __slate_unit_inet_hashtables;
#[path = "inet_timewait_sock.rs"]
mod __slate_unit_inet_timewait_sock;
#[path = "inet_connection_sock.rs"]
mod __slate_unit_inet_connection_sock;
#[path = "tcp.rs"]
mod __slate_unit_tcp;
#[path = "tcp_input.rs"]
mod __slate_unit_tcp_input;
#[path = "tcp_output.rs"]
mod __slate_unit_tcp_output;
#[path = "tcp_timer.rs"]
mod __slate_unit_tcp_timer;
#[path = "tcp_ipv4.rs"]
mod __slate_unit_tcp_ipv4;
#[path = "tcp_minisocks.rs"]
mod __slate_unit_tcp_minisocks;
#[path = "tcp_cong.rs"]
mod __slate_unit_tcp_cong;
#[path = "tcp_metrics.rs"]
mod __slate_unit_tcp_metrics;
#[path = "tcp_fastopen.rs"]
mod __slate_unit_tcp_fastopen;
#[path = "tcp_recovery.rs"]
mod __slate_unit_tcp_recovery;
#[path = "tcp_ulp.rs"]
mod __slate_unit_tcp_ulp;
#[path = "tcp_offload.rs"]
mod __slate_unit_tcp_offload;
#[path = "tcp_plb.rs"]
mod __slate_unit_tcp_plb;
#[path = "datagram.rs"]
mod __slate_unit_datagram;
#[path = "raw.rs"]
mod __slate_unit_raw;
#[path = "udp.rs"]
mod __slate_unit_udp;
#[path = "udp_offload.rs"]
mod __slate_unit_udp_offload;
#[path = "arp.rs"]
mod __slate_unit_arp;
#[path = "icmp.rs"]
mod __slate_unit_icmp;
#[path = "devinet.rs"]
mod __slate_unit_devinet;
#[path = "af_inet.rs"]
mod __slate_unit_af_inet;
#[path = "igmp.rs"]
mod __slate_unit_igmp;
#[path = "fib_frontend.rs"]
mod __slate_unit_fib_frontend;
#[path = "fib_semantics.rs"]
mod __slate_unit_fib_semantics;
#[path = "fib_trie.rs"]
mod __slate_unit_fib_trie;
#[path = "fib_notifier.rs"]
mod __slate_unit_fib_notifier;
#[path = "inet_fragment.rs"]
mod __slate_unit_inet_fragment;
#[path = "ping.rs"]
mod __slate_unit_ping;
#[path = "ip_tunnel_core.rs"]
mod __slate_unit_ip_tunnel_core;
#[path = "gre_offload.rs"]
mod __slate_unit_gre_offload;
#[path = "metrics.rs"]
mod __slate_unit_metrics;
#[path = "netlink.rs"]
mod __slate_unit_netlink;
#[path = "nexthop.rs"]
mod __slate_unit_nexthop;
#[path = "udp_tunnel_stub.rs"]
mod __slate_unit_udp_tunnel_stub;
#[path = "sysctl_net_ipv4.rs"]
mod __slate_unit_sysctl_net_ipv4;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "fib_rules.rs"]
mod __slate_unit_fib_rules;
#[path = "ipmr.rs"]
mod __slate_unit_ipmr;
#[path = "ipmr_base.rs"]
mod __slate_unit_ipmr_base;
#[path = "syncookies.rs"]
mod __slate_unit_syncookies;
#[path = "ipconfig.rs"]
mod __slate_unit_ipconfig;
#[path = "netfilter.rs"]
mod __slate_unit_netfilter;
#[path = "nf_defrag_ipv4.rs"]
mod __slate_unit_nf_defrag_ipv4;
#[path = "tcp_cubic.rs"]
mod __slate_unit_tcp_cubic;
#[path = "cipso_ipv4.rs"]
mod __slate_unit_cipso_ipv4;
#[path = "xfrm4_policy.rs"]
mod __slate_unit_xfrm4_policy;
#[path = "xfrm4_state.rs"]
mod __slate_unit_xfrm4_state;
#[path = "xfrm4_input.rs"]
mod __slate_unit_xfrm4_input;
#[path = "xfrm4_output.rs"]
mod __slate_unit_xfrm4_output;
#[path = "xfrm4_protocol.rs"]
mod __slate_unit_xfrm4_protocol;

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

