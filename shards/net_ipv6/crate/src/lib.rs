#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "af_inet6.rs"]
mod __slate_unit_af_inet6;
#[path = "anycast.rs"]
mod __slate_unit_anycast;
#[path = "ip6_output.rs"]
mod __slate_unit_ip6_output;
#[path = "ip6_input.rs"]
mod __slate_unit_ip6_input;
#[path = "addrconf.rs"]
mod __slate_unit_addrconf;
#[path = "addrlabel.rs"]
mod __slate_unit_addrlabel;
#[path = "route.rs"]
mod __slate_unit_route;
#[path = "ip6_fib.rs"]
mod __slate_unit_ip6_fib;
#[path = "ipv6_sockglue.rs"]
mod __slate_unit_ipv6_sockglue;
#[path = "ndisc.rs"]
mod __slate_unit_ndisc;
#[path = "udp.rs"]
mod __slate_unit_udp;
#[path = "raw.rs"]
mod __slate_unit_raw;
#[path = "icmp.rs"]
mod __slate_unit_icmp;
#[path = "mcast.rs"]
mod __slate_unit_mcast;
#[path = "reassembly.rs"]
mod __slate_unit_reassembly;
#[path = "tcp_ipv6.rs"]
mod __slate_unit_tcp_ipv6;
#[path = "ping.rs"]
mod __slate_unit_ping;
#[path = "exthdrs.rs"]
mod __slate_unit_exthdrs;
#[path = "datagram.rs"]
mod __slate_unit_datagram;
#[path = "ip6_flowlabel.rs"]
mod __slate_unit_ip6_flowlabel;
#[path = "inet6_connection_sock.rs"]
mod __slate_unit_inet6_connection_sock;
#[path = "udp_offload.rs"]
mod __slate_unit_udp_offload;
#[path = "seg6.rs"]
mod __slate_unit_seg6;
#[path = "fib6_notifier.rs"]
mod __slate_unit_fib6_notifier;
#[path = "rpl.rs"]
mod __slate_unit_rpl;
#[path = "ioam6.rs"]
mod __slate_unit_ioam6;
#[path = "sysctl_net_ipv6.rs"]
mod __slate_unit_sysctl_net_ipv6;
#[path = "xfrm6_policy.rs"]
mod __slate_unit_xfrm6_policy;
#[path = "xfrm6_state.rs"]
mod __slate_unit_xfrm6_state;
#[path = "xfrm6_input.rs"]
mod __slate_unit_xfrm6_input;
#[path = "xfrm6_output.rs"]
mod __slate_unit_xfrm6_output;
#[path = "xfrm6_protocol.rs"]
mod __slate_unit_xfrm6_protocol;
#[path = "netfilter.rs"]
mod __slate_unit_netfilter;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "syncookies.rs"]
mod __slate_unit_syncookies;
#[path = "calipso.rs"]
mod __slate_unit_calipso;
#[path = "ah6.rs"]
mod __slate_unit_ah6;
#[path = "esp6.rs"]
mod __slate_unit_esp6;
#[path = "nf_defrag_ipv6_hooks.rs"]
mod __slate_unit_nf_defrag_ipv6_hooks;
#[path = "nf_conntrack_reasm.rs"]
mod __slate_unit_nf_conntrack_reasm;
#[path = "ip6t_ipv6header.rs"]
mod __slate_unit_ip6t_ipv6header;
#[path = "addrconf_core.rs"]
mod __slate_unit_addrconf_core;
#[path = "exthdrs_core.rs"]
mod __slate_unit_exthdrs_core;
#[path = "ip6_checksum.rs"]
mod __slate_unit_ip6_checksum;
#[path = "ip6_icmp.rs"]
mod __slate_unit_ip6_icmp;
#[path = "output_core.rs"]
mod __slate_unit_output_core;
#[path = "protocol.rs"]
mod __slate_unit_protocol;
#[path = "ip6_offload.rs"]
mod __slate_unit_ip6_offload;
#[path = "exthdrs_offload.rs"]
mod __slate_unit_exthdrs_offload;
#[path = "inet6_hashtables.rs"]
mod __slate_unit_inet6_hashtables;
#[path = "mcast_snoop.rs"]
mod __slate_unit_mcast_snoop;
#[path = "ndisc_snoop.rs"]
mod __slate_unit_ndisc_snoop;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

