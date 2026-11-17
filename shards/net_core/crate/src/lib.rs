#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "sock.rs"]
mod __slate_unit_sock;
#[path = "skbuff.rs"]
mod __slate_unit_skbuff;
#[path = "datagram.rs"]
mod __slate_unit_datagram;
#[path = "stream.rs"]
mod __slate_unit_stream;
#[path = "scm.rs"]
mod __slate_unit_scm;
#[path = "gen_stats.rs"]
mod __slate_unit_gen_stats;
#[path = "gen_estimator.rs"]
mod __slate_unit_gen_estimator;
#[path = "net_namespace.rs"]
mod __slate_unit_net_namespace;
#[path = "secure_seq.rs"]
mod __slate_unit_secure_seq;
#[path = "flow_dissector.rs"]
mod __slate_unit_flow_dissector;
#[path = "sysctl_net_core.rs"]
mod __slate_unit_sysctl_net_core;
#[path = "dev.rs"]
mod __slate_unit_dev;
#[path = "dev_api.rs"]
mod __slate_unit_dev_api;
#[path = "dev_addr_lists.rs"]
mod __slate_unit_dev_addr_lists;
#[path = "dst.rs"]
mod __slate_unit_dst;
#[path = "netevent.rs"]
mod __slate_unit_netevent;
#[path = "neighbour.rs"]
mod __slate_unit_neighbour;
#[path = "rtnetlink.rs"]
mod __slate_unit_rtnetlink;
#[path = "utils.rs"]
mod __slate_unit_utils;
#[path = "link_watch.rs"]
mod __slate_unit_link_watch;
#[path = "filter.rs"]
mod __slate_unit_filter;
#[path = "sock_diag.rs"]
mod __slate_unit_sock_diag;
#[path = "dev_ioctl.rs"]
mod __slate_unit_dev_ioctl;
#[path = "tso.rs"]
mod __slate_unit_tso;
#[path = "sock_reuseport.rs"]
mod __slate_unit_sock_reuseport;
#[path = "fib_notifier.rs"]
mod __slate_unit_fib_notifier;
#[path = "xdp.rs"]
mod __slate_unit_xdp;
#[path = "flow_offload.rs"]
mod __slate_unit_flow_offload;
#[path = "gro.rs"]
mod __slate_unit_gro;
#[path = "netdev_genl.rs"]
mod __slate_unit_netdev_genl;
#[path = "netdev_genl_gen.rs"]
mod __slate_unit_netdev_genl_gen;
#[path = "netdev_work.rs"]
mod __slate_unit_netdev_work;
#[path = "gso.rs"]
mod __slate_unit_gso;
#[path = "net_sysfs.rs"]
mod __slate_unit_net_sysfs;
#[path = "hotdata.rs"]
mod __slate_unit_hotdata;
#[path = "netdev_config.rs"]
mod __slate_unit_netdev_config;
#[path = "netdev_rx_queue.rs"]
mod __slate_unit_netdev_rx_queue;
#[path = "netdev_queues.rs"]
mod __slate_unit_netdev_queues;
#[path = "page_pool.rs"]
mod __slate_unit_page_pool;
#[path = "page_pool_user.rs"]
mod __slate_unit_page_pool_user;
#[path = "net_procfs.rs"]
mod __slate_unit_net_procfs;
#[path = "netpoll.rs"]
mod __slate_unit_netpoll;
#[path = "fib_rules.rs"]
mod __slate_unit_fib_rules;
#[path = "net_traces.rs"]
mod __slate_unit_net_traces;
#[path = "selftests.rs"]
mod __slate_unit_selftests;
#[path = "ptp_classifier.rs"]
mod __slate_unit_ptp_classifier;
#[path = "netprio_cgroup.rs"]
mod __slate_unit_netprio_cgroup;
#[path = "netclassid_cgroup.rs"]
mod __slate_unit_netclassid_cgroup;
#[path = "dst_cache.rs"]
mod __slate_unit_dst_cache;
#[path = "gro_cells.rs"]
mod __slate_unit_gro_cells;
#[path = "failover.rs"]
mod __slate_unit_failover;
#[path = "devmem.rs"]
mod __slate_unit_devmem;

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

